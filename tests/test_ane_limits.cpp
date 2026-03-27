// test_ane_limits.cpp — Empirically find ANE compile count limit and max kernel size
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "core/ane_runtime.h"
#include <ane_lm/common.h>

extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void  objc_autoreleasePoolPop(void*);

using namespace ane_lm;

// Generate random bf16 weights
static uint16_t* make_random_bf16(int numel) {
    uint16_t* w = (uint16_t*)malloc((size_t)numel * 2);
    for (int i = 0; i < numel; i++) {
        float v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        w[i] = f32_to_bf16(v);
    }
    return w;
}

// Test 1: Find max compile count with small kernels
static void test_compile_count_limit() {
    printf("\n=== TEST 1: Compile count limit (small 64x64 matmul kernels) ===\n");
    
    // Clear cache to get fresh compiles
    printf("NOTE: Using --no-cache mode to force fresh compiles\n");
    ane_set_persist_cache(false);
    
    int in_dim = 64;
    int out_dim = 64;
    uint16_t* w = make_random_bf16(in_dim * out_dim);
    
    std::vector<ANEKernel*> kernels;
    int max_compiles = 300;
    
    for (int i = 0; i < max_compiles; i++) {
        // Slightly different weights each time so we get unique kernels
        w[0] = f32_to_bf16((float)i * 0.001f);
        
        ANEKernel* k = ane_compile_matmul(w, out_dim, in_dim);
        if (!k) {
            printf("  COMPILE FAILED at kernel #%d (compile_count=%d, cache_loads=%d)\n",
                   i + 1, ane_compile_count(), ane_cache_loads());
            break;
        }
        kernels.push_back(k);
        
        if ((i + 1) % 10 == 0 || i < 5) {
            printf("  Compiled kernel #%d OK (compile_count=%d)\n", i + 1, ane_compile_count());
        }
    }
    
    printf("  MAX COMPILES BEFORE FAILURE: %d\n", (int)kernels.size());
    printf("  ane_compile_count()=%d, ane_cache_loads()=%d\n",
           ane_compile_count(), ane_cache_loads());
    
    // Cleanup
    for (auto* k : kernels) ane_free(k);
    free(w);
}

// Test 2: Find max kernel size (matmul)
static void test_max_kernel_size_matmul() {
    printf("\n=== TEST 2: Max matmul kernel size ===\n");
    ane_set_persist_cache(true);  // Use cache to not waste compile budget
    
    struct TestCase { int out_dim; int in_dim; };
    std::vector<TestCase> cases = {
        // Small baseline
        {1024, 1024},     // 2MB
        {2048, 1024},     // 4MB
        {4096, 1024},     // 8MB
        {8192, 1024},     // 16MB
        {16384, 1024},    // 32MB
        {16384, 2560},    // 80MB
        {32768, 1024},    // 64MB
        {32768, 2560},    // 160MB
        // Targeted for 4B model dims
        {2560, 2560},     // 12.5MB
        {4096, 2560},     // 20MB
        {9216, 2560},     // 45MB - single gate/up proj
        {2560, 9216},     // 45MB - down proj
    };
    
    for (auto& tc : cases) {
        size_t bytes = (size_t)tc.out_dim * tc.in_dim * 2;
        printf("  matmul [%d, %d] (%.1f MB weights): ",
               tc.out_dim, tc.in_dim, bytes / (1024.0 * 1024.0));
        fflush(stdout);
        
        uint16_t* w = make_random_bf16(tc.out_dim * tc.in_dim);
        
        Timer t;
        ANEKernel* k = ane_compile_matmul(w, tc.out_dim, tc.in_dim);
        double ms = t.elapsed_ms();
        
        if (k) {
            printf("OK (%.0f ms, compiles=%d)\n", ms, ane_compile_count());
            ane_free(k);
        } else {
            printf("FAILED (%.0f ms, compiles=%d)\n", ms, ane_compile_count());
        }
        free(w);
    }
}

// Test 3: Find max fused FFN size
static void test_max_fused_ffn_size() {
    printf("\n=== TEST 3: Max fused FFN kernel size ===\n");
    
    struct TestCase { int dim; int inter; };
    std::vector<TestCase> cases = {
        {1024, 3584},    // 0.8B — ~22MB total
        {1024, 4096},    // ~24MB
        {1024, 8192},    // ~48MB
        {2048, 4096},    // ~48MB
        {2560, 3072},    // ~47MB - 3 * 3072 * 2560 * 2
        {2560, 4096},    // ~60MB
        {2560, 6144},    // ~94MB
        {2560, 9216},    // ~141MB - actual 4B size
    };
    
    for (auto& tc : cases) {
        size_t total = (size_t)3 * tc.inter * tc.dim * 2;
        printf("  fused_ffn dim=%d inter=%d (%.1f MB total weights): ",
               tc.dim, tc.inter, total / (1024.0 * 1024.0));
        fflush(stdout);
        
        uint16_t* gate = make_random_bf16(tc.inter * tc.dim);
        uint16_t* up = make_random_bf16(tc.inter * tc.dim);
        uint16_t* down = make_random_bf16(tc.dim * tc.inter);
        
        Timer t;
        ANEKernel* k = ane_compile_fused_ffn(gate, up, down, tc.dim, tc.inter);
        double ms = t.elapsed_ms();
        
        if (k) {
            printf("OK (%.0f ms, compiles=%d)\n", ms, ane_compile_count());
            ane_free(k);
        } else {
            printf("FAILED (%.0f ms, compiles=%d)\n", ms, ane_compile_count());
        }
        free(gate); free(up); free(down);
    }
}

// Test 4: Find max chunked FFN size
static void test_max_chunked_ffn() {
    printf("\n=== TEST 4: Chunked FFN for 4B dims (dim=2560, inter=9216) ===\n");
    
    int dim = 2560;
    int inter = 9216;
    
    uint16_t* gate = make_random_bf16(inter * dim);
    uint16_t* up = make_random_bf16(inter * dim);
    uint16_t* down = make_random_bf16(dim * inter);
    
    // Try different chunk counts
    int chunk_counts[] = {2, 3, 4, 6, 8, 12};
    for (int nc : chunk_counts) {
        if (inter % nc != 0) {
            printf("  %d chunks: SKIP (inter=%d not divisible)\n", nc, inter);
            continue;
        }
        int chunk_inter = inter / nc;
        size_t chunk_bytes = (size_t)(2 * chunk_inter * dim + dim * chunk_inter) * 2;
        printf("  %d chunks (chunk_inter=%d, %.1f MB/chunk): ",
               nc, chunk_inter, chunk_bytes / (1024.0 * 1024.0));
        fflush(stdout);
        
        ChunkedFFN cffn = {};
        Timer t;
        bool ok = ane_compile_chunked_ffn(&cffn, gate, up, down, dim, inter, nc);
        double ms = t.elapsed_ms();
        
        if (ok) {
            printf("OK (%.0f ms, compiles=%d)\n", ms, ane_compile_count());
            ane_free_chunked_ffn(&cffn);
        } else {
            printf("FAILED at chunk (%.0f ms, compiles=%d)\n", ms, ane_compile_count());
        }
    }
    
    free(gate); free(up); free(down);
}

// Test 5: fused_2 and fused_3 size limits
static void test_fused_proj_sizes() {
    printf("\n=== TEST 5: Fused projection kernel sizes (for attention) ===\n");
    
    // Qwen3.5-4B attention dims:
    // DeltaNet: fused_2(lin_qkv_dim=6144, lin_total_val=4096, hidden=2560)
    // Full: fused_3(full_q_dim=5120, full_kv_dim=2560, full_kv_dim=2560, hidden=2560)
    
    struct F2Case { int a_out; int b_out; int in_dim; const char* desc; };
    std::vector<F2Case> f2_cases = {
        {6144, 4096, 2560, "4B deltanet: qkv+z"},
        {4096, 2048, 2560, "smaller"},
        {8192, 4096, 2560, "larger"},
    };
    
    for (auto& tc : f2_cases) {
        size_t bytes = (size_t)(tc.a_out + tc.b_out) * tc.in_dim * 2;
        printf("  fused_2 [%d+%d, %d] (%.1f MB): %s — ",
               tc.a_out, tc.b_out, tc.in_dim, bytes / (1024.0 * 1024.0), tc.desc);
        fflush(stdout);
        
        uint16_t* wa = make_random_bf16(tc.a_out * tc.in_dim);
        uint16_t* wb = make_random_bf16(tc.b_out * tc.in_dim);
        
        Timer t;
        ANEKernel* k = ane_compile_fused_2(wa, tc.a_out, wb, tc.b_out, tc.in_dim);
        double ms = t.elapsed_ms();
        
        if (k) {
            printf("OK (%.0f ms, compiles=%d)\n", ms, ane_compile_count());
            ane_free(k);
        } else {
            printf("FAILED (%.0f ms, compiles=%d)\n", ms, ane_compile_count());
        }
        free(wa); free(wb);
    }
    
    struct F3Case { int a; int b; int c; int in_dim; const char* desc; };
    std::vector<F3Case> f3_cases = {
        {5120, 2560, 2560, 2560, "4B full_attn: q+k+v (q has gate)"},
        {2560, 2560, 2560, 2560, "4B full_attn: q+k+v (no gate)"},
    };
    
    for (auto& tc : f3_cases) {
        size_t bytes = (size_t)(tc.a + tc.b + tc.c) * tc.in_dim * 2;
        printf("  fused_3 [%d+%d+%d, %d] (%.1f MB): %s — ",
               tc.a, tc.b, tc.c, tc.in_dim, bytes / (1024.0 * 1024.0), tc.desc);
        fflush(stdout);
        
        uint16_t* wa = make_random_bf16(tc.a * tc.in_dim);
        uint16_t* wb = make_random_bf16(tc.b * tc.in_dim);
        uint16_t* wc = make_random_bf16(tc.c * tc.in_dim);
        
        Timer t;
        ANEKernel* k = ane_compile_fused_3(wa, tc.a, wb, tc.b, wc, tc.c, tc.in_dim);
        double ms = t.elapsed_ms();
        
        if (k) {
            printf("OK (%.0f ms, compiles=%d)\n", ms, ane_compile_count());
            ane_free(k);
        } else {
            printf("FAILED (%.0f ms, compiles=%d)\n", ms, ane_compile_count());
        }
        free(wa); free(wb); free(wc);
    }
}

static void print_usage() {
    printf("Usage: test_ane_limits [test_number]\n");
    printf("  1 = Compile count limit (WARNING: takes minutes, no cache)\n");
    printf("  2 = Max matmul kernel size\n");
    printf("  3 = Max fused FFN kernel size\n");
    printf("  4 = Chunked FFN for 4B dims\n");
    printf("  5 = Fused projection sizes\n");
    printf("  all = Run tests 2-5 (skip compile count test)\n");
    printf("  count = Just run compile count test\n");
}

int main(int argc, char* argv[]) {
    void* pool = objc_autoreleasePoolPush();
    srand(42);
    g_verbose = true;
    
    ane_init();
    if (!ane_available()) {
        fprintf(stderr, "ANE not available!\n");
        return 1;
    }
    printf("ANE initialized OK\n");
    
    if (argc < 2) {
        print_usage();
        objc_autoreleasePoolPop(pool);
        return 1;
    }
    
    const char* test = argv[1];
    
    if (strcmp(test, "1") == 0 || strcmp(test, "count") == 0) {
        test_compile_count_limit();
    } else if (strcmp(test, "2") == 0) {
        test_max_kernel_size_matmul();
    } else if (strcmp(test, "3") == 0) {
        test_max_fused_ffn_size();
    } else if (strcmp(test, "4") == 0) {
        test_max_chunked_ffn();
    } else if (strcmp(test, "5") == 0) {
        test_fused_proj_sizes();
    } else if (strcmp(test, "all") == 0) {
        test_max_kernel_size_matmul();
        test_max_fused_ffn_size();
        test_fused_proj_sizes();
        test_max_chunked_ffn();
    } else {
        print_usage();
        objc_autoreleasePoolPop(pool);
        return 1;
    }
    
    printf("\nFinal state: compile_count=%d, cache_loads=%d\n",
           ane_compile_count(), ane_cache_loads());
    
    objc_autoreleasePoolPop(pool);
    return 0;
}
