// test_ane_limits2.cpp — Find how compile limit scales with kernel size
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "core/ane_runtime.h"
#include <ane_lm/common.h>

extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void  objc_autoreleasePoolPop(void*);

using namespace ane_lm;

static uint16_t* make_random_bf16(size_t numel) {
    uint16_t* w = (uint16_t*)malloc(numel * 2);
    for (size_t i = 0; i < numel; i++) {
        w[i] = f32_to_bf16(((float)(rand() % 1000) / 10000.0f) - 0.05f);
    }
    return w;
}

// Compile as many kernels of a given size as possible, return count
static int compile_until_failure(int out_dim, int in_dim, int max_attempts) {
    ane_set_persist_cache(false);
    
    size_t numel = (size_t)out_dim * in_dim;
    uint16_t* w = make_random_bf16(numel);
    
    std::vector<ANEKernel*> kernels;
    int count = 0;
    
    for (int i = 0; i < max_attempts; i++) {
        // Perturb a weight to get unique kernel hash
        w[i % numel] = f32_to_bf16((float)i * 0.0001f);
        
        ANEKernel* k = ane_compile_matmul(w, out_dim, in_dim);
        if (!k) break;
        kernels.push_back(k);
        count++;
    }
    
    // Cleanup
    for (auto* k : kernels) ane_free(k);
    free(w);
    return count;
}

// Same but with fused FFN kernels
static int compile_ffn_until_failure(int dim, int inter, int max_attempts) {
    ane_set_persist_cache(false);
    
    size_t numel_gu = (size_t)inter * dim;
    size_t numel_d = (size_t)dim * inter;
    uint16_t* gate = make_random_bf16(numel_gu);
    uint16_t* up = make_random_bf16(numel_gu);
    uint16_t* down = make_random_bf16(numel_d);
    
    std::vector<ANEKernel*> kernels;
    int count = 0;
    
    for (int i = 0; i < max_attempts; i++) {
        gate[i % numel_gu] = f32_to_bf16((float)i * 0.0001f);
        
        ANEKernel* k = ane_compile_fused_ffn(gate, up, down, dim, inter);
        if (!k) break;
        kernels.push_back(k);
        count++;
    }
    
    for (auto* k : kernels) ane_free(k);
    free(gate); free(up); free(down);
    return count;
}

// Test: compile the actual 4B model's kernel mix
static void test_4b_model_mix() {
    printf("\n=== TEST: Simulate 4B model compile pattern ===\n");
    ane_set_persist_cache(false);
    
    // Qwen3.5-4B: 32 layers, hidden=2560, inter=9216
    // 24 deltanet layers: fused_2(6144+4096, 2560) + matmul(2560, 4096) + fused_ffn(2560, 9216)
    // 8 full_attn layers: fused_3(5120+2560+2560, 2560) + matmul(2560, 5120) + fused_ffn(2560, 9216)
    // LM head: 16 × matmul(16384, 2560)
    
    int dim = 2560;
    int inter = 9216;
    
    // Deltanet dims
    int lin_qkv = 6144;
    int lin_val = 4096;
    // Full attn dims (with gate)
    int full_q = 5120;
    int full_kv = 2560;
    int full_out = 5120;
    
    std::vector<ANEKernel*> all_kernels;
    int layer_types[] = { // 0=linear, 1=full; pattern from config
        0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1,
        0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1
    };
    
    int total_layers = 32;
    bool failed = false;
    
    for (int L = 0; L < total_layers && !failed; L++) {
        bool is_full = (layer_types[L] == 1);
        
        // 1. First projection
        ANEKernel* fp = nullptr;
        if (!is_full) {
            uint16_t* wa = make_random_bf16((size_t)lin_qkv * dim);
            uint16_t* wb = make_random_bf16((size_t)lin_val * dim);
            wa[L] = f32_to_bf16((float)L * 0.01f);
            fp = ane_compile_fused_2(wa, lin_qkv, wb, lin_val, dim);
            free(wa); free(wb);
        } else {
            uint16_t* wa = make_random_bf16((size_t)full_q * dim);
            uint16_t* wb = make_random_bf16((size_t)full_kv * dim);
            uint16_t* wc = make_random_bf16((size_t)full_kv * dim);
            wa[L] = f32_to_bf16((float)L * 0.01f);
            fp = ane_compile_fused_3(wa, full_q, wb, full_kv, wc, full_kv, dim);
            free(wa); free(wb); free(wc);
        }
        if (!fp) {
            printf("  Layer %d/%d: first_proj FAILED (compiles=%d)\n", L+1, total_layers, ane_compile_count());
            failed = true; break;
        }
        all_kernels.push_back(fp);
        
        // 2. O projection
        int o_in = is_full ? full_out : lin_val;
        uint16_t* wo = make_random_bf16((size_t)dim * o_in);
        wo[L] = f32_to_bf16((float)(L+100) * 0.01f);
        ANEKernel* op = ane_compile_matmul(wo, dim, o_in);
        free(wo);
        if (!op) {
            printf("  Layer %d/%d: o_proj FAILED (compiles=%d)\n", L+1, total_layers, ane_compile_count());
            failed = true; break;
        }
        all_kernels.push_back(op);
        
        // 3. Fused FFN (single kernel — we proved 135MB works)
        uint16_t* gate = make_random_bf16((size_t)inter * dim);
        uint16_t* up = make_random_bf16((size_t)inter * dim);
        uint16_t* down = make_random_bf16((size_t)dim * inter);
        gate[L] = f32_to_bf16((float)(L+200) * 0.01f);
        ANEKernel* ffn = ane_compile_fused_ffn(gate, up, down, dim, inter);
        free(gate); free(up); free(down);
        if (!ffn) {
            printf("  Layer %d/%d: fused_ffn FAILED (compiles=%d)\n", L+1, total_layers, ane_compile_count());
            failed = true; break;
        }
        all_kernels.push_back(ffn);
        
        printf("  Layer %d/%d done (compiles=%d, %s)\n",
               L+1, total_layers, ane_compile_count(),
               is_full ? "full_attn" : "deltanet");
    }
    
    if (!failed) {
        printf("  All 32 layers compiled! compiles=%d\n", ane_compile_count());
        
        // Now try LM head
        printf("  Compiling LM head (16 chunks of 16384x2560)...\n");
        for (int c = 0; c < 16 && !failed; c++) {
            int rows = (c < 15) ? 16384 : (248320 - 15 * 16384);
            uint16_t* w = make_random_bf16((size_t)rows * dim);
            w[c] = f32_to_bf16((float)(c+300) * 0.01f);
            ANEKernel* k = ane_compile_matmul(w, rows, dim);
            free(w);
            if (!k) {
                printf("  LM head chunk %d/16 FAILED (compiles=%d)\n", c+1, ane_compile_count());
                failed = true; break;
            }
            all_kernels.push_back(k);
            if ((c+1) % 4 == 0)
                printf("  LM head chunk %d/16 OK (compiles=%d)\n", c+1, ane_compile_count());
        }
        
        if (!failed)
            printf("  FULL MODEL COMPILED! Total compiles=%d\n", ane_compile_count());
    }
    
    printf("\n  SUMMARY: compiled %d kernels, ane_compile_count()=%d\n",
           (int)all_kernels.size(), ane_compile_count());
    
    for (auto* k : all_kernels) ane_free(k);
}

int main(int argc, char* argv[]) {
    void* pool = objc_autoreleasePoolPush();
    srand(42);
    g_verbose = false;  // quiet ANE init
    
    ane_init();
    if (!ane_available()) {
        fprintf(stderr, "ANE not available!\n");
        return 1;
    }
    
    if (argc >= 2 && strcmp(argv[1], "sizes") == 0) {
        // Test compile limit at different kernel sizes
        printf("=== Compile count limit vs kernel size ===\n");
        struct { int out; int in; const char* label; } sizes[] = {
            {64, 64, "8KB"},
            {256, 256, "128KB"},
            {1024, 1024, "2MB"},
            {2560, 2560, "12.5MB"},
            {9216, 2560, "45MB"},
            {16384, 2560, "80MB"},
        };
        
        for (auto& s : sizes) {
            // Need a fresh process for each size test — the limit is per-process cumulative
            // So we test ONE size per invocation
            printf("  [%s] %dx%d: ", s.label, s.out, s.in);
            fflush(stdout);
            int n = compile_until_failure(s.out, s.in, 300);
            printf("%d compiles before failure (ane_compile_count=%d)\n", n, ane_compile_count());
            // Can't continue testing other sizes — compiler state is polluted
            break;
        }
    } else if (argc >= 2 && strcmp(argv[1], "ffn") == 0) {
        // How many fused FFN kernels can we compile?
        printf("=== Fused FFN compile limit (dim=2560, inter=9216, 135MB each) ===\n");
        int n = compile_ffn_until_failure(2560, 9216, 200);
        printf("  Compiled %d fused FFN kernels before failure\n", n);
    } else if (argc >= 2 && strcmp(argv[1], "mix") == 0) {
        test_4b_model_mix();
    } else {
        printf("Usage:\n");
        printf("  test_ane_limits2 sizes   — compile limit for one kernel size\n");
        printf("  test_ane_limits2 ffn     — how many 135MB FFN kernels fit\n");
        printf("  test_ane_limits2 mix     — simulate full 4B model compile\n");
    }
    
    objc_autoreleasePoolPop(pool);
    return 0;
}
