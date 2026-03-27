// test_dynamic.cpp — Test dynamic-weight matmul and FFN on ANE
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "core/ane_runtime.h"
#include <ane_lm/common.h>

extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void  objc_autoreleasePoolPop(void*);

using namespace ane_lm;

static uint16_t* make_bf16(size_t numel, int seed) {
    uint16_t* w = (uint16_t*)malloc(numel * 2);
    srand(seed);
    for (size_t i = 0; i < numel; i++)
        w[i] = f32_to_bf16(((float)(rand() % 2000) / 10000.0f) - 0.1f);
    return w;
}

static uint16_t* bf16_to_fp16_array(const uint16_t* bf16, size_t n) {
    uint16_t* fp16 = (uint16_t*)malloc(n * 2);
    bf16_to_f16_vec(fp16, bf16, n);
    return fp16;
}

// CPU reference: y = W @ x  (W is [out, in], x is [in])
static void cpu_matvec(float* y, const uint16_t* W_bf16, const float* x, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        float sum = 0;
        for (int i = 0; i < in_dim; i++) {
            sum += bf16_to_f32(W_bf16[o * in_dim + i]) * x[i];
        }
        y[o] = sum;
    }
}

static float max_abs_diff(const float* a, const float* b, int n) {
    float max_d = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > max_d) max_d = d;
    }
    return max_d;
}

static float max_rel_diff(const float* a, const float* b, int n) {
    float max_d = 0;
    for (int i = 0; i < n; i++) {
        float denom = fmaxf(fabsf(a[i]), fabsf(b[i]));
        if (denom < 1e-8f) continue;
        float d = fabsf(a[i] - b[i]) / denom;
        if (d > max_d) max_d = d;
    }
    return max_d;
}

static void test_dynamic_matmul() {
    printf("=== Test 1: Dynamic matmul (small) ===\n");
    int in_dim = 64, out_dim = 64;
    
    printf("  Compiling dynamic matmul [%d, %d]... ", out_dim, in_dim);
    fflush(stdout);
    Timer t;
    ANEKernel* k = ane_compile_dynamic_matmul(out_dim, in_dim);
    if (!k) {
        printf("FAILED (compile)\n");
        return;
    }
    printf("OK (%.1f ms)\n", t.elapsed_ms());
    
    // Test with random weights and input
    uint16_t* W_bf16 = make_bf16((size_t)out_dim * in_dim, 42);
    uint16_t* W_fp16 = bf16_to_fp16_array(W_bf16, (size_t)out_dim * in_dim);
    float* x = (float*)calloc(in_dim, sizeof(float));
    for (int i = 0; i < in_dim; i++) x[i] = ((float)(i + 1)) / in_dim;
    
    float* y_ane = (float*)calloc(out_dim, sizeof(float));
    float* y_cpu = (float*)calloc(out_dim, sizeof(float));
    
    printf("  Evaluating... ");
    fflush(stdout);
    if (!ane_dynamic_matvec(k, y_ane, x, W_fp16, in_dim, out_dim)) {
        printf("FAILED (eval)\n");
    } else {
        cpu_matvec(y_cpu, W_bf16, x, out_dim, in_dim);
        float mad = max_abs_diff(y_ane, y_cpu, out_dim);
        float mrd = max_rel_diff(y_ane, y_cpu, out_dim);
        printf("OK — max_abs_diff=%.6f, max_rel_diff=%.4f\n", mad, mrd);
        printf("  First 5 ANE: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
               y_ane[0], y_ane[1], y_ane[2], y_ane[3], y_ane[4]);
        printf("  First 5 CPU: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
               y_cpu[0], y_cpu[1], y_cpu[2], y_cpu[3], y_cpu[4]);
    }
    
    // Test with DIFFERENT weights (reuse same compiled kernel)
    printf("  Re-evaluating with different weights... ");
    uint16_t* W2_bf16 = make_bf16((size_t)out_dim * in_dim, 123);
    uint16_t* W2_fp16 = bf16_to_fp16_array(W2_bf16, (size_t)out_dim * in_dim);
    float* y_ane2 = (float*)calloc(out_dim, sizeof(float));
    float* y_cpu2 = (float*)calloc(out_dim, sizeof(float));
    
    if (!ane_dynamic_matvec(k, y_ane2, x, W2_fp16, in_dim, out_dim)) {
        printf("FAILED (eval)\n");
    } else {
        cpu_matvec(y_cpu2, W2_bf16, x, out_dim, in_dim);
        float mad = max_abs_diff(y_ane2, y_cpu2, out_dim);
        printf("OK — max_abs_diff=%.6f (different output confirms weights updated)\n", mad);
        printf("  y1[0]=%.4f, y2[0]=%.4f (should differ)\n", y_ane[0], y_ane2[0]);
    }
    
    ane_free(k);
    free(W_bf16); free(W_fp16); free(W2_bf16); free(W2_fp16);
    free(x); free(y_ane); free(y_cpu); free(y_ane2); free(y_cpu2);
}

static void test_dynamic_matmul_large() {
    printf("\n=== Test 2: Dynamic matmul at 4B dims ===\n");
    
    struct TestCase { int out; int in; const char* desc; };
    TestCase cases[] = {
        {2560, 2560, "o_proj size"},
        {9216, 2560, "gate/up_proj size"},
        {2560, 9216, "down_proj size"},
        {16384, 2560, "LM head chunk"},
    };
    
    for (auto& tc : cases) {
        printf("  Dynamic matmul [%d, %d] (%s, %.1f MB): ",
               tc.out, tc.in, tc.desc,
               (float)tc.out * tc.in * 2 / (1024*1024));
        fflush(stdout);
        
        Timer t;
        ANEKernel* k = ane_compile_dynamic_matmul(tc.out, tc.in);
        if (!k) {
            printf("FAILED (compile, %.0f ms)\n", t.elapsed_ms());
            continue;
        }
        double compile_ms = t.elapsed_ms();
        printf("compiled (%.0f ms), ", compile_ms);
        fflush(stdout);
        
        // Quick correctness check
        uint16_t* W = make_bf16((size_t)tc.out * tc.in, 42);
        uint16_t* W_fp16 = bf16_to_fp16_array(W, (size_t)tc.out * tc.in);
        float* x = (float*)calloc(tc.in, sizeof(float));
        x[0] = 1.0f;
        float* y = (float*)calloc(tc.out, sizeof(float));
        
        // Warmup
        ane_dynamic_matvec(k, y, x, W_fp16, tc.in, tc.out);
        
        // Time eval
        t.reset();
        int N = 5;
        for (int i = 0; i < N; i++)
            ane_dynamic_matvec(k, y, x, W_fp16, tc.in, tc.out);
        double eval_ms = t.elapsed_ms() / N;
        printf("eval=%.1f ms\n", eval_ms);
        
        ane_free(k);
        free(W); free(W_fp16); free(x); free(y);
    }
}

static void test_dynamic_ffn() {
    printf("\n=== Test 3: Dynamic FFN ===\n");
    
    // Small test first
    int dim = 64, inter = 128;
    printf("  Compiling dynamic FFN (dim=%d, inter=%d)... ", dim, inter);
    fflush(stdout);
    Timer t;
    ANEKernel* k = ane_compile_dynamic_ffn(dim, inter);
    if (!k) {
        printf("FAILED\n");
    } else {
        printf("OK (%.1f ms)\n", t.elapsed_ms());
        ane_free(k);
    }
    
    // 4B dims
    dim = 2560; inter = 9216;
    printf("  Compiling dynamic FFN (dim=%d, inter=%d, 4B size)... ", dim, inter);
    fflush(stdout);
    t.reset();
    k = ane_compile_dynamic_ffn(dim, inter);
    if (!k) {
        printf("FAILED (%.1f ms)\n", t.elapsed_ms());
    } else {
        double compile_ms = t.elapsed_ms();
        printf("OK (%.0f ms)\n", compile_ms);
        
        // Test eval with weight data
        uint16_t* g = make_bf16((size_t)inter * dim, 10);
        uint16_t* u = make_bf16((size_t)inter * dim, 11);
        uint16_t* d = make_bf16((size_t)dim * inter, 12);
        uint16_t* g16 = bf16_to_fp16_array(g, (size_t)inter * dim);
        uint16_t* u16 = bf16_to_fp16_array(u, (size_t)inter * dim);
        uint16_t* d16 = bf16_to_fp16_array(d, (size_t)dim * inter);
        
        float* x = (float*)calloc(dim, sizeof(float));
        x[0] = 1.0f;
        float* out = (float*)calloc(dim, sizeof(float));
        
        // Warmup
        ane_dynamic_ffn_eval(k, out, x, g16, u16, d16, dim, inter);
        
        // Time
        int N = 10;
        t.reset();
        for (int i = 0; i < N; i++)
            ane_dynamic_ffn_eval(k, out, x, g16, u16, d16, dim, inter);
        double eval_ms = t.elapsed_ms() / N;
        printf("  Eval (avg %d): %.1f ms (includes %.1f MB memcpy)\n",
               N, eval_ms, (float)(2 * inter * dim + dim * inter) * 2 / (1024*1024));
        printf("  Projected 32-layer overhead: %.0f ms → %.1f tok/s\n",
               32 * eval_ms, 1000.0 / (32 * eval_ms));
        
        ane_free(k);
        free(g); free(u); free(d); free(g16); free(u16); free(d16);
        free(x); free(out);
    }
}

static void test_loaded_budget() {
    printf("\n=== Test 4: Does dynamic matmul use less loaded memory? ===\n");
    printf("  (Dynamic kernels have no baked weights → much smaller loaded footprint)\n");
    
    ane_set_persist_cache(false);
    
    // How many dynamic matmul kernels can we load simultaneously?
    int out_dim = 9216, in_dim = 2560;
    printf("  Loading dynamic matmul [%d, %d] kernels until failure...\n", out_dim, in_dim);
    
    std::vector<ANEKernel*> kernels;
    for (int i = 0; i < 200; i++) {
        ANEKernel* k = ane_compile_dynamic_matmul(out_dim, in_dim);
        if (!k) {
            printf("  LIMIT: %d dynamic matmul kernels loaded (compiles=%d)\n",
                   (int)kernels.size(), ane_compile_count());
            break;
        }
        kernels.push_back(k);
        if ((i + 1) % 20 == 0)
            printf("  %d loaded OK (compiles=%d)\n", i + 1, ane_compile_count());
    }
    
    for (auto* k : kernels) ane_free(k);
}

int main(int argc, char* argv[]) {
    void* pool = objc_autoreleasePoolPush();
    g_verbose = false;
    ane_init();
    if (!ane_available()) { fprintf(stderr, "ANE not available!\n"); return 1; }
    
    if (argc >= 2 && strcmp(argv[1], "matmul") == 0) {
        test_dynamic_matmul();
        test_dynamic_matmul_large();
    } else if (argc >= 2 && strcmp(argv[1], "ffn") == 0) {
        test_dynamic_ffn();
    } else if (argc >= 2 && strcmp(argv[1], "budget") == 0) {
        test_loaded_budget();
    } else if (argc >= 2 && strcmp(argv[1], "all") == 0) {
        test_dynamic_matmul();
        test_dynamic_matmul_large();
        test_dynamic_ffn();
        test_loaded_budget();
    } else {
        printf("Usage: test_dynamic [matmul|ffn|budget|all]\n");
    }
    
    printf("\nFinal: compiles=%d, cache_loads=%d\n", ane_compile_count(), ane_cache_loads());
    objc_autoreleasePoolPop(pool);
    return 0;
}
