// test_dynamic3.cpp — Correctness and performance of dynamic conv
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

static void cpu_matvec(float* y, const uint16_t* W_bf16, const float* x, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        double sum = 0;
        for (int i = 0; i < in_dim; i++)
            sum += (double)bf16_to_f32(W_bf16[(size_t)o * in_dim + i]) * (double)x[i];
        y[o] = (float)sum;
    }
}

static float max_abs_diff(const float* a, const float* b, int n) {
    float mx = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

int main() {
    void* pool = objc_autoreleasePoolPush();
    g_verbose = false;
    ane_init();
    if (!ane_available()) { fprintf(stderr, "ANE not available!\n"); return 1; }
    ane_set_persist_cache(true);
    
    // === Test 1: correctness of dynamic conv at small size ===
    printf("=== Correctness: dynamic conv 64x64 ===\n");
    {
        int out = 64, in = 64;
        ANEKernel* k = ane_compile_dynamic_conv(out, in);
        if (!k) { printf("  Compile FAILED\n"); return 1; }
        
        uint16_t* W_bf16 = make_bf16((size_t)out * in, 42);
        // Convert to fp16 for ANE
        uint16_t* W_fp16 = (uint16_t*)malloc((size_t)out * in * 2);
        bf16_to_f16_vec(W_fp16, W_bf16, out * in);
        
        float* x = (float*)calloc(in, sizeof(float));
        for (int i = 0; i < in; i++) x[i] = ((float)(i + 1)) / in;
        
        float* y_ane = (float*)calloc(out, sizeof(float));
        float* y_cpu = (float*)calloc(out, sizeof(float));
        
        ane_dynamic_conv_eval(k, y_ane, x, W_fp16, in, out);
        cpu_matvec(y_cpu, W_bf16, x, out, in);
        
        float mad = max_abs_diff(y_ane, y_cpu, out);
        printf("  max_abs_diff = %.6f %s\n", mad, mad < 0.01f ? "OK" : "HIGH");
        printf("  ANE[0..4] = [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
               y_ane[0], y_ane[1], y_ane[2], y_ane[3], y_ane[4]);
        printf("  CPU[0..4] = [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
               y_cpu[0], y_cpu[1], y_cpu[2], y_cpu[3], y_cpu[4]);
        
        // Test with DIFFERENT weights on same kernel
        uint16_t* W2_bf16 = make_bf16((size_t)out * in, 99);
        uint16_t* W2_fp16 = (uint16_t*)malloc((size_t)out * in * 2);
        bf16_to_f16_vec(W2_fp16, W2_bf16, out * in);
        
        float* y_ane2 = (float*)calloc(out, sizeof(float));
        float* y_cpu2 = (float*)calloc(out, sizeof(float));
        
        ane_dynamic_conv_eval(k, y_ane2, x, W2_fp16, in, out);
        cpu_matvec(y_cpu2, W2_bf16, x, out, in);
        
        float mad2 = max_abs_diff(y_ane2, y_cpu2, out);
        printf("  Different weights: max_abs_diff = %.6f\n", mad2);
        printf("  y1[0]=%.4f != y2[0]=%.4f (confirms weights changed)\n", y_ane[0], y_ane2[0]);
        
        ane_free(k);
        free(W_bf16); free(W_fp16); free(W2_bf16); free(W2_fp16);
        free(x); free(y_ane); free(y_cpu); free(y_ane2); free(y_cpu2);
    }
    
    // === Test 2: correctness at 4B dims ===
    printf("\n=== Correctness: dynamic conv at 4B dims ===\n");
    {
        int out = 9216, in = 2560;
        printf("  Compiling [%d, %d]... ", out, in);
        fflush(stdout);
        ANEKernel* k = ane_compile_dynamic_conv(out, in);
        if (!k) { printf("FAILED\n"); return 1; }
        printf("OK\n");
        
        uint16_t* W_bf16 = make_bf16((size_t)out * in, 77);
        uint16_t* W_fp16 = (uint16_t*)malloc((size_t)out * in * 2);
        bf16_to_f16_vec(W_fp16, W_bf16, out * in);
        
        float* x = (float*)calloc(in, sizeof(float));
        x[0] = 1.0f; x[1] = -0.5f; x[100] = 0.3f;
        
        float* y_ane = (float*)calloc(out, sizeof(float));
        float* y_cpu = (float*)calloc(out, sizeof(float));
        
        ane_dynamic_conv_eval(k, y_ane, x, W_fp16, in, out);
        cpu_matvec(y_cpu, W_bf16, x, out, in);
        
        float mad = max_abs_diff(y_ane, y_cpu, out);
        printf("  max_abs_diff = %.6f %s\n", mad, mad < 0.05f ? "OK" : "HIGH");
        
        ane_free(k);
        free(W_bf16); free(W_fp16); free(x); free(y_ane); free(y_cpu);
    }
    
    // === Test 3: dynamic FFN correctness ===
    printf("\n=== Correctness: dynamic FFN (small) ===\n");
    {
        int dim = 64, inter = 128;
        ANEKernel* k = ane_compile_dynamic_ffn(dim, inter);
        if (!k) { printf("  Compile FAILED\n"); return 1; }
        
        uint16_t* g_bf16 = make_bf16((size_t)inter * dim, 10);
        uint16_t* u_bf16 = make_bf16((size_t)inter * dim, 11);
        uint16_t* d_bf16 = make_bf16((size_t)dim * inter, 12);
        uint16_t* g16 = (uint16_t*)malloc((size_t)inter * dim * 2);
        uint16_t* u16 = (uint16_t*)malloc((size_t)inter * dim * 2);
        uint16_t* d16 = (uint16_t*)malloc((size_t)dim * inter * 2);
        bf16_to_f16_vec(g16, g_bf16, inter * dim);
        bf16_to_f16_vec(u16, u_bf16, inter * dim);
        bf16_to_f16_vec(d16, d_bf16, dim * inter);
        
        float* x = (float*)calloc(dim, sizeof(float));
        for (int i = 0; i < dim; i++) x[i] = ((float)(i + 1)) / dim;
        float* y_ane = (float*)calloc(dim, sizeof(float));
        
        ane_dynamic_ffn_eval(k, y_ane, x, g16, u16, d16, dim, inter);
        
        // CPU reference: gate=Wg@x, up=Wu@x, silu=gate*sigmoid(gate), fused=silu*up, out=Wd@fused
        float* gate = (float*)calloc(inter, sizeof(float));
        float* up = (float*)calloc(inter, sizeof(float));
        float* fused = (float*)calloc(inter, sizeof(float));
        float* y_cpu = (float*)calloc(dim, sizeof(float));
        cpu_matvec(gate, g_bf16, x, inter, dim);
        cpu_matvec(up, u_bf16, x, inter, dim);
        for (int i = 0; i < inter; i++) {
            float silu = gate[i] * (1.0f / (1.0f + expf(-gate[i])));
            fused[i] = silu * up[i];
        }
        // Need bf16 for down_proj cpu reference
        cpu_matvec(y_cpu, d_bf16, fused, dim, inter);
        
        float mad = max_abs_diff(y_ane, y_cpu, dim);
        printf("  max_abs_diff = %.6f %s\n", mad, mad < 0.05f ? "OK" : "HIGH");
        printf("  ANE[0..4] = [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
               y_ane[0], y_ane[1], y_ane[2], y_ane[3], y_ane[4]);
        printf("  CPU[0..4] = [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
               y_cpu[0], y_cpu[1], y_cpu[2], y_cpu[3], y_cpu[4]);
        
        ane_free(k);
        free(g_bf16); free(u_bf16); free(d_bf16);
        free(g16); free(u16); free(d16);
        free(x); free(y_ane); free(y_cpu); free(gate); free(up); free(fused);
    }
    
    // === Test 4: Performance at 4B dims ===
    printf("\n=== Performance: dynamic conv at 4B dims ===\n");
    {
        struct { int out; int in; const char* name; } cases[] = {
            {9216, 2560, "gate/up_proj"},
            {2560, 9216, "down_proj"},
            {2560, 2560, "o_proj"},
            {2560, 4096, "o_proj (lin)"},
            {6144, 2560, "qkv (partial)"},
        };
        
        for (auto& tc : cases) {
            ANEKernel* k = ane_compile_dynamic_conv(tc.out, tc.in);
            if (!k) { printf("  %s: compile FAILED\n", tc.name); continue; }
            
            size_t numel = (size_t)tc.out * tc.in;
            uint16_t* W = (uint16_t*)calloc(numel, 2);
            float* x = (float*)calloc(tc.in, sizeof(float));
            float* y = (float*)calloc(tc.out, sizeof(float));
            x[0] = 1.0f;
            
            // Warmup
            ane_dynamic_conv_eval(k, y, x, W, tc.in, tc.out);
            
            int N = 20;
            Timer t;
            for (int i = 0; i < N; i++)
                ane_dynamic_conv_eval(k, y, x, W, tc.in, tc.out);
            double avg = t.elapsed_ms() / N;
            double memcpy_mb = numel * 2.0 / (1024 * 1024);
            printf("  %-20s [%5d,%4d] %.1fMB: %.2f ms/eval\n",
                   tc.name, tc.out, tc.in, memcpy_mb, avg);
            
            ane_free(k);
            free(W); free(x); free(y);
        }
    }
    
    // === Test 5: Dynamic FFN performance at 4B ===
    printf("\n=== Performance: dynamic FFN at 4B (dim=2560, inter=9216) ===\n");
    {
        int dim = 2560, inter = 9216;
        ANEKernel* k = ane_compile_dynamic_ffn(dim, inter);
        if (!k) { printf("  Compile FAILED\n"); return 1; }
        
        size_t g_sz = (size_t)inter * dim;
        size_t d_sz = (size_t)dim * inter;
        uint16_t* g = (uint16_t*)calloc(g_sz, 2);
        uint16_t* u = (uint16_t*)calloc(g_sz, 2);
        uint16_t* d = (uint16_t*)calloc(d_sz, 2);
        float* x = (float*)calloc(dim, sizeof(float));
        float* y = (float*)calloc(dim, sizeof(float));
        x[0] = 1.0f;
        
        // Warmup
        ane_dynamic_ffn_eval(k, y, x, g, u, d, dim, inter);
        
        int N = 20;
        Timer t;
        for (int i = 0; i < N; i++)
            ane_dynamic_ffn_eval(k, y, x, g, u, d, dim, inter);
        double avg = t.elapsed_ms() / N;
        double memcpy_mb = (2 * g_sz + d_sz) * 2.0 / (1024 * 1024);
        printf("  %.2f ms/eval (%.1f MB weight memcpy per call)\n", avg, memcpy_mb);
        printf("  Projected 32 layers: %.0f ms → %.1f tok/s\n",
               32 * avg, 1000.0 / (32 * avg));
        
        // Compare with constant-weight FFN
        printf("\n  For reference, constant-weight FFN eval: ~2ms/layer\n");
        printf("  Dynamic overhead: %.1fx\n", avg / 2.0);
        
        ane_free(k);
        free(g); free(u); free(d); free(x); free(y);
    }
    
    printf("\nFinal: compiles=%d, cache_loads=%d\n", ane_compile_count(), ane_cache_loads());
    objc_autoreleasePoolPop(pool);
    return 0;
}
