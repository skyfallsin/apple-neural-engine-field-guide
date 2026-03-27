// test_ane_limits4.cpp — Measure load-from-cache and unload times
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "core/ane_runtime.h"
#include <ane_lm/common.h>

extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void  objc_autoreleasePoolPop(void*);

using namespace ane_lm;

static uint16_t* make_bf16(size_t numel, int seed) {
    uint16_t* w = (uint16_t*)malloc(numel * 2);
    srand(seed);
    for (size_t i = 0; i < numel; i++)
        w[i] = f32_to_bf16(((float)(rand() % 1000) / 10000.0f) - 0.05f);
    return w;
}

// Test: measure compile, cache-load, unload, eval times for different kernel types
static void test_timing() {
    printf("=== Kernel timing: compile / cache-load / eval / unload ===\n\n");
    ane_set_persist_cache(true);
    
    int dim = 2560, inter = 9216;
    
    // --- FFN kernel ---
    {
        printf("Fused FFN (dim=%d, inter=%d, ~135MB):\n", dim, inter);
        uint16_t* g = make_bf16((size_t)inter * dim, 9001);
        uint16_t* u = make_bf16((size_t)inter * dim, 9002);
        uint16_t* d = make_bf16((size_t)dim * inter, 9003);
        
        // First compile (cold)
        Timer t;
        ANEKernel* k1 = ane_compile_fused_ffn(g, u, d, dim, inter);
        double compile_ms = t.elapsed_ms();
        printf("  Cold compile: %.1f ms\n", compile_ms);
        
        // Eval
        float* in = (float*)calloc(dim, sizeof(float));
        float* out = (float*)calloc(dim, sizeof(float));
        in[0] = 1.0f;
        
        // Warmup eval
        ane_matvec(k1, out, in, dim, dim);
        
        t.reset();
        for (int i = 0; i < 10; i++)
            ane_matvec(k1, out, in, dim, dim);
        printf("  Eval (avg of 10): %.3f ms\n", t.elapsed_ms() / 10.0);
        
        // Unload
        t.reset();
        ane_free(k1);
        printf("  Unload+free: %.1f ms\n", t.elapsed_ms());
        
        // Cache load (same weights = same hash)
        t.reset();
        ANEKernel* k2 = ane_compile_fused_ffn(g, u, d, dim, inter);
        double cache_ms = t.elapsed_ms();
        printf("  Cache load: %.1f ms (vs compile %.1f ms, %.1fx faster)\n",
               cache_ms, compile_ms, compile_ms / cache_ms);
        
        // Eval after cache load
        t.reset();
        for (int i = 0; i < 10; i++)
            ane_matvec(k2, out, in, dim, dim);
        printf("  Eval after cache load (avg of 10): %.3f ms\n", t.elapsed_ms() / 10.0);
        
        ane_free(k2);
        free(in); free(out);
        free(g); free(u); free(d);
    }
    
    printf("\n");
    
    // --- fused_2 kernel (deltanet first_proj) ---
    {
        int a_out = 6144, b_out = 4096;
        printf("Fused_2 (a=%d, b=%d, in=%d, ~50MB):\n", a_out, b_out, dim);
        uint16_t* wa = make_bf16((size_t)a_out * dim, 8001);
        uint16_t* wb = make_bf16((size_t)b_out * dim, 8002);
        
        Timer t;
        ANEKernel* k1 = ane_compile_fused_2(wa, a_out, wb, b_out, dim);
        double compile_ms = t.elapsed_ms();
        printf("  Cold compile: %.1f ms\n", compile_ms);
        
        float* in = (float*)calloc(dim, sizeof(float));
        float* out = (float*)calloc(a_out + b_out, sizeof(float));
        in[0] = 1.0f;
        ane_matvec(k1, out, in, dim, a_out + b_out);
        t.reset();
        for (int i = 0; i < 10; i++)
            ane_matvec(k1, out, in, dim, a_out + b_out);
        printf("  Eval (avg of 10): %.3f ms\n", t.elapsed_ms() / 10.0);
        
        t.reset();
        ane_free(k1);
        printf("  Unload+free: %.1f ms\n", t.elapsed_ms());
        
        t.reset();
        ANEKernel* k2 = ane_compile_fused_2(wa, a_out, wb, b_out, dim);
        printf("  Cache load: %.1f ms\n", t.elapsed_ms());
        
        ane_free(k2);
        free(in); free(out); free(wa); free(wb);
    }
    
    printf("\n");
    
    // --- matmul kernel (o_proj) ---
    {
        int o_out = 2560, o_in = 4096;
        printf("Matmul (out=%d, in=%d, ~20MB):\n", o_out, o_in);
        uint16_t* w = make_bf16((size_t)o_out * o_in, 7001);
        
        Timer t;
        ANEKernel* k1 = ane_compile_matmul(w, o_out, o_in);
        double compile_ms = t.elapsed_ms();
        printf("  Cold compile: %.1f ms\n", compile_ms);
        
        float* in = (float*)calloc(o_in, sizeof(float));
        float* out = (float*)calloc(o_out, sizeof(float));
        in[0] = 1.0f;
        ane_matvec(k1, out, in, o_in, o_out);
        t.reset();
        for (int i = 0; i < 10; i++)
            ane_matvec(k1, out, in, o_in, o_out);
        printf("  Eval (avg of 10): %.3f ms\n", t.elapsed_ms() / 10.0);
        
        t.reset();
        ane_free(k1);
        printf("  Unload+free: %.1f ms\n", t.elapsed_ms());
        
        t.reset();
        ANEKernel* k2 = ane_compile_matmul(w, o_out, o_in);
        printf("  Cache load: %.1f ms\n", t.elapsed_ms());
        
        ane_free(k2);
        free(in); free(out); free(w);
    }
    
    printf("\n");
    
    // --- LM head chunk ---
    {
        int rows = 16384;
        printf("LM head chunk (out=%d, in=%d, ~80MB):\n", rows, dim);
        uint16_t* w = make_bf16((size_t)rows * dim, 6001);
        
        Timer t;
        ANEKernel* k1 = ane_compile_matmul(w, rows, dim);
        double compile_ms = t.elapsed_ms();
        printf("  Cold compile: %.1f ms\n", compile_ms);
        
        float* in = (float*)calloc(dim, sizeof(float));
        float* out = (float*)calloc(rows, sizeof(float));
        ane_matvec(k1, out, in, dim, rows);
        t.reset();
        for (int i = 0; i < 10; i++)
            ane_matvec(k1, out, in, dim, rows);
        printf("  Eval (avg of 10): %.3f ms\n", t.elapsed_ms() / 10.0);
        
        t.reset();
        ane_free(k1);
        printf("  Unload+free: %.1f ms\n", t.elapsed_ms());
        
        t.reset();
        ANEKernel* k2 = ane_compile_matmul(w, rows, dim);
        printf("  Cache load: %.1f ms\n", t.elapsed_ms());
        
        ane_free(k2);
        free(in); free(out); free(w);
    }
    
    printf("\n");
    
    // --- Simulate per-layer load/eval/unload cycle ---
    {
        printf("=== Simulated per-layer load/eval/unload cycle ===\n");
        int lin_qkv = 6144, lin_val = 4096;
        
        // Pre-compile all 3 layer kernels to get them cached
        uint16_t* wa = make_bf16((size_t)lin_qkv * dim, 3001);
        uint16_t* wb = make_bf16((size_t)lin_val * dim, 3002);
        ANEKernel* tmp1 = ane_compile_fused_2(wa, lin_qkv, wb, lin_val, dim);
        ane_free(tmp1);
        
        uint16_t* wo = make_bf16((size_t)dim * lin_val, 3003);
        ANEKernel* tmp2 = ane_compile_matmul(wo, dim, lin_val);
        ane_free(tmp2);
        
        uint16_t* g = make_bf16((size_t)inter * dim, 3004);
        uint16_t* u = make_bf16((size_t)inter * dim, 3005);
        uint16_t* d = make_bf16((size_t)dim * inter, 3006);
        ANEKernel* tmp3 = ane_compile_fused_ffn(g, u, d, dim, inter);
        ane_free(tmp3);
        
        printf("  All 3 kernels pre-compiled and cached\n");
        
        // Now simulate: load all 3, eval each once, unload all 3
        float* in1 = (float*)calloc(dim, sizeof(float));
        float* out1 = (float*)calloc(lin_qkv + lin_val, sizeof(float));
        float* out2 = (float*)calloc(dim, sizeof(float));
        float* out3 = (float*)calloc(dim, sizeof(float));
        in1[0] = 1.0f;
        
        int N = 20;
        Timer t;
        for (int iter = 0; iter < N; iter++) {
            // Load from cache
            ANEKernel* k_fp = ane_compile_fused_2(wa, lin_qkv, wb, lin_val, dim);
            ANEKernel* k_op = ane_compile_matmul(wo, dim, lin_val);
            ANEKernel* k_ffn = ane_compile_fused_ffn(g, u, d, dim, inter);
            
            // Eval
            ane_matvec(k_fp, out1, in1, dim, lin_qkv + lin_val);
            ane_matvec(k_op, out2, out1 + lin_qkv, lin_val, dim);
            ane_matvec(k_ffn, out3, in1, dim, dim);
            
            // Unload
            ane_free(k_fp);
            ane_free(k_op);
            ane_free(k_ffn);
        }
        double total_ms = t.elapsed_ms();
        printf("  %d iterations of (3×cache-load + 3×eval + 3×free)\n", N);
        printf("  Total: %.1f ms, per-layer: %.1f ms\n", total_ms, total_ms / N);
        printf("  Projected 32-layer forward: %.1f ms (%.1f tok/s)\n",
               total_ms / N * 32, 1000.0 / (total_ms / N * 32));
        
        free(in1); free(out1); free(out2); free(out3);
        free(wa); free(wb); free(wo); free(g); free(u); free(d);
    }
}

int main(int argc, char* argv[]) {
    void* pool = objc_autoreleasePoolPush();
    g_verbose = false;
    ane_init();
    if (!ane_available()) { fprintf(stderr, "ANE not available!\n"); return 1; }
    
    test_timing();
    
    printf("\nFinal: compiles=%d, cache_loads=%d\n", ane_compile_count(), ane_cache_loads());
    objc_autoreleasePoolPop(pool);
    return 0;
}
