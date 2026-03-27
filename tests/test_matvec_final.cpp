// test_matvec_final.cpp — Dynamic matvec using CPU-side tiling
//
// Strategy: tile x on CPU (repeat out_dim times), so both inputs have same shape.
// x_tiled: [out_dim, T, 1, SP] — same T*SP values in every N slice
// W:       [out_dim, T, 1, SP] — weight matrix
// ANE: mul(x_tiled, W) -> [out, T, 1, SP] (same shape, no broadcast)
//      reduce_sum(axis=1) -> [out, 1, 1, SP]  
// CPU: sum SP values per output channel -> y[o]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "core/ane_runtime.h"
#include <ane_lm/common.h>
extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void  objc_autoreleasePoolPop(void*);
using namespace ane_lm;
#define SP ANE_SPATIAL

static float max_abs_diff(const float* a, const float* b, int n) {
    float mx = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static void cpu_matvec_f16ref(float* y, const float* W, const float* x, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        float sum = 0;
        for (int i = 0; i < in_dim; i++) {
            float w16 = f16_to_f32(f32_to_f16(W[(size_t)o * in_dim + i]));
            float x16 = f16_to_f32(f32_to_f16(x[i]));
            sum += w16 * x16;
        }
        y[o] = sum;
    }
}

static bool test_matvec(int out_dim, int in_dim, const char* label, bool verbose) {
    int T = (in_dim + SP - 1) / SP;
    printf("=== Matvec %dx%d (%s) T=%d ===\n", out_dim, in_dim, label, T);
    
    // Both inputs same shape: [out_dim, T, 1, SP]
    char mil[4096];
    snprintf(mil, sizeof(mil),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [%d, %d, 1, %d]> xt, tensor<fp16, [%d, %d, 1, %d]> W) {\n"
        "        tensor<fp16, [%d, %d, 1, %d]> prod = mul(x = xt, y = W)[name = tensor<string, []>(\"m\")];\n"
        "        tensor<int32, [1]> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, [1]>([1])];\n"
        "        tensor<bool, []> kd = const()[name = tensor<string, []>(\"kd\"), val = tensor<bool, []>(true)];\n"
        "        tensor<fp16, [%d, 1, 1, %d]> y = reduce_sum(x = prod, axes = ax, keep_dims = kd)"
        "[name = tensor<string, []>(\"rs\")];\n"
        "    } -> (y);\n"
        "}\n",
        out_dim, T, SP, out_dim, T, SP,
        out_dim, T, SP,
        out_dim, SP);
    
    // BOTH inputs: [out_dim, T, 1, SP]
    size_t tensor_bytes = (size_t)out_dim * T * SP * sizeof(uint16_t);
    size_t out_bytes = (size_t)out_dim * SP * sizeof(uint16_t);
    size_t in_sizes[2] = {tensor_bytes, tensor_bytes};
    
    Timer timer;
    ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &out_bytes);
    if (!k) { printf("  COMPILE FAIL\n"); return false; }
    printf("  Compiled (%.0f ms)  each input=%.1f MB  output=%.0f KB\n",
           timer.elapsed_ms(), (float)tensor_bytes/1e6, (float)out_bytes/1e3);
    
    // Test data
    srand(42);
    float* x_f32 = (float*)calloc(in_dim, sizeof(float));
    float* W_f32 = (float*)calloc((size_t)out_dim * in_dim, sizeof(float));
    for (int i = 0; i < in_dim; i++)
        x_f32[i] = ((float)(rand() % 2000) / 1000.0f) - 1.0f;
    for (size_t i = 0; i < (size_t)out_dim * in_dim; i++)
        W_f32[i] = ((float)(rand() % 200) / 10000.0f) - 0.01f;
    
    float* y_ref = (float*)calloc(out_dim, sizeof(float));
    cpu_matvec_f16ref(y_ref, W_f32, x_f32, out_dim, in_dim);
    
    // Build x_tiled fp16: [out_dim, T, 1, SP] — same T*SP tile repeated out_dim times
    uint16_t* x_tile_row = (uint16_t*)calloc(T * SP, sizeof(uint16_t));
    for (int i = 0; i < in_dim; i++)
        x_tile_row[(i / SP) * SP + (i % SP)] = f32_to_f16(x_f32[i]);
    
    uint16_t* x_tiled = (uint16_t*)calloc(tensor_bytes / 2, sizeof(uint16_t));
    for (int o = 0; o < out_dim; o++)
        memcpy(x_tiled + (size_t)o * T * SP, x_tile_row, T * SP * sizeof(uint16_t));
    
    // Build W fp16: [out_dim, T, 1, SP]
    uint16_t* w_fp16 = (uint16_t*)calloc(tensor_bytes / 2, sizeof(uint16_t));
    for (int o = 0; o < out_dim; o++)
        for (int i = 0; i < in_dim; i++)
            w_fp16[(size_t)(o * T + i / SP) * SP + i % SP] = f32_to_f16(W_f32[(size_t)o * in_dim + i]);
    
    // Write to IOSurfaces
    ane_write_surface_raw(k, 0, x_tiled, tensor_bytes);
    ane_write_surface_raw(k, 1, w_fp16, tensor_bytes);
    
    // Eval
    timer.reset();
    float dummy; float* dp[] = {&dummy}; int dc[] = {1};
    bool ok = ane_eval_raw_outputs(k, dp, dc);
    if (!ok) {
        printf("  EVAL FAIL\n");
        ane_free(k); free(x_f32); free(W_f32); free(y_ref);
        free(x_tile_row); free(x_tiled); free(w_fp16);
        return false;
    }
    
    // Read full output + CPU reduce
    float* out_raw = (float*)calloc(out_dim * SP, sizeof(float));
    ane_read_output_raw(k, 0, out_raw, out_dim * SP);
    double eval_ms = timer.elapsed_ms();
    
    float* y_ane = (float*)calloc(out_dim, sizeof(float));
    for (int o = 0; o < out_dim; o++) {
        float sum = 0;
        for (int s = 0; s < SP; s++) sum += out_raw[o * SP + s];
        y_ane[o] = sum;
    }
    
    float mad = max_abs_diff(y_ane, y_ref, out_dim);
    float thr = 0.1f + 0.002f * in_dim;  // fp16 accumulation tolerance
    bool pass = mad < thr;
    printf("  Eval %.2f ms  max_abs_diff=%.6f (thr=%.3f)  %s\n", eval_ms, mad, thr, pass?"PASS":"FAIL");
    if (verbose || !pass) {
        int show = out_dim < 8 ? out_dim : 8;
        printf("  ANE: ["); for(int i=0;i<show;i++) printf("%.4f%s",y_ane[i],i<show-1?", ":"");
        printf("]\n  REF: ["); for(int i=0;i<show;i++) printf("%.4f%s",y_ref[i],i<show-1?", ":"");
        printf("]\n");
    }
    
    ane_free(k); free(x_f32); free(W_f32); free(y_ref); free(y_ane);
    free(x_tile_row); free(x_tiled); free(w_fp16); free(out_raw);
    return pass;
}

static void benchmark_matvec(int out_dim, int in_dim, int iters) {
    int T = (in_dim + SP - 1) / SP;
    printf("\n=== Benchmark %dx%d T=%d iters=%d ===\n", out_dim, in_dim, T, iters);
    
    char mil[4096];
    snprintf(mil, sizeof(mil),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [%d, %d, 1, %d]> xt, tensor<fp16, [%d, %d, 1, %d]> W) {\n"
        "        tensor<fp16, [%d, %d, 1, %d]> prod = mul(x = xt, y = W)[name = tensor<string, []>(\"m\")];\n"
        "        tensor<int32, [1]> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, [1]>([1])];\n"
        "        tensor<bool, []> kd = const()[name = tensor<string, []>(\"kd\"), val = tensor<bool, []>(true)];\n"
        "        tensor<fp16, [%d, 1, 1, %d]> y = reduce_sum(x = prod, axes = ax, keep_dims = kd)"
        "[name = tensor<string, []>(\"rs\")];\n"
        "    } -> (y);\n"
        "}\n",
        out_dim, T, SP, out_dim, T, SP, out_dim, T, SP, out_dim, SP);
    
    size_t tensor_bytes = (size_t)out_dim * T * SP * sizeof(uint16_t);
    size_t out_bytes = (size_t)out_dim * SP * sizeof(uint16_t);
    size_t in_sizes[2] = {tensor_bytes, tensor_bytes};
    
    ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &out_bytes);
    if (!k) { printf("  COMPILE FAIL\n"); return; }
    
    // Pre-build fp16 buffers
    srand(42);
    uint16_t* x_tile = (uint16_t*)calloc(T * SP, sizeof(uint16_t));
    float* x_f32 = (float*)calloc(in_dim, sizeof(float));
    float* W_f32 = (float*)calloc((size_t)out_dim * in_dim, sizeof(float));
    for (int i = 0; i < in_dim; i++) {
        x_f32[i] = ((float)(rand()%2000)/1000.f)-1.f;
        x_tile[(i/SP)*SP+(i%SP)] = f32_to_f16(x_f32[i]);
    }
    
    uint16_t* x_tiled = (uint16_t*)calloc(tensor_bytes/2, sizeof(uint16_t));
    for (int o = 0; o < out_dim; o++)
        memcpy(x_tiled + (size_t)o*T*SP, x_tile, T*SP*2);
    
    uint16_t* w_fp16 = (uint16_t*)calloc(tensor_bytes/2, sizeof(uint16_t));
    for (int o = 0; o < out_dim; o++)
        for (int i = 0; i < in_dim; i++) {
            float v = ((float)(rand()%200)/10000.f)-0.01f;
            W_f32[(size_t)o*in_dim+i] = v;
            w_fp16[(size_t)(o*T+i/SP)*SP+i%SP] = f32_to_f16(v);
        }
    
    float* out_raw = (float*)calloc(out_dim*SP, sizeof(float));
    float* y_buf = (float*)calloc(out_dim, sizeof(float));
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        ane_write_surface_raw(k, 0, x_tiled, tensor_bytes);
        ane_write_surface_raw(k, 1, w_fp16, tensor_bytes);
        float d; float*dp[]={&d}; int dc[]={1};
        ane_eval_raw_outputs(k, dp, dc);
    }
    
    // Full pipeline: tile x on CPU + write + write W + eval + read + reduce
    Timer timer;
    for (int i = 0; i < iters; i++) {
        // CPU tile x (in practice: memcpy x_tile row repeated)
        for (int o = 0; o < out_dim; o++)
            memcpy(x_tiled + (size_t)o*T*SP, x_tile, T*SP*2);
        ane_write_surface_raw(k, 0, x_tiled, tensor_bytes);
        ane_write_surface_raw(k, 1, w_fp16, tensor_bytes);
        float d; float*dp[]={&d}; int dc[]={1};
        ane_eval_raw_outputs(k, dp, dc);
        ane_read_output_raw(k, 0, out_raw, out_dim*SP);
        for (int o = 0; o < out_dim; o++) {
            float s=0; for(int j=0;j<SP;j++) s+=out_raw[o*SP+j];
            y_buf[o]=s;
        }
    }
    double total = timer.elapsed_ms()/iters;
    
    // Breakdown: x tile + write
    timer.reset();
    for (int i = 0; i < iters; i++) {
        for (int o = 0; o < out_dim; o++)
            memcpy(x_tiled + (size_t)o*T*SP, x_tile, T*SP*2);
        ane_write_surface_raw(k, 0, x_tiled, tensor_bytes);
    }
    double x_ms = timer.elapsed_ms()/iters;
    
    // Breakdown: W write
    timer.reset();
    for (int i = 0; i < iters; i++)
        ane_write_surface_raw(k, 1, w_fp16, tensor_bytes);
    double w_ms = timer.elapsed_ms()/iters;
    
    // Breakdown: eval only
    timer.reset();
    for (int i = 0; i < iters; i++) {
        float d; float*dp[]={&d}; int dc[]={1};
        ane_eval_raw_outputs(k, dp, dc);
    }
    double eval_ms = timer.elapsed_ms()/iters;
    
    printf("  Input size: %.2f MB each (x_tiled + W)\n", (float)tensor_bytes/1e6);
    printf("  Total: %.3f ms (x_tile+write: %.3f, W_write: %.3f, eval: %.3f, rest: %.3f)\n",
           total, x_ms, w_ms, eval_ms, total-x_ms-w_ms-eval_ms);
    printf("  Throughput: %.0f matvecs/sec\n", 1000.0/total);
    
    // Const-weight conv comparison
    uint16_t* bf16 = (uint16_t*)calloc((size_t)out_dim*in_dim, sizeof(uint16_t));
    for (size_t i=0;i<(size_t)out_dim*in_dim;i++) bf16[i]=f32_to_bf16(W_f32[i]);
    ANEKernel* kc = ane_compile_matmul(bf16, out_dim, in_dim);
    if (kc) {
        for (int i=0;i<5;i++) ane_matvec(kc, y_buf, x_f32, in_dim, out_dim);
        timer.reset();
        for (int i=0;i<iters;i++) ane_matvec(kc, y_buf, x_f32, in_dim, out_dim);
        double conv = timer.elapsed_ms()/iters;
        printf("  Const conv: %.3f ms  Dynamic/Conv: %.1fx\n", conv, total/conv);
        ane_free(kc);
    }
    
    ane_free(k); free(x_tile); free(x_f32); free(W_f32); free(x_tiled);
    free(w_fp16); free(out_raw); free(y_buf); free(bf16);
}

int main() {
    void* pool = objc_autoreleasePoolPush();
    g_verbose = false;
    ane_init();
    if (!ane_available()) { fprintf(stderr, "ANE not available!\n"); return 1; }
    ane_set_persist_cache(false);
    
    printf("==============================================\n");
    printf("  Dynamic Matvec — CPU-tiled x, same-shape mul\n");
    printf("==============================================\n\n");
    
    // Correctness
    bool p1 = test_matvec(8, 32, "8x32", true);
    bool p2 = test_matvec(32, 64, "32x64", true);
    bool p3 = test_matvec(64, 64, "64x64", false);
    bool p4 = test_matvec(128, 128, "128x128", false);
    bool p5 = test_matvec(256, 256, "256x256", false);
    
    printf("\n  8x32:%s 32x64:%s 64:%s 128:%s 256:%s\n",
           p1?"P":"F", p2?"P":"F", p3?"P":"F", p4?"P":"F", p5?"P":"F");
    
    if (p3 && p4 && p5) {
        bool s1 = test_matvec(512, 512, "512", false);
        bool s2 = test_matvec(1024, 1024, "1K", false);
        bool s3 = test_matvec(2560, 2560, "2560", false);
        printf("  512:%s 1K:%s 2560:%s\n", s1?"P":"F", s2?"P":"F", s3?"P":"F");
        
        if (s1) benchmark_matvec(512, 512, 100);
        if (s2) benchmark_matvec(1024, 1024, 50);
        if (s3) benchmark_matvec(2560, 2560, 20);
    }
    
    printf("\n  compiles=%d\n", ane_compile_count());
    objc_autoreleasePoolPop(pool);
    return 0;
}
