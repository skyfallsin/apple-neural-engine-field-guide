// test_dynamic5.cpp — Test mul+reduce_sum manual matvec on ANE
// The key insight: conv with dynamic weights fails at eval because ANE's conv unit
// reads weights from a dedicated internal weight bus (populated from BLOBFILE),
// not from general-purpose input IOSurfaces.
// But mul() and reduce_sum() are ALU ops that read from the normal data path.
// So W * x (broadcast multiply) + reduce_sum should work!
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

static void cpu_matvec(float* y, const float* W, const float* x, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        double sum = 0;
        for (int i = 0; i < in_dim; i++)
            sum += (double)W[o * in_dim + i] * (double)x[i];
        y[o] = (float)sum;
    }
}

// ===== Test: mul + reduce_sum matvec =====
// MIL: W:[D,D,1,1] * x:[1,D,1,SP] broadcasts to [D,D,1,SP]
// reduce_sum axis=1 -> [D,1,1,SP], reshape to [1,D,1,SP]
//
// For the weight tensor W:[out,in,1,1], the innermost dim is W=1.
// ANE pads W to SP=32 for IOSurface layout.
// So each element W[o,i,0,0] is at byte offset (o*in*SP + i*SP) * sizeof(fp16)
// But wait — the broadcast multiply W[o,i,0,0] * x[0,i,0,w] needs W to broadcast
// along the W dimension (1 -> SP). ANE knows the shape from the MIL, so it handles 
// this broadcast internally. The IOSurface just needs the data at the right position.
//
// Key question: does ANE expect the weight data at position [o,i,0,0] in the 
// IOSurface, or does it want all SP copies pre-filled?
static bool test_matvec_small(int D) {
    printf("=== mul+reduce_sum matvec %dx%d ===\n", D, D);
    
    char mil[4096];
    snprintf(mil, sizeof(mil),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x, tensor<fp16, [%d, %d, 1, 1]> W) {\n"
        "        tensor<fp16, [%d, %d, 1, %d]> prod = mul(x = W, y = x)[name = tensor<string, []>(\"m\")];\n"
        "        tensor<int32, [1]> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, [1]>([1])];\n"
        "        tensor<bool, []> kd = const()[name = tensor<string, []>(\"kd\"), val = tensor<bool, []>(false)];\n"
        "        tensor<fp16, [%d, 1, 1, %d]> s = reduce_sum(x = prod, axes = ax, keep_dims = kd)"
        "[name = tensor<string, []>(\"rs\")];\n"
        "        tensor<int32, [4]> os = const()[name = tensor<string, []>(\"os\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y = reshape(x = s, shape = os)[name = tensor<string, []>(\"yr\")];\n"
        "    } -> (y);\n"
        "}\n",
        D, SP, D, D,
        D, D, SP,
        D, SP,
        D, SP,
        D, SP);
    
    // Try with SP-padded W surface
    size_t w_bytes = (size_t)D * D * SP * sizeof(uint16_t);
    size_t in_sizes[2] = {
        (size_t)D * SP * sizeof(uint16_t),  // x: [1,D,1,SP]
        w_bytes                               // W: [D,D,1,1] SP-padded
    };
    size_t out_size = (size_t)D * SP * sizeof(uint16_t);
    
    ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &out_size);
    if (!k) { printf("  Compile FAILED\n"); return false; }
    printf("  Compiled OK\n");
    
    // Prepare data
    float* x = (float*)calloc(D, sizeof(float));
    float* W = (float*)calloc(D * D, sizeof(float));
    float* y_ref = (float*)calloc(D, sizeof(float));
    
    for (int i = 0; i < D; i++) x[i] = (float)(i + 1);
    // Identity matrix: y should equal x
    for (int o = 0; o < D; o++)
        for (int i = 0; i < D; i++)
            W[o * D + i] = (o == i) ? 1.0f : 0.0f;
    cpu_matvec(y_ref, W, x, D, D);
    
    // Write x to input 0 (SP-strided)
    uint16_t* x_fp16 = (uint16_t*)calloc(D, sizeof(uint16_t));
    for (int i = 0; i < D; i++) x_fp16[i] = f32_to_f16(x[i]);
    ane_write_surface_strided(k, 0, x_fp16, D);
    
    // Write W to input 1 (SP-strided for each [o,i] element)
    // W[o,i,0,0] at offset (o * D * SP + i * SP) in uint16 units
    uint16_t* w_buf = (uint16_t*)calloc(D * D * SP, sizeof(uint16_t));
    for (int o = 0; o < D; o++)
        for (int i = 0; i < D; i++)
            w_buf[o * D * SP + i * SP] = f32_to_f16(W[o * D + i]);
    ane_write_surface_raw(k, 1, w_buf, w_bytes);
    
    // Eval (just dispatch, read outputs)
    float* y = (float*)calloc(D, sizeof(float));
    float* outptrs[] = {y};
    int out_chs[] = {D};
    bool ok = ane_eval_raw_outputs(k, outptrs, out_chs);
    
    if (!ok) {
        printf("  EVAL FAILED!\n");
        ane_free(k);
        free(x); free(W); free(y); free(y_ref); free(x_fp16); free(w_buf);
        return false;
    }
    
    float mad = max_abs_diff(y, y_ref, D);
    printf("  EVAL OK! max_abs_diff = %.6f %s\n", mad, mad < 0.05f ? "PASS" : "FAIL");
    printf("  y[0..min(4,D)] = [");
    for (int i = 0; i < (D < 5 ? D : 5); i++) printf("%.4f%s", y[i], i < 4 ? ", " : "");
    printf("]\n  ref = [");
    for (int i = 0; i < (D < 5 ? D : 5); i++) printf("%.4f%s", y_ref[i], i < 4 ? ", " : "");
    printf("]\n");
    
    bool pass = mad < 0.05f;
    
    if (pass) {
        // Test with non-identity matrix
        printf("  Testing with random matrix...\n");
        srand(42);
        for (int o = 0; o < D; o++)
            for (int i = 0; i < D; i++)
                W[o * D + i] = ((float)(rand() % 2000) / 10000.0f) - 0.1f;
        cpu_matvec(y_ref, W, x, D, D);
        
        for (int o = 0; o < D; o++)
            for (int i = 0; i < D; i++)
                w_buf[o * D * SP + i * SP] = f32_to_f16(W[o * D + i]);
        ane_write_surface_raw(k, 1, w_buf, w_bytes);
        
        ok = ane_eval_raw_outputs(k, outptrs, out_chs);
        if (!ok) {
            printf("  Random EVAL FAILED!\n");
            pass = false;
        } else {
            mad = max_abs_diff(y, y_ref, D);
            printf("  Random matrix: max_abs_diff = %.6f %s\n", mad, mad < 0.05f ? "PASS" : "FAIL");
            printf("  y[0..min(4,D)] = [");
            for (int i = 0; i < (D < 5 ? D : 5); i++) printf("%.4f%s", y[i], i < 4 ? ", " : "");
            printf("]\n  ref = [");
            for (int i = 0; i < (D < 5 ? D : 5); i++) printf("%.4f%s", y_ref[i], i < 4 ? ", " : "");
            printf("]\n");
            if (mad >= 0.05f) pass = false;
        }
    }
    
    ane_free(k);
    free(x); free(W); free(y); free(y_ref); free(x_fp16); free(w_buf);
    return pass;
}

// Test at larger sizes  
static bool test_matvec_rect(int out_dim, int in_dim, const char* label) {
    printf("\n=== mul+reduce_sum matvec %dx%d (%s) ===\n", out_dim, in_dim, label);
    
    char mil[4096];
    snprintf(mil, sizeof(mil),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x, tensor<fp16, [%d, %d, 1, 1]> W) {\n"
        "        tensor<fp16, [%d, %d, 1, %d]> prod = mul(x = W, y = x)[name = tensor<string, []>(\"m\")];\n"
        "        tensor<int32, [1]> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, [1]>([1])];\n"
        "        tensor<bool, []> kd = const()[name = tensor<string, []>(\"kd\"), val = tensor<bool, []>(false)];\n"
        "        tensor<fp16, [%d, 1, 1, %d]> s = reduce_sum(x = prod, axes = ax, keep_dims = kd)"
        "[name = tensor<string, []>(\"rs\")];\n"
        "        tensor<int32, [4]> os = const()[name = tensor<string, []>(\"os\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y = reshape(x = s, shape = os)[name = tensor<string, []>(\"yr\")];\n"
        "    } -> (y);\n"
        "}\n",
        in_dim, SP, out_dim, in_dim,
        out_dim, in_dim, SP,
        out_dim, SP,
        out_dim, SP,
        out_dim, SP);
    
    size_t w_bytes = (size_t)out_dim * in_dim * SP * sizeof(uint16_t);
    size_t in_sizes[2] = {
        (size_t)in_dim * SP * sizeof(uint16_t),
        w_bytes
    };
    size_t out_size = (size_t)out_dim * SP * sizeof(uint16_t);
    
    printf("  W IOSurface: %.1f MB\n", (float)w_bytes / (1024*1024));
    
    Timer t;
    ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &out_size);
    if (!k) { printf("  Compile FAILED\n"); return false; }
    printf("  Compiled OK (%.0f ms)\n", t.elapsed_ms());
    
    // Random weights
    float* x = (float*)calloc(in_dim, sizeof(float));
    float* W = (float*)calloc((size_t)out_dim * in_dim, sizeof(float));
    float* y_ref = (float*)calloc(out_dim, sizeof(float));
    srand(77);
    for (int i = 0; i < in_dim; i++) x[i] = ((float)(rand() % 200) / 100.0f) - 1.0f;
    for (int o = 0; o < out_dim; o++)
        for (int i = 0; i < in_dim; i++)
            W[(size_t)o * in_dim + i] = ((float)(rand() % 2000) / 10000.0f) - 0.1f;
    cpu_matvec(y_ref, W, x, out_dim, in_dim);
    
    // Write inputs
    uint16_t* x_fp16 = (uint16_t*)calloc(in_dim, sizeof(uint16_t));
    for (int i = 0; i < in_dim; i++) x_fp16[i] = f32_to_f16(x[i]);
    ane_write_surface_strided(k, 0, x_fp16, in_dim);
    
    uint16_t* w_buf = (uint16_t*)calloc((size_t)out_dim * in_dim * SP, sizeof(uint16_t));
    for (int o = 0; o < out_dim; o++)
        for (int i = 0; i < in_dim; i++)
            w_buf[(size_t)o * in_dim * SP + (size_t)i * SP] = f32_to_f16(W[(size_t)o * in_dim + i]);
    ane_write_surface_raw(k, 1, w_buf, w_bytes);
    
    // Eval
    float* y = (float*)calloc(out_dim, sizeof(float));
    float* outptrs[] = {y};
    int out_chs[] = {out_dim};
    
    bool ok = ane_eval_raw_outputs(k, outptrs, out_chs);
    if (!ok) {
        printf("  EVAL FAILED!\n");
        ane_free(k);
        free(x); free(W); free(y); free(y_ref); free(x_fp16); free(w_buf);
        return false;
    }
    
    float mad = max_abs_diff(y, y_ref, out_dim);
    printf("  max_abs_diff = %.6f %s\n", mad, mad < 0.5f ? "PASS" : "FAIL");
    printf("  y[0..4] = [%.4f, %.4f, %.4f, %.4f, %.4f]\n", y[0], y[1], y[2], y[3], y[4]);
    printf("  ref[0..4] = [%.4f, %.4f, %.4f, %.4f, %.4f]\n", y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[4]);
    
    if (ok && mad < 0.5f) {
        // Performance test
        // First: measure weight memcpy time
        t.reset();
        int N = 10;
        for (int i = 0; i < N; i++) {
            ane_write_surface_raw(k, 1, w_buf, w_bytes);
            ane_eval_raw_outputs(k, outptrs, out_chs);
        }
        double avg = t.elapsed_ms() / N;
        printf("  Perf: %.2f ms/eval (write+dispatch+read)\n", avg);
    }
    
    ane_free(k);
    free(x); free(W); free(y); free(y_ref); free(x_fp16); free(w_buf);
    return ok && mad < 0.5f;
}

int main() {
    void* pool = objc_autoreleasePoolPush();
    g_verbose = false;
    ane_init();
    if (!ane_available()) { fprintf(stderr, "ANE not available!\n"); return 1; }
    ane_set_persist_cache(false);
    
    // Small sizes first
    bool p1 = test_matvec_small(8);
    bool p2 = test_matvec_small(16);
    bool p3 = test_matvec_small(32);
    bool p4 = test_matvec_small(64);
    
    // Larger if small works
    bool p5 = false, p6 = false;
    if (p1 && p2) {
        p5 = test_matvec_rect(128, 64, "128x64");
        p6 = test_matvec_rect(256, 128, "256x128");
    }
    
    // Model-scale if medium works
    if (p5 || p6) {
        test_matvec_rect(512, 512, "512x512");
        test_matvec_rect(1024, 1024, "1024x1024");
    }
    
    printf("\n=== Summary ===\n");
    printf("  8x8:   %s\n", p1 ? "PASS" : "FAIL");
    printf("  16x16: %s\n", p2 ? "PASS" : "FAIL");
    printf("  32x32: %s\n", p3 ? "PASS" : "FAIL");
    printf("  64x64: %s\n", p4 ? "PASS" : "FAIL");
    printf("  128x64:  %s\n", p5 ? "PASS" : "FAIL");
    printf("  256x128: %s\n", p6 ? "PASS" : "FAIL");
    printf("  compiles=%d\n", ane_compile_count());
    
    objc_autoreleasePoolPop(pool);
    return 0;
}
