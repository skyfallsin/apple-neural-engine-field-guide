// test_dynamic11.cpp — Working around mul's N-broadcast limitation
//
// Discovery from test_dynamic10:
//   - reshape on runtime tensors: WORKS (even changing N!)
//   - add with N broadcast: WORKS
//   - mul with N broadcast: FAILS (0x1d)
//   - mul with C broadcast: WORKS (test_dynamic8 confirmed)
//   - reduce_sum axis=1: WORKS
//
// Strategy A: Use MIL `tile` to replicate x along N, then mul (same shape, no broadcast)
// Strategy B: Put out_dim in C, tiles in H — mul with C broadcast [1,1,T,SP]*[1,out,T,SP]
// Strategy C: Since reshape works, reshape W from [1,out*T,1,SP] to [out,T,1,SP] and
//             tile x from [1,T,1,SP] to [out,T,1,SP], then mul (no broadcast)

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

// ============================================================
// Strategy A: tile x along N to match W's N dimension
// ============================================================
static bool test_tile_x() {
    int out_dim = 4, T = 2;
    printf("--- Strategy A: tile x[1,%d,1,SP] -> [%d,%d,1,SP], then mul ---\n", T, out_dim, T);
    
    char mil[4096];
    snprintf(mil, sizeof(mil),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x, tensor<fp16, [%d, %d, 1, %d]> W) {\n"
        // tile x: [1,T,1,SP] -> [out,T,1,SP]
        "        tensor<int32, [4]> reps = const()[name = tensor<string, []>(\"rp\"), val = tensor<int32, [4]>([%d, 1, 1, 1])];\n"
        "        tensor<fp16, [%d, %d, 1, %d]> xt = tile(x = x, reps = reps)[name = tensor<string, []>(\"t\")];\n"
        // mul: same shape, no broadcast
        "        tensor<fp16, [%d, %d, 1, %d]> prod = mul(x = W, y = xt)[name = tensor<string, []>(\"m\")];\n"
        // reduce_sum axis=1
        "        tensor<int32, [1]> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, [1]>([1])];\n"
        "        tensor<bool, []> kd = const()[name = tensor<string, []>(\"kd\"), val = tensor<bool, []>(true)];\n"
        "        tensor<fp16, [%d, 1, 1, %d]> y = reduce_sum(x = prod, axes = ax, keep_dims = kd)"
        "[name = tensor<string, []>(\"rs\")];\n"
        "    } -> (y);\n"
        "}\n",
        T, SP, out_dim, T, SP,
        out_dim,
        out_dim, T, SP,
        out_dim, T, SP,
        out_dim, SP);
    
    size_t x_sz = (size_t)T * SP * sizeof(uint16_t);
    size_t w_sz = (size_t)out_dim * T * SP * sizeof(uint16_t);
    size_t o_sz = (size_t)out_dim * SP * sizeof(uint16_t);
    size_t in_sizes[2] = {x_sz, w_sz};
    ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &o_sz);
    if (!k) { printf("  COMPILE FAIL\n"); return false; }
    printf("  Compiled OK\n");
    
    float x_data[2] = {3.0f, 5.0f};
    float w_data[8];
    for (int o = 0; o < out_dim; o++)
        for (int t = 0; t < T; t++)
            w_data[o*T+t] = (float)((o+1)*(t+1));
    
    float* inputs[] = {x_data, w_data};
    int in_chs[] = {T, out_dim*T};
    float out[4]; float* op[] = {out}; int och[] = {out_dim};
    bool ok = ane_eval_multi(k, inputs, in_chs, op, och);
    if (ok) {
        printf("  EVAL OK!\n");
        bool correct = true;
        for (int o = 0; o < out_dim; o++) {
            float expected = 0;
            for (int t = 0; t < T; t++)
                expected += w_data[o*T+t] * x_data[t];
            printf("  o=%d: %.1f expected %.1f %s\n", o, out[o], expected,
                   fabsf(out[o]-expected)<0.5f ? "OK" : "MISMATCH");
            if (fabsf(out[o]-expected) > 0.5f) correct = false;
        }
        if (correct) printf("  Strategy A CORRECT!\n");
    } else printf("  EVAL FAIL\n");
    ane_free(k); return ok;
}

// ============================================================
// Strategy B: C-broadcast approach — out_dim in C, tiles in H
// x: [1, 1, T, SP], W: [1, out_dim, T, SP]
// mul broadcasts C: [1,1,T,SP]*[1,out,T,SP] -> [1,out,T,SP]
// reduce_sum axis=2 -> [1,out,1,SP]
// ============================================================
static bool test_c_broadcast() {
    int out_dim = 4, T = 2;
    printf("--- Strategy B: C-broadcast [1,1,%d,SP]*[1,%d,%d,SP] ---\n", T, out_dim, T);
    
    char mil[4096];
    snprintf(mil, sizeof(mil),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, 1, %d, %d]> x, tensor<fp16, [1, %d, %d, %d]> W) {\n"
        // mul: broadcasts C (1->out_dim)
        "        tensor<fp16, [1, %d, %d, %d]> prod = mul(x = W, y = x)[name = tensor<string, []>(\"m\")];\n"
        // reduce_sum axis=2 (H dim) -> [1, out, 1, SP]
        "        tensor<int32, [1]> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, [1]>([2])];\n"
        "        tensor<bool, []> kd = const()[name = tensor<string, []>(\"kd\"), val = tensor<bool, []>(true)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y = reduce_sum(x = prod, axes = ax, keep_dims = kd)"
        "[name = tensor<string, []>(\"rs\")];\n"
        "    } -> (y);\n"
        "}\n",
        T, SP, out_dim, T, SP,
        out_dim, T, SP,
        out_dim, SP);
    
    // x: [1, 1, T, SP] — T*SP values
    // W: [1, out_dim, T, SP] — out_dim*T*SP values
    size_t x_sz = (size_t)1 * T * SP * sizeof(uint16_t);  // C=1, H=T
    size_t w_sz = (size_t)out_dim * T * SP * sizeof(uint16_t);  // C=out, H=T
    size_t o_sz = (size_t)out_dim * 1 * SP * sizeof(uint16_t);  // C=out, H=1
    size_t in_sizes[2] = {x_sz, w_sz};
    ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &o_sz);
    if (!k) { printf("  COMPILE FAIL\n"); return false; }
    printf("  Compiled OK\n");
    
    // x: channels=1*T=T (in SP-strided layout)
    // SP-strided layout for [1,1,T,SP]: the "channels" seen by ane_eval_multi = N*C*H = 1*1*T = T
    // Each channel h has SP values, but ane_eval_multi only writes pos 0 per channel
    // So x_data[0] goes to (h=0,s=0), x_data[1] goes to (h=1,s=0)
    float x_data[2] = {3.0f, 5.0f};
    // W: channels = N*C*H = 1*out*T = out*T
    float w_data[8];
    for (int o = 0; o < out_dim; o++)
        for (int t = 0; t < T; t++)
            w_data[o*T+t] = (float)((o+1)*(t+1));
    
    float* inputs[] = {x_data, w_data};
    int in_chs[] = {T, out_dim*T};
    float out[4]; float* op[] = {out}; int och[] = {out_dim};
    bool ok = ane_eval_multi(k, inputs, in_chs, op, och);
    if (ok) {
        printf("  EVAL OK!\n");
        bool correct = true;
        for (int o = 0; o < out_dim; o++) {
            float expected = 0;
            for (int t = 0; t < T; t++)
                expected += w_data[o*T+t] * x_data[t];
            printf("  o=%d: %.1f expected %.1f %s\n", o, out[o], expected,
                   fabsf(out[o]-expected)<0.5f ? "OK" : "MISMATCH");
            if (fabsf(out[o]-expected) > 0.5f) correct = false;
        }
        if (correct) printf("  Strategy B CORRECT!\n");
    } else printf("  EVAL FAIL\n");
    ane_free(k); return ok;
}

// ============================================================
// Strategy C: reshape W + tile x (both on ANE)
// Input W as [1, out*T, 1, SP] (flat), reshape to [out, T, 1, SP]
// Input x as [1, T, 1, SP], tile to [out, T, 1, SP]
// mul (same shape) -> reduce
// ============================================================
static bool test_reshape_tile() {
    int out_dim = 4, T = 2;
    printf("--- Strategy C: reshape W[1,%d,1,SP]->[%d,%d,1,SP] + tile x ---\n",
           out_dim*T, out_dim, T);
    
    char mil[4096];
    snprintf(mil, sizeof(mil),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x, tensor<fp16, [1, %d, 1, %d]> Wflat) {\n"
        // Reshape W: [1, out*T, 1, SP] -> [out, T, 1, SP]
        "        tensor<int32, [4]> ws = const()[name = tensor<string, []>(\"ws\"), val = tensor<int32, [4]>([%d, %d, 1, %d])];\n"
        "        tensor<fp16, [%d, %d, 1, %d]> W = reshape(x = Wflat, shape = ws)[name = tensor<string, []>(\"rw\")];\n"
        // Tile x: [1,T,1,SP] -> [out,T,1,SP]
        "        tensor<int32, [4]> reps = const()[name = tensor<string, []>(\"rp\"), val = tensor<int32, [4]>([%d, 1, 1, 1])];\n"
        "        tensor<fp16, [%d, %d, 1, %d]> xt = tile(x = x, reps = reps)[name = tensor<string, []>(\"t\")];\n"
        // mul (same shape, no broadcast)
        "        tensor<fp16, [%d, %d, 1, %d]> prod = mul(x = W, y = xt)[name = tensor<string, []>(\"m\")];\n"
        // reduce_sum axis=1
        "        tensor<int32, [1]> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, [1]>([1])];\n"
        "        tensor<bool, []> kd = const()[name = tensor<string, []>(\"kd\"), val = tensor<bool, []>(true)];\n"
        "        tensor<fp16, [%d, 1, 1, %d]> y = reduce_sum(x = prod, axes = ax, keep_dims = kd)"
        "[name = tensor<string, []>(\"rs\")];\n"
        "    } -> (y);\n"
        "}\n",
        T, SP, out_dim*T, SP,
        out_dim, T, SP,
        out_dim, T, SP,
        out_dim,
        out_dim, T, SP,
        out_dim, T, SP,
        out_dim, SP);
    
    size_t x_sz = (size_t)T * SP * sizeof(uint16_t);
    size_t w_sz = (size_t)out_dim * T * SP * sizeof(uint16_t);
    size_t o_sz = (size_t)out_dim * SP * sizeof(uint16_t);
    size_t in_sizes[2] = {x_sz, w_sz};
    ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &o_sz);
    if (!k) { printf("  COMPILE FAIL\n"); return false; }
    printf("  Compiled OK\n");
    
    float x_data[2] = {3.0f, 5.0f};
    float w_data[8];
    for (int o = 0; o < out_dim; o++)
        for (int t = 0; t < T; t++)
            w_data[o*T+t] = (float)((o+1)*(t+1));
    
    float* inputs[] = {x_data, w_data};
    int in_chs[] = {T, out_dim*T};
    float out[4]; float* op[] = {out}; int och[] = {out_dim};
    bool ok = ane_eval_multi(k, inputs, in_chs, op, och);
    if (ok) {
        printf("  EVAL OK!\n");
        bool correct = true;
        for (int o = 0; o < out_dim; o++) {
            float expected = 0;
            for (int t = 0; t < T; t++)
                expected += w_data[o*T+t] * x_data[t];
            printf("  o=%d: %.1f expected %.1f %s\n", o, out[o], expected,
                   fabsf(out[o]-expected)<0.5f ? "OK" : "MISMATCH");
            if (fabsf(out[o]-expected) > 0.5f) correct = false;
        }
        if (correct) printf("  Strategy C CORRECT!\n");
    } else printf("  EVAL FAIL\n");
    ane_free(k); return ok;
}

// ============================================================
// Full matvec using winning strategy + scale + benchmark
// ============================================================
static bool test_full_matvec(int out_dim, int in_dim, const char* label, bool verbose,
                             const char* strategy) {
    int T = (in_dim + SP - 1) / SP;
    printf("=== Matvec %dx%d (%s) T=%d [%s] ===\n", out_dim, in_dim, label, T, strategy);
    
    char mil[8192];
    if (strategy[0] == 'A') {
        // Strategy A: tile x along N
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x, tensor<fp16, [%d, %d, 1, %d]> W) {\n"
            "        tensor<int32, [4]> reps = const()[name = tensor<string, []>(\"rp\"), val = tensor<int32, [4]>([%d, 1, 1, 1])];\n"
            "        tensor<fp16, [%d, %d, 1, %d]> xt = tile(x = x, reps = reps)[name = tensor<string, []>(\"t\")];\n"
            "        tensor<fp16, [%d, %d, 1, %d]> prod = mul(x = W, y = xt)[name = tensor<string, []>(\"m\")];\n"
            "        tensor<int32, [1]> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, [1]>([1])];\n"
            "        tensor<bool, []> kd = const()[name = tensor<string, []>(\"kd\"), val = tensor<bool, []>(true)];\n"
            "        tensor<fp16, [%d, 1, 1, %d]> y = reduce_sum(x = prod, axes = ax, keep_dims = kd)[name = tensor<string, []>(\"rs\")];\n"
            "    } -> (y);\n"
            "}\n",
            T, SP, out_dim, T, SP,
            out_dim, out_dim, T, SP,
            out_dim, T, SP,
            out_dim, SP);
    } else {
        // Strategy B: C-broadcast
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, 1, %d, %d]> x, tensor<fp16, [1, %d, %d, %d]> W) {\n"
            "        tensor<fp16, [1, %d, %d, %d]> prod = mul(x = W, y = x)[name = tensor<string, []>(\"m\")];\n"
            "        tensor<int32, [1]> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, [1]>([2])];\n"
            "        tensor<bool, []> kd = const()[name = tensor<string, []>(\"kd\"), val = tensor<bool, []>(true)];\n"
            "        tensor<fp16, [1, %d, 1, %d]> y = reduce_sum(x = prod, axes = ax, keep_dims = kd)[name = tensor<string, []>(\"rs\")];\n"
            "    } -> (y);\n"
            "}\n",
            T, SP, out_dim, T, SP,
            out_dim, T, SP,
            out_dim, SP);
    }
    
    size_t x_sz, w_sz, o_sz;
    if (strategy[0] == 'A') {
        x_sz = (size_t)T * SP * sizeof(uint16_t);
        w_sz = (size_t)out_dim * T * SP * sizeof(uint16_t);
        o_sz = (size_t)out_dim * SP * sizeof(uint16_t);
    } else {
        x_sz = (size_t)1 * T * SP * sizeof(uint16_t);  // [1,1,T,SP]
        w_sz = (size_t)out_dim * T * SP * sizeof(uint16_t);  // [1,out,T,SP]
        o_sz = (size_t)out_dim * 1 * SP * sizeof(uint16_t);  // [1,out,1,SP]
    }
    size_t in_sizes[2] = {x_sz, w_sz};
    
    Timer timer;
    ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &o_sz);
    if (!k) { printf("  COMPILE FAIL\n"); return false; }
    printf("  Compiled (%.0f ms)  W=%.1f MB\n", timer.elapsed_ms(), (float)w_sz/1e6);
    
    srand(42);
    float* x_f32 = (float*)calloc(in_dim, sizeof(float));
    float* W_f32 = (float*)calloc((size_t)out_dim * in_dim, sizeof(float));
    for (int i = 0; i < in_dim; i++) x_f32[i] = ((float)(rand()%2000)/1000.f)-1.f;
    for (size_t i = 0; i < (size_t)out_dim*in_dim; i++) W_f32[i] = ((float)(rand()%200)/10000.f)-0.01f;
    
    float* y_ref = (float*)calloc(out_dim, sizeof(float));
    cpu_matvec_f16ref(y_ref, W_f32, x_f32, out_dim, in_dim);
    
    // Write x and W using raw surface writes
    uint16_t* x_fp16 = (uint16_t*)calloc(x_sz / 2, sizeof(uint16_t));
    uint16_t* w_fp16 = (uint16_t*)calloc(w_sz / 2, sizeof(uint16_t));
    
    if (strategy[0] == 'A') {
        // x: [1,T,1,SP] — x[i] at (c=i/SP, s=i%SP) -> offset (i/SP)*SP + i%SP
        for (int i = 0; i < in_dim; i++)
            x_fp16[(i/SP)*SP + (i%SP)] = f32_to_f16(x_f32[i]);
        // W: [out,T,1,SP] — W[o,i] at (n=o,c=i/SP,s=i%SP) -> (o*T+i/SP)*SP+i%SP
        for (int o = 0; o < out_dim; o++)
            for (int i = 0; i < in_dim; i++)
                w_fp16[(size_t)(o*T+i/SP)*SP + i%SP] = f32_to_f16(W_f32[(size_t)o*in_dim+i]);
    } else {
        // x: [1,1,T,SP] — same layout (C=1, H=T, W=SP) -> offset t*SP + s
        for (int i = 0; i < in_dim; i++)
            x_fp16[(i/SP)*SP + (i%SP)] = f32_to_f16(x_f32[i]);
        // W: [1,out,T,SP] — (n=0,c=o,h=t,s) -> (o*T+t)*SP + s
        for (int o = 0; o < out_dim; o++)
            for (int i = 0; i < in_dim; i++)
                w_fp16[(size_t)(o*T+i/SP)*SP + i%SP] = f32_to_f16(W_f32[(size_t)o*in_dim+i]);
    }
    
    ane_write_surface_raw(k, 0, x_fp16, x_sz);
    ane_write_surface_raw(k, 1, w_fp16, w_sz);
    
    // Eval + read all output values
    timer.reset();
    float dummy; float* dp[] = {&dummy}; int dc[] = {1};
    bool ok = ane_eval_raw_outputs(k, dp, dc);
    if (!ok) {
        printf("  EVAL FAIL\n");
        ane_free(k); free(x_f32); free(W_f32); free(y_ref); free(x_fp16); free(w_fp16);
        return false;
    }
    
    float* out_raw = (float*)calloc(out_dim * SP, sizeof(float));
    ane_read_output_raw(k, 0, out_raw, out_dim * SP);
    double eval_ms = timer.elapsed_ms();
    
    // CPU reduce: sum SP positions per output channel
    float* y_ane = (float*)calloc(out_dim, sizeof(float));
    for (int o = 0; o < out_dim; o++) {
        float sum = 0;
        for (int s = 0; s < SP; s++) sum += out_raw[o*SP+s];
        y_ane[o] = sum;
    }
    
    float mad = max_abs_diff(y_ane, y_ref, out_dim);
    float thr = 0.05f + 0.001f * in_dim;
    bool pass = mad < thr;
    printf("  Eval %.2f ms  max_abs_diff=%.6f (thr=%.4f)  %s\n", eval_ms, mad, thr, pass?"PASS":"FAIL");
    if (verbose || !pass) {
        int show = out_dim < 8 ? out_dim : 8;
        printf("  ANE: ["); for(int i=0;i<show;i++) printf("%.4f%s",y_ane[i],i<show-1?", ":"");
        printf("]\n  REF: ["); for(int i=0;i<show;i++) printf("%.4f%s",y_ref[i],i<show-1?", ":"");
        printf("]\n");
    }
    
    ane_free(k); free(x_f32); free(W_f32); free(y_ref); free(y_ane);
    free(x_fp16); free(w_fp16); free(out_raw);
    return pass;
}

// ============================================================
// Benchmark
// ============================================================
static void benchmark(int out_dim, int in_dim, int iters, const char* strategy) {
    int T = (in_dim + SP - 1) / SP;
    printf("\n=== Bench %dx%d T=%d [%s] iters=%d ===\n", out_dim, in_dim, T, strategy, iters);
    
    char mil[8192];
    if (strategy[0] == 'A') {
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x, tensor<fp16, [%d, %d, 1, %d]> W) {\n"
            "        tensor<int32, [4]> reps = const()[name = tensor<string, []>(\"rp\"), val = tensor<int32, [4]>([%d, 1, 1, 1])];\n"
            "        tensor<fp16, [%d, %d, 1, %d]> xt = tile(x = x, reps = reps)[name = tensor<string, []>(\"t\")];\n"
            "        tensor<fp16, [%d, %d, 1, %d]> prod = mul(x = W, y = xt)[name = tensor<string, []>(\"m\")];\n"
            "        tensor<int32, [1]> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, [1]>([1])];\n"
            "        tensor<bool, []> kd = const()[name = tensor<string, []>(\"kd\"), val = tensor<bool, []>(true)];\n"
            "        tensor<fp16, [%d, 1, 1, %d]> y = reduce_sum(x = prod, axes = ax, keep_dims = kd)[name = tensor<string, []>(\"rs\")];\n"
            "    } -> (y);\n"
            "}\n",
            T, SP, out_dim, T, SP, out_dim, out_dim, T, SP, out_dim, T, SP, out_dim, SP);
    } else {
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, 1, %d, %d]> x, tensor<fp16, [1, %d, %d, %d]> W) {\n"
            "        tensor<fp16, [1, %d, %d, %d]> prod = mul(x = W, y = x)[name = tensor<string, []>(\"m\")];\n"
            "        tensor<int32, [1]> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, [1]>([2])];\n"
            "        tensor<bool, []> kd = const()[name = tensor<string, []>(\"kd\"), val = tensor<bool, []>(true)];\n"
            "        tensor<fp16, [1, %d, 1, %d]> y = reduce_sum(x = prod, axes = ax, keep_dims = kd)[name = tensor<string, []>(\"rs\")];\n"
            "    } -> (y);\n"
            "}\n",
            T, SP, out_dim, T, SP, out_dim, T, SP, out_dim, SP);
    }
    
    size_t x_sz = (strategy[0]=='A') ? (size_t)T*SP*2 : (size_t)1*T*SP*2;
    size_t w_sz = (size_t)out_dim*T*SP*2;
    size_t o_sz = (size_t)out_dim*SP*2;
    size_t in_sizes[2] = {x_sz, w_sz};
    ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &o_sz);
    if (!k) { printf("  COMPILE FAIL\n"); return; }
    
    srand(42);
    uint16_t* x_fp16 = (uint16_t*)calloc(x_sz/2, sizeof(uint16_t));
    uint16_t* w_fp16 = (uint16_t*)calloc(w_sz/2, sizeof(uint16_t));
    float* x_f32 = (float*)calloc(in_dim, sizeof(float));
    float* W_f32 = (float*)calloc((size_t)out_dim*in_dim, sizeof(float));
    for (int i = 0; i < in_dim; i++) { 
        x_f32[i] = ((float)(rand()%2000)/1000.f)-1.f;
        x_fp16[(i/SP)*SP+(i%SP)] = f32_to_f16(x_f32[i]);
    }
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
        ane_write_surface_raw(k, 0, x_fp16, x_sz);
        ane_write_surface_raw(k, 1, w_fp16, w_sz);
        float d; float*dp[]={&d}; int dc[]={1};
        ane_eval_raw_outputs(k, dp, dc);
    }
    
    // Full pipeline
    Timer timer;
    for (int i = 0; i < iters; i++) {
        ane_write_surface_raw(k, 0, x_fp16, x_sz);
        ane_write_surface_raw(k, 1, w_fp16, w_sz);
        float d; float*dp[]={&d}; int dc[]={1};
        ane_eval_raw_outputs(k, dp, dc);
        ane_read_output_raw(k, 0, out_raw, out_dim*SP);
        for (int o = 0; o < out_dim; o++) {
            float s = 0; for (int j = 0; j < SP; j++) s += out_raw[o*SP+j];
            y_buf[o] = s;
        }
    }
    double total = timer.elapsed_ms()/iters;
    
    // W memcpy only
    timer.reset();
    for (int i = 0; i < iters; i++) ane_write_surface_raw(k, 1, w_fp16, w_sz);
    double memcpy = timer.elapsed_ms()/iters;
    
    // Eval only
    timer.reset();
    for (int i = 0; i < iters; i++) { float d; float*dp[]={&d}; int dc[]={1}; ane_eval_raw_outputs(k,dp,dc); }
    double eval = timer.elapsed_ms()/iters;
    
    printf("  W: %.2f MB  Total: %.3f ms (memcpy: %.3f, eval: %.3f, rest: %.3f)\n",
           (float)w_sz/1e6, total, memcpy, eval, total-memcpy-eval);
    
    // Const-weight conv comparison
    uint16_t* bf16 = (uint16_t*)calloc((size_t)out_dim*in_dim, sizeof(uint16_t));
    for (size_t i = 0; i < (size_t)out_dim*in_dim; i++) bf16[i] = f32_to_bf16(W_f32[i]);
    ANEKernel* kc = ane_compile_matmul(bf16, out_dim, in_dim);
    if (kc) {
        for (int i = 0; i < 5; i++) ane_matvec(kc, y_buf, x_f32, in_dim, out_dim);
        timer.reset();
        for (int i = 0; i < iters; i++) ane_matvec(kc, y_buf, x_f32, in_dim, out_dim);
        double conv = timer.elapsed_ms()/iters;
        printf("  Const conv: %.3f ms  Dynamic/Conv: %.1fx\n", conv, total/conv);
        ane_free(kc);
    }
    ane_free(k); free(x_fp16); free(w_fp16); free(x_f32); free(W_f32);
    free(out_raw); free(y_buf); free(bf16);
}

int main() {
    void* pool = objc_autoreleasePoolPush();
    g_verbose = false;
    ane_init();
    if (!ane_available()) { fprintf(stderr, "ANE not available!\n"); return 1; }
    ane_set_persist_cache(false);
    
    printf("============================================\n");
    printf("  Dynamic Matvec — Workaround Strategies\n");
    printf("============================================\n\n");
    
    bool sa = test_tile_x();
    bool sb = test_c_broadcast();
    bool sc = test_reshape_tile();
    
    printf("\n  Strategy A (tile x): %s\n", sa?"WORKS":"BROKEN");
    printf("  Strategy B (C-bcast): %s\n", sb?"WORKS":"BROKEN");
    printf("  Strategy C (reshape+tile): %s\n", sc?"WORKS":"BROKEN");
    
    // Test the winning strategy(ies) at scale
    const char* strat = sa ? "A" : (sb ? "B" : "C");
    printf("\n=== Scale-up with strategy %s ===\n", strat);
    
    bool s1 = test_full_matvec(8, 32, "8x32", true, strat);
    bool s2 = test_full_matvec(32, 64, "32x64", true, strat);
    bool s3 = test_full_matvec(64, 64, "64x64", false, strat);
    bool s4 = test_full_matvec(128, 128, "128", false, strat);
    bool s5 = test_full_matvec(256, 256, "256", false, strat);
    printf("\n  8x32:%s 32x64:%s 64:%s 128:%s 256:%s\n",
           s1?"P":"F", s2?"P":"F", s3?"P":"F", s4?"P":"F", s5?"P":"F");
    
    if (s3 && s4 && s5) {
        bool s6 = test_full_matvec(512, 512, "512", false, strat);
        bool s7 = test_full_matvec(1024, 1024, "1K", false, strat);
        bool s8 = test_full_matvec(2560, 2560, "2560", false, strat);
        printf("  512:%s 1K:%s 2560:%s\n", s6?"P":"F", s7?"P":"F", s8?"P":"F");
        
        if (s6) benchmark(512, 512, 100, strat);
        if (s7) benchmark(1024, 1024, 50, strat);
        if (s8) benchmark(2560, 2560, 20, strat);
    }
    
    printf("\n  compiles=%d\n", ane_compile_count());
    objc_autoreleasePoolPop(pool);
    return 0;
}
