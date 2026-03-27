// test_dynamic10.cpp — Systematic dynamic matvec investigation
//
// Key constraints discovered:
// - All runtime input IOSurfaces MUST have W (innermost dim) >= SP=32
// - Broadcasting works on N and C dims  
// - Reshape changing N dim on runtime tensors -> eval fail (0x1d)
//
// Strategy: avoid reshape entirely. Pass W as [out_dim, T, 1, SP] directly.
// x: [1, T, 1, SP], W: [out, T, 1, SP]
// mul(W, x) -> [out, T, 1, SP]  (broadcast on N)
// reduce_sum(axis=1) -> [out, 1, 1, SP]
// CPU: sum SP values per output row -> scalar y[o]

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
    // fp16 precision reference: quantize inputs to fp16, accumulate in fp32
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
// PHASE 1: Does reshape work on runtime tensors?
// ============================================================

static bool test_reshape_same_n() {
    printf("--- 1a: reshape [1,8,1,SP] -> [1,4,2,SP] (N unchanged) ---\n");
    char mil[2048];
    snprintf(mil, sizeof(mil),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, 8, 1, %d]> x) {\n"
        "        tensor<int32, [4]> s = const()[name = tensor<string, []>(\"s\"), val = tensor<int32, [4]>([1, 4, 2, %d])];\n"
        "        tensor<fp16, [1, 4, 2, %d]> y = reshape(x = x, shape = s)[name = tensor<string, []>(\"r\")];\n"
        "    } -> (y);\n"
        "}\n", SP, SP, SP);
    size_t in_sz = 8 * SP * sizeof(uint16_t);
    size_t out_sz = 4 * 2 * SP * sizeof(uint16_t);
    ANEKernel* k = ane_compile_mil(mil, 1, &in_sz, 1, &out_sz);
    if (!k) { printf("  COMPILE FAIL\n"); return false; }
    float vals[8]; for (int i = 0; i < 8; i++) vals[i] = (float)(i + 1);
    float* inp[] = {vals}; int in_ch[] = {8};
    float out[8] = {}; float* outp[] = {out}; int out_ch[] = {8};
    bool ok = ane_eval_multi(k, inp, in_ch, outp, out_ch);
    if (ok) {
        printf("  EVAL OK: [%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f]\n",
               out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
        bool match = true;
        for (int i = 0; i < 8; i++) if (fabsf(out[i] - vals[i]) > 0.01f) match = false;
        printf("  Data preserved: %s\n", match ? "YES" : "NO");
    } else printf("  EVAL FAIL\n");
    ane_free(k); return ok;
}

static bool test_reshape_changes_n() {
    printf("--- 1b: reshape [1,8,1,SP] -> [2,4,1,SP] (N: 1->2) ---\n");
    char mil[2048];
    snprintf(mil, sizeof(mil),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, 8, 1, %d]> x) {\n"
        "        tensor<int32, [4]> s = const()[name = tensor<string, []>(\"s\"), val = tensor<int32, [4]>([2, 4, 1, %d])];\n"
        "        tensor<fp16, [2, 4, 1, %d]> y = reshape(x = x, shape = s)[name = tensor<string, []>(\"r\")];\n"
        "    } -> (y);\n"
        "}\n", SP, SP, SP);
    size_t in_sz = 8 * SP * sizeof(uint16_t);
    size_t out_sz = 2 * 4 * SP * sizeof(uint16_t);
    ANEKernel* k = ane_compile_mil(mil, 1, &in_sz, 1, &out_sz);
    if (!k) { printf("  COMPILE FAIL\n"); return false; }
    float vals[8]; for (int i = 0; i < 8; i++) vals[i] = (float)(i + 1);
    float* inp[] = {vals}; int in_ch[] = {8};
    float out[8] = {}; float* outp[] = {out}; int out_ch[] = {8};
    bool ok = ane_eval_multi(k, inp, in_ch, outp, out_ch);
    if (ok) {
        printf("  EVAL OK: [%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f]\n",
               out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
    } else printf("  EVAL FAIL (reshape changing N broken)\n");
    ane_free(k); return ok;
}

static bool test_reshape_output_n() {
    printf("--- 1c: input [2,4,1,SP], reshape -> [1,8,1,SP] ---\n");
    char mil[2048];
    snprintf(mil, sizeof(mil),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [2, 4, 1, %d]> x) {\n"
        "        tensor<int32, [4]> s = const()[name = tensor<string, []>(\"s\"), val = tensor<int32, [4]>([1, 8, 1, %d])];\n"
        "        tensor<fp16, [1, 8, 1, %d]> y = reshape(x = x, shape = s)[name = tensor<string, []>(\"r\")];\n"
        "    } -> (y);\n"
        "}\n", SP, SP, SP);
    size_t in_sz = 2 * 4 * SP * sizeof(uint16_t);
    size_t out_sz = 8 * SP * sizeof(uint16_t);
    ANEKernel* k = ane_compile_mil(mil, 1, &in_sz, 1, &out_sz);
    if (!k) { printf("  COMPILE FAIL\n"); return false; }
    float vals[8]; for (int i = 0; i < 8; i++) vals[i] = (float)(i + 1);
    float* inp[] = {vals}; int in_ch[] = {8};
    float out[8] = {}; float* outp[] = {out}; int out_ch[] = {8};
    bool ok = ane_eval_multi(k, inp, in_ch, outp, out_ch);
    if (ok) {
        printf("  EVAL OK: [%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f]\n",
               out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
    } else printf("  EVAL FAIL\n");
    ane_free(k); return ok;
}

// ============================================================
// PHASE 2: Operations with N broadcast (no reshape needed)
// ============================================================

// First: confirm add with N broadcast actually evaluates
static bool test_add_broadcast_n() {
    printf("--- 2pre: add [1,2,1,SP] + [4,2,1,SP] (broadcast N, sanity) ---\n");
    char mil[2048];
    snprintf(mil, sizeof(mil),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, 2, 1, %d]> a, tensor<fp16, [4, 2, 1, %d]> b) {\n"
        "        tensor<fp16, [4, 2, 1, %d]> y = add(x = a, y = b)[name = tensor<string, []>(\"a\")];\n"
        "    } -> (y);\n"
        "}\n", SP, SP, SP);
    size_t a_sz = 2 * SP * sizeof(uint16_t);
    size_t b_sz = 4 * 2 * SP * sizeof(uint16_t);
    size_t o_sz = 4 * 2 * SP * sizeof(uint16_t);
    size_t in_sizes[2] = {a_sz, b_sz};
    ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &o_sz);
    if (!k) { printf("  COMPILE FAIL\n"); return false; }
    float a_data[2] = {10.0f, 20.0f};
    float b_data[8] = {1,2,3,4,5,6,7,8};
    float* inputs[] = {a_data, b_data};
    int in_chs[] = {2, 8};
    float out[8]; float* op[] = {out}; int och[] = {8};
    bool ok = ane_eval_multi(k, inputs, in_chs, op, och);
    if (ok) {
        printf("  EVAL OK: [%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f]\n",
               out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
        // Expected: a broadcasts on N, so out[n,c] = a[0,c] + b[n,c]
        // out[0,0]=10+1=11, out[0,1]=20+2=22, out[1,0]=10+3=13, ...
        bool correct = true;
        for (int n = 0; n < 4; n++)
            for (int c = 0; c < 2; c++) {
                float exp = a_data[c] + b_data[n*2+c];
                if (fabsf(out[n*2+c] - exp) > 0.1f) {
                    printf("  MISMATCH [%d,%d]: %.1f vs %.1f\n", n, c, out[n*2+c], exp);
                    correct = false;
                }
            }
        if (correct) printf("  All CORRECT!\n");
    } else printf("  EVAL FAIL\n");
    ane_free(k); return ok;
}

static bool test_mul_broadcast_n() {
    int out_dim = 4, T = 2;
    printf("--- 2a: mul [1,%d,1,SP] * [%d,%d,1,SP] (broadcast N) ---\n", T, out_dim, T);
    char mil[2048];
    snprintf(mil, sizeof(mil),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x, tensor<fp16, [%d, %d, 1, %d]> W) {\n"
        "        tensor<fp16, [%d, %d, 1, %d]> y = mul(x = W, y = x)[name = tensor<string, []>(\"m\")];\n"
        "    } -> (y);\n"
        "}\n", T, SP, out_dim, T, SP, out_dim, T, SP);
    size_t x_sz = (size_t)T * SP * sizeof(uint16_t);
    size_t w_sz = (size_t)out_dim * T * SP * sizeof(uint16_t);
    size_t o_sz = (size_t)out_dim * T * SP * sizeof(uint16_t);
    size_t in_sizes[2] = {x_sz, w_sz};
    ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &o_sz);
    if (!k) { printf("  COMPILE FAIL\n"); return false; }
    printf("  Compiled OK\n");

    // x: [1,T,1,SP] — values at position 0 of each channel
    float x_data[2] = {3.0f, 5.0f};
    // W: [out,T,1,SP] — values at position 0
    float w_data[8]; // 4*2
    for (int o = 0; o < out_dim; o++)
        for (int t = 0; t < T; t++)
            w_data[o * T + t] = (float)((o + 1) * (t + 1));

    // Write using tiled API: [N=1, C=T, H=1, W=SP]
    // For x: only SP-strided positions matter
    float* x_tiled = (float*)calloc(1 * T * 1 * SP, sizeof(float));
    for (int t = 0; t < T; t++) x_tiled[t * SP] = x_data[t];
    ane_write_input_tiled(k, 0, x_tiled, 1, T, 1, SP);

    float* w_tiled = (float*)calloc(out_dim * T * 1 * SP, sizeof(float));
    for (int o = 0; o < out_dim; o++)
        for (int t = 0; t < T; t++)
            w_tiled[(o * T + t) * SP] = w_data[o * T + t];
    ane_write_input_tiled(k, 1, w_tiled, out_dim, T, 1, SP);

    // Use ane_eval_multi: pass BOTH inputs (it expects nInputs arrays)
    float* inputs[] = {x_data, w_data};
    int in_chs[] = {T, out_dim * T};
    float out_strided[8]; 
    float* op[] = {out_strided}; int och[] = {out_dim * T};
    bool ok = ane_eval_multi(k, inputs, in_chs, op, och);

    if (ok) {
        printf("  EVAL OK!\n");
        bool correct = true;
        for (int o = 0; o < out_dim; o++) {
            for (int t = 0; t < T; t++) {
                float expected = w_data[o * T + t] * x_data[t];
                float actual = out_strided[o * T + t];
                if (fabsf(actual - expected) > 0.1f) {
                    printf("  MISMATCH [%d,%d]: %.3f vs %.3f\n", o, t, actual, expected);
                    correct = false;
                }
            }
        }
        if (correct) printf("  All values CORRECT!\n");
    } else printf("  EVAL FAIL\n");
    ane_free(k); free(x_tiled); free(w_tiled);
    return ok;
}

// ============================================================
// PHASE 3: reduce_sum and mul+reduce pipeline
// ============================================================

static bool test_reduce_sum() {
    int N = 4, C = 3;
    printf("--- 3a: reduce_sum [%d,%d,1,SP] axis=1 -> [%d,1,1,SP] ---\n", N, C, N);
    char mil[2048];
    snprintf(mil, sizeof(mil),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [%d, %d, 1, %d]> x) {\n"
        "        tensor<int32, [1]> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, [1]>([1])];\n"
        "        tensor<bool, []> kd = const()[name = tensor<string, []>(\"kd\"), val = tensor<bool, []>(true)];\n"
        "        tensor<fp16, [%d, 1, 1, %d]> y = reduce_sum(x = x, axes = ax, keep_dims = kd)"
        "[name = tensor<string, []>(\"rs\")];\n"
        "    } -> (y);\n"
        "}\n", N, C, SP, N, SP);
    size_t in_sz = (size_t)N * C * SP * sizeof(uint16_t);
    size_t out_sz = (size_t)N * 1 * SP * sizeof(uint16_t);
    ANEKernel* k = ane_compile_mil(mil, 1, &in_sz, 1, &out_sz);
    if (!k) { printf("  COMPILE FAIL\n"); return false; }

    // Write: x[n,c] = n*10 + c + 1 (at SP position 0)
    float in_data[12]; // N*C = 12
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
            in_data[n * C + c] = (float)(n * 10 + c + 1);
    float* ip[] = {in_data}; int ich[] = {N * C};
    float out[4]; float* op[] = {out}; int och[] = {N};
    bool ok = ane_eval_multi(k, ip, ich, op, och);
    if (ok) {
        printf("  EVAL OK!\n");
        bool correct = true;
        for (int n = 0; n < N; n++) {
            float expected = 0;
            for (int c = 0; c < C; c++) expected += in_data[n * C + c];
            printf("  n=%d: %.1f expected %.1f %s\n", n, out[n], expected,
                   fabsf(out[n]-expected)<0.5f ? "OK" : "MISMATCH");
            if (fabsf(out[n]-expected) > 0.5f) correct = false;
        }
        if (correct) printf("  All CORRECT!\n");
    } else printf("  EVAL FAIL\n");
    ane_free(k); return ok;
}

static bool test_mul_reduce() {
    int out_dim = 4, T = 2;
    printf("--- 3b: mul+reduce [1,%d,1,SP]*[%d,%d,1,SP] -> [%d,1,1,SP] ---\n",
           T, out_dim, T, out_dim);
    char mil[4096];
    snprintf(mil, sizeof(mil),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x, tensor<fp16, [%d, %d, 1, %d]> W) {\n"
        "        tensor<fp16, [%d, %d, 1, %d]> prod = mul(x = W, y = x)[name = tensor<string, []>(\"m\")];\n"
        "        tensor<int32, [1]> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, [1]>([1])];\n"
        "        tensor<bool, []> kd = const()[name = tensor<string, []>(\"kd\"), val = tensor<bool, []>(true)];\n"
        "        tensor<fp16, [%d, 1, 1, %d]> y = reduce_sum(x = prod, axes = ax, keep_dims = kd)"
        "[name = tensor<string, []>(\"rs\")];\n"
        "    } -> (y);\n"
        "}\n", T, SP, out_dim, T, SP, out_dim, T, SP, out_dim, SP);
    size_t x_sz = (size_t)T * SP * sizeof(uint16_t);
    size_t w_sz = (size_t)out_dim * T * SP * sizeof(uint16_t);
    size_t o_sz = (size_t)out_dim * SP * sizeof(uint16_t);
    size_t in_sizes[2] = {x_sz, w_sz};
    ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &o_sz);
    if (!k) { printf("  COMPILE FAIL\n"); return false; }
    printf("  Compiled OK\n");

    float x_data[2] = {3.0f, 5.0f};
    float w_data[8]; // out*T
    for (int o = 0; o < out_dim; o++)
        for (int t = 0; t < T; t++)
            w_data[o * T + t] = (float)((o + 1) * (t + 1));

    float* inputs[] = {x_data, w_data};
    int in_chs[] = {T, out_dim * T};
    float out[4]; float* op[] = {out}; int och[] = {out_dim};
    bool ok = ane_eval_multi(k, inputs, in_chs, op, och);
    if (ok) {
        printf("  EVAL OK!\n");
        bool correct = true;
        for (int o = 0; o < out_dim; o++) {
            float expected = 0;
            for (int t = 0; t < T; t++)
                expected += w_data[o * T + t] * x_data[t];
            printf("  o=%d: %.1f expected %.1f %s\n", o, out[o], expected,
                   fabsf(out[o]-expected)<0.5f ? "OK" : "MISMATCH");
            if (fabsf(out[o]-expected) > 0.5f) correct = false;
        }
        if (correct) printf("  All CORRECT!\n");
    } else printf("  EVAL FAIL\n");
    ane_free(k); return ok;
}

// ============================================================
// PHASE 4: Full dynamic matvec with CPU spatial reduction
// ============================================================
// x: [1, T, 1, SP], W: [out, T, 1, SP]
// ANE: mul -> [out, T, 1, SP], reduce_sum(axis=1) -> [out, 1, 1, SP]
// CPU: sum SP values per output -> scalar y[o]
// 
// The key: after reduce_sum on tiles, output[o, 0, 0, s] =
//   sum_{t=0}^{T-1} W[o, t*SP+s] * x[t*SP+s]
// Then y[o] = sum_{s=0}^{SP-1} output[o, 0, 0, s]

static bool test_full_matvec(int out_dim, int in_dim, const char* label, bool verbose) {
    int T = (in_dim + SP - 1) / SP;
    printf("=== Full matvec %dx%d (%s) T=%d ===\n", out_dim, in_dim, label, T);

    char mil[4096];
    snprintf(mil, sizeof(mil),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x, tensor<fp16, [%d, %d, 1, %d]> W) {\n"
        "        tensor<fp16, [%d, %d, 1, %d]> prod = mul(x = W, y = x)[name = tensor<string, []>(\"m\")];\n"
        "        tensor<int32, [1]> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, [1]>([1])];\n"
        "        tensor<bool, []> kd = const()[name = tensor<string, []>(\"kd\"), val = tensor<bool, []>(true)];\n"
        "        tensor<fp16, [%d, 1, 1, %d]> y = reduce_sum(x = prod, axes = ax, keep_dims = kd)"
        "[name = tensor<string, []>(\"rs\")];\n"
        "    } -> (y);\n"
        "}\n",
        T, SP, out_dim, T, SP,
        out_dim, T, SP,
        out_dim, SP);

    size_t x_bytes = (size_t)T * SP * sizeof(uint16_t);
    size_t w_bytes = (size_t)out_dim * T * SP * sizeof(uint16_t);
    size_t o_bytes = (size_t)out_dim * SP * sizeof(uint16_t);
    size_t in_sizes[2] = {x_bytes, w_bytes};

    Timer timer;
    ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &o_bytes);
    if (!k) { printf("  COMPILE FAIL\n"); return false; }
    printf("  Compiled (%.0f ms)  W=%.1f MB\n", timer.elapsed_ms(), (float)w_bytes/1e6);

    // Random test data
    srand(42);
    float* x_f32 = (float*)calloc(in_dim, sizeof(float));
    float* W_f32 = (float*)calloc((size_t)out_dim * in_dim, sizeof(float));
    for (int i = 0; i < in_dim; i++)
        x_f32[i] = ((float)(rand() % 2000) / 1000.0f) - 1.0f;
    for (size_t i = 0; i < (size_t)out_dim * in_dim; i++)
        W_f32[i] = ((float)(rand() % 200) / 10000.0f) - 0.01f;

    float* y_ref = (float*)calloc(out_dim, sizeof(float));
    cpu_matvec_f16ref(y_ref, W_f32, x_f32, out_dim, in_dim);

    // Write x: tiled [1, T, 1, SP]
    // x[i] goes to channel (i/SP), spatial (i%SP)
    uint16_t* x_fp16 = (uint16_t*)calloc(T * SP, sizeof(uint16_t));
    for (int i = 0; i < in_dim; i++)
        x_fp16[(i / SP) * SP + (i % SP)] = f32_to_f16(x_f32[i]);
    ane_write_surface_raw(k, 0, x_fp16, x_bytes);

    // Write W: [out, T, 1, SP]
    // W[o, i] at N=o, C=i/SP, spatial=i%SP -> offset (o*T + i/SP)*SP + i%SP
    uint16_t* w_fp16 = (uint16_t*)calloc((size_t)out_dim * T * SP, sizeof(uint16_t));
    for (int o = 0; o < out_dim; o++)
        for (int i = 0; i < in_dim; i++) {
            int t = i / SP, s = i % SP;
            w_fp16[(size_t)(o * T + t) * SP + s] = f32_to_f16(W_f32[(size_t)o * in_dim + i]);
        }
    ane_write_surface_raw(k, 1, w_fp16, w_bytes);

    // Eval + read raw output
    timer.reset();
    float* out_raw = (float*)calloc(out_dim * SP, sizeof(float));
    // First eval, then read
    float dummy; float* dp[] = {&dummy}; int dc[] = {1};
    bool ok = ane_eval_raw_outputs(k, dp, dc);  // just eval
    if (!ok) {
        printf("  EVAL FAIL\n");
        ane_free(k); free(x_f32); free(W_f32); free(y_ref);
        free(x_fp16); free(w_fp16); free(out_raw);
        return false;
    }
    // Now read all SP values per output channel
    ok = ane_read_output_raw(k, 0, out_raw, out_dim * SP);
    double eval_ms = timer.elapsed_ms();

    // CPU reduction: sum across SP for each output row
    float* y_ane = (float*)calloc(out_dim, sizeof(float));
    for (int o = 0; o < out_dim; o++) {
        float sum = 0;
        for (int s = 0; s < SP; s++)
            sum += out_raw[o * SP + s];
        y_ane[o] = sum;
    }

    float mad = max_abs_diff(y_ane, y_ref, out_dim);
    // For fp16 matvec, error scales with in_dim (accumulation of quantization errors)
    // Allow ~0.01 per element accumulated
    float threshold = 0.05f + 0.001f * in_dim;
    bool pass = mad < threshold;
    printf("  Eval %.2f ms  max_abs_diff=%.6f (thr=%.4f)  %s\n", eval_ms, mad, threshold,
           pass ? "PASS" : "FAIL");
    if (verbose || !pass) {
        int show = out_dim < 8 ? out_dim : 8;
        printf("  ANE: ["); for (int i=0;i<show;i++) printf("%.4f%s",y_ane[i],i<show-1?", ":"");
        printf("]\n  REF: ["); for (int i=0;i<show;i++) printf("%.4f%s",y_ref[i],i<show-1?", ":"");
        printf("]\n");
    }

    ane_free(k); free(x_f32); free(W_f32); free(y_ref); free(y_ane);
    free(x_fp16); free(w_fp16); free(out_raw);
    return pass;
}

// ============================================================
// PHASE 5: Benchmark
// ============================================================

static void benchmark_dynamic(int out_dim, int in_dim, int iters) {
    int T = (in_dim + SP - 1) / SP;
    printf("\n=== Benchmark: %dx%d T=%d iters=%d ===\n", out_dim, in_dim, T, iters);

    char mil[4096];
    snprintf(mil, sizeof(mil),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x, tensor<fp16, [%d, %d, 1, %d]> W) {\n"
        "        tensor<fp16, [%d, %d, 1, %d]> prod = mul(x = W, y = x)[name = tensor<string, []>(\"m\")];\n"
        "        tensor<int32, [1]> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, [1]>([1])];\n"
        "        tensor<bool, []> kd = const()[name = tensor<string, []>(\"kd\"), val = tensor<bool, []>(true)];\n"
        "        tensor<fp16, [%d, 1, 1, %d]> y = reduce_sum(x = prod, axes = ax, keep_dims = kd)"
        "[name = tensor<string, []>(\"rs\")];\n"
        "    } -> (y);\n"
        "}\n", T, SP, out_dim, T, SP, out_dim, T, SP, out_dim, SP);

    size_t x_bytes = (size_t)T * SP * sizeof(uint16_t);
    size_t w_bytes = (size_t)out_dim * T * SP * sizeof(uint16_t);
    size_t o_bytes = (size_t)out_dim * SP * sizeof(uint16_t);
    size_t in_sizes[2] = {x_bytes, w_bytes};
    ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &o_bytes);
    if (!k) { printf("  COMPILE FAIL\n"); return; }

    // Pre-create fp16 buffers
    srand(42);
    uint16_t* x_fp16 = (uint16_t*)calloc(T * SP, sizeof(uint16_t));
    uint16_t* w_fp16 = (uint16_t*)calloc((size_t)out_dim * T * SP, sizeof(uint16_t));
    float* x_f32 = (float*)calloc(in_dim, sizeof(float));
    float* W_f32 = (float*)calloc((size_t)out_dim * in_dim, sizeof(float));
    for (int i = 0; i < in_dim; i++) {
        x_f32[i] = ((float)(rand()%2000)/1000.0f) - 1.0f;
        x_fp16[(i/SP)*SP + (i%SP)] = f32_to_f16(x_f32[i]);
    }
    for (int o = 0; o < out_dim; o++)
        for (int i = 0; i < in_dim; i++) {
            float v = ((float)(rand()%200)/10000.0f) - 0.01f;
            W_f32[(size_t)o*in_dim+i] = v;
            w_fp16[(size_t)(o*(in_dim/SP+((in_dim%SP)?1:0)) + i/SP)*SP + i%SP] = f32_to_f16(v);
        }
    // Recompute w_fp16 properly
    for (size_t i = 0; i < (size_t)out_dim * T * SP; i++) w_fp16[i] = 0;
    for (int o = 0; o < out_dim; o++)
        for (int i = 0; i < in_dim; i++)
            w_fp16[(size_t)(o*T + i/SP)*SP + i%SP] = f32_to_f16(W_f32[(size_t)o*in_dim+i]);

    float* y_buf = (float*)calloc(out_dim, sizeof(float));
    float* out_raw = (float*)calloc(out_dim * SP, sizeof(float));

    // Warmup
    for (int i = 0; i < 5; i++) {
        ane_write_surface_raw(k, 0, x_fp16, x_bytes);
        ane_write_surface_raw(k, 1, w_fp16, w_bytes);
        float d; float* dp[]={&d}; int dc[]={1};
        ane_eval_raw_outputs(k, dp, dc);
    }

    // Full pipeline: write x + write W + eval + read + CPU reduce
    Timer timer;
    for (int i = 0; i < iters; i++) {
        ane_write_surface_raw(k, 0, x_fp16, x_bytes);
        ane_write_surface_raw(k, 1, w_fp16, w_bytes);
        float d; float* dp[]={&d}; int dc[]={1};
        ane_eval_raw_outputs(k, dp, dc);
        ane_read_output_raw(k, 0, out_raw, out_dim * SP);
        for (int o = 0; o < out_dim; o++) {
            float sum = 0;
            for (int s = 0; s < SP; s++) sum += out_raw[o*SP+s];
            y_buf[o] = sum;
        }
    }
    double total_ms = timer.elapsed_ms();
    double per_iter = total_ms / iters;

    // W memcpy only
    timer.reset();
    for (int i = 0; i < iters; i++)
        ane_write_surface_raw(k, 1, w_fp16, w_bytes);
    double memcpy_ms = timer.elapsed_ms() / iters;

    // Eval only
    timer.reset();
    for (int i = 0; i < iters; i++) {
        float d; float* dp[]={&d}; int dc[]={1};
        ane_eval_raw_outputs(k, dp, dc);
    }
    double eval_ms = timer.elapsed_ms() / iters;

    printf("  W: %.2f MB fp16\n", (float)w_bytes/1e6);
    printf("  Total: %.3f ms  (W memcpy: %.3f, ANE eval: %.3f, rest: %.3f)\n",
           per_iter, memcpy_ms, eval_ms, per_iter - memcpy_ms - eval_ms);
    printf("  Throughput: %.0f matvecs/sec\n", 1000.0 / per_iter);

    // Compare: const-weight conv
    uint16_t* bf16 = (uint16_t*)calloc((size_t)out_dim*in_dim, sizeof(uint16_t));
    for (size_t i = 0; i < (size_t)out_dim*in_dim; i++) bf16[i] = f32_to_bf16(W_f32[i]);
    Timer ct;
    ANEKernel* kc = ane_compile_matmul(bf16, out_dim, in_dim);
    double cc_ms = ct.elapsed_ms();
    if (kc) {
        for (int i = 0; i < 5; i++) ane_matvec(kc, y_buf, x_f32, in_dim, out_dim);
        timer.reset();
        for (int i = 0; i < iters; i++) ane_matvec(kc, y_buf, x_f32, in_dim, out_dim);
        double conv_ms = timer.elapsed_ms() / iters;
        printf("  Const conv: %.3f ms (compile %.0f ms)  ratio: %.1fx\n",
               conv_ms, cc_ms, per_iter / conv_ms);
        ane_free(kc);
    }

    ane_free(k); free(x_fp16); free(w_fp16); free(x_f32); free(W_f32);
    free(y_buf); free(out_raw); free(bf16);
}

// ============================================================
// main
// ============================================================
int main() {
    void* pool = objc_autoreleasePoolPush();
    g_verbose = false;
    ane_init();
    if (!ane_available()) { fprintf(stderr, "ANE not available!\n"); return 1; }
    ane_set_persist_cache(false);

    printf("============================================\n");
    printf("  Dynamic Matvec Investigation v10\n");
    printf("============================================\n\n");

    // Phase 1
    printf("====== PHASE 1: Reshape on runtime tensors ======\n");
    bool r1a = test_reshape_same_n();
    bool r1b = test_reshape_changes_n();
    bool r1c = test_reshape_output_n();
    printf("\n  1a N unchanged: %s\n  1b N changes: %s\n  1c output N: %s\n\n",
           r1a?"WORKS":"BROKEN", r1b?"WORKS":"BROKEN", r1c?"WORKS":"BROKEN");

    // Phase 2
    printf("====== PHASE 2: Operations with N broadcast ======\n");
    bool r2pre = test_add_broadcast_n();
    bool r2 = test_mul_broadcast_n();
    printf("\n  2pre add broadcast: %s\n  2a mul broadcast: %s\n\n",
           r2pre?"WORKS":"BROKEN", r2?"WORKS":"BROKEN");

    // Phase 3
    printf("====== PHASE 3: reduce_sum ======\n");
    bool r3a = test_reduce_sum();
    bool r3b = test_mul_reduce();
    printf("\n  3a reduce alone: %s\n  3b mul+reduce: %s\n\n",
           r3a?"WORKS":"BROKEN", r3b?"WORKS":"BROKEN");

    // Phase 4
    printf("====== PHASE 4: Full dynamic matvec ======\n");
    bool p4a = test_full_matvec(8, 32, "8x32", true);
    bool p4b = test_full_matvec(32, 64, "32x64", true);
    bool p4c = test_full_matvec(64, 64, "64x64", false);
    bool p4d = test_full_matvec(128, 128, "128x128", false);
    bool p4e = test_full_matvec(256, 256, "256x256", false);
    printf("\n  8x32:%s  32x64:%s  64x64:%s  128:%s  256:%s\n\n",
           p4a?"PASS":"FAIL", p4b?"PASS":"FAIL", p4c?"PASS":"FAIL",
           p4d?"PASS":"FAIL", p4e?"PASS":"FAIL");

    // Phase 5
    if (p4a && p4b && p4c) {
        printf("====== PHASE 5: Scale + Benchmark ======\n");
        bool s1 = test_full_matvec(512, 512, "512", false);
        bool s2 = test_full_matvec(1024, 1024, "1K", false);
        bool s3 = test_full_matvec(2560, 2560, "2560", false);
        printf("\n  512:%s  1K:%s  2560:%s\n",
               s1?"PASS":"FAIL", s2?"PASS":"FAIL", s3?"PASS":"FAIL");
        if (s1) benchmark_dynamic(512, 512, 100);
        if (s2) benchmark_dynamic(1024, 1024, 50);
        if (s3) benchmark_dynamic(2560, 2560, 20);
    }

    printf("\n  compiles=%d\n", ane_compile_count());
    objc_autoreleasePoolPop(pool);
    return 0;
}
