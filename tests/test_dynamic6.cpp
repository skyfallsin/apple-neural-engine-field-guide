// test_dynamic6.cpp — THE KEY HYPOTHESIS:
// ANE runtime input IOSurfaces MUST have innermost dim = SP (32).
// Working: add(a:[1,C,1,SP], b:[1,C,1,SP]) — both inputs have W=SP
// Failing: mul(x:[1,D,1,SP], W:[D,D,1,1]) — W has innermost dim=1, not SP
//
// Solution: declare weight as [out*in, 1, 1, SP] (flatten to 1D, with SP spatial)
// Then manually compute the matvec using elementwise ops.
// 
// Alternative: Pass weight as [1, out*in, 1, SP] and reshape inside the MIL.
// Or even better: reshape x and W so multiply+reduce works with SP everywhere.
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

// ===== Test A: Two inputs BOTH with [*, *, 1, SP] shape =====
// Weight as [1, out*in, 1, SP] — flattened, then reshape inside MIL
static bool test_weight_as_flat_sp() {
    printf("=== Test A: Weight tensor with SP as innermost dim ===\n");
    int D = 8;
    
    // Strategy: W input is [1, D*D, 1, SP], contains weight matrix flattened.
    // Inside MIL: reshape W to [D, D, 1, SP] then use with x.
    // But reshape might change the physical layout... let's try.
    char mil[4096];
    snprintf(mil, sizeof(mil),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x, tensor<fp16, [1, %d, 1, %d]> Wflat) {\n"
        // Reshape Wflat from [1, D*D, 1, SP] to [D, D, 1, SP]
        "        tensor<int32, [4]> ws = const()[name = tensor<string, []>(\"ws\"), val = tensor<int32, [4]>([%d, %d, 1, %d])];\n"
        "        tensor<fp16, [%d, %d, 1, %d]> W = reshape(x = Wflat, shape = ws)[name = tensor<string, []>(\"rw\")];\n"
        // mul: W[D,D,1,SP] * x[1,D,1,SP] -> [D,D,1,SP] (broadcast dim 0)
        "        tensor<fp16, [%d, %d, 1, %d]> prod = mul(x = W, y = x)[name = tensor<string, []>(\"m\")];\n"
        // reduce_sum axis=1 -> [D,1,1,SP]
        "        tensor<int32, [1]> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, [1]>([1])];\n"
        "        tensor<bool, []> kd = const()[name = tensor<string, []>(\"kd\"), val = tensor<bool, []>(false)];\n"
        "        tensor<fp16, [%d, 1, 1, %d]> s = reduce_sum(x = prod, axes = ax, keep_dims = kd)"
        "[name = tensor<string, []>(\"rs\")];\n"
        // reshape to [1, D, 1, SP]
        "        tensor<int32, [4]> os = const()[name = tensor<string, []>(\"os\"), val = tensor<int32, [4]>([1, %d, 1, %d])];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y = reshape(x = s, shape = os)[name = tensor<string, []>(\"yr\")];\n"
        "    } -> (y);\n"
        "}\n",
        D, SP, D*D, SP,
        D, D, SP,
        D, D, SP,
        D, D, SP,
        D, SP,
        D, SP,
        D, SP);
    
    // BOTH inputs have SP as innermost dim
    size_t in_sizes[2] = {
        (size_t)D * SP * sizeof(uint16_t),       // x: [1, D, 1, SP]
        (size_t)D * D * SP * sizeof(uint16_t)     // Wflat: [1, D*D, 1, SP]
    };
    size_t out_size = (size_t)D * SP * sizeof(uint16_t);
    
    ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &out_size);
    if (!k) { printf("  Compile FAILED\n"); return false; }
    printf("  Compiled OK\n");
    
    // Identity matrix test: y should equal x
    float* x = (float*)calloc(D, sizeof(float));
    for (int i = 0; i < D; i++) x[i] = (float)(i + 1);
    
    // Write x SP-strided
    uint16_t* x_fp16 = (uint16_t*)calloc(D, sizeof(uint16_t));
    for (int i = 0; i < D; i++) x_fp16[i] = f32_to_f16(x[i]);
    ane_write_surface_strided(k, 0, x_fp16, D);
    
    // Write W as [1, D*D, 1, SP] — each "channel" (row of the matrix) at SP stride
    // Channel c contains W[c/D, c%D] (row-major flatten)
    // Each channel position: only position 0 of SP has the value
    uint16_t* w_buf = (uint16_t*)calloc(D * D * SP, sizeof(uint16_t));
    for (int o = 0; o < D; o++)
        for (int i = 0; i < D; i++) {
            float val = (o == i) ? 1.0f : 0.0f;
            w_buf[(size_t)(o * D + i) * SP] = f32_to_f16(val);
        }
    ane_write_surface_raw(k, 1, w_buf, D * D * SP * sizeof(uint16_t));
    
    float* y = (float*)calloc(D, sizeof(float));
    float* outptrs[] = {y};
    int out_chs[] = {D};
    bool ok = ane_eval_raw_outputs(k, outptrs, out_chs);
    
    if (!ok) { printf("  EVAL FAILED!\n"); }
    else {
        float mad = max_abs_diff(y, x, D);
        printf("  Identity: max_abs_diff = %.6f %s\n", mad, mad < 0.05f ? "PASS" : "FAIL");
        printf("  y = ["); for (int i = 0; i < D; i++) printf("%.2f%s", y[i], i<D-1?", ":""); printf("]\n");
        printf("  x = ["); for (int i = 0; i < D; i++) printf("%.2f%s", x[i], i<D-1?", ":""); printf("]\n");
    }
    
    ane_free(k);
    free(x); free(x_fp16); free(w_buf); free(y);
    return ok;
}

// ===== Test B: Even simpler — both inputs same shape, pure elementwise =====
// If two [1, D, 1, SP] inputs work for add/mul, what about [D, 1, 1, SP]?
// Let's test various 4D shapes to find which dimension configs work.
static bool test_shape_combos() {
    printf("\n=== Test B: Which input tensor shapes work for eval? ===\n");
    
    struct TestCase {
        int n1, c1, h1, w1;  // input 1 shape
        int n2, c2, h2, w2;  // input 2 shape (same for output)
        const char* desc;
    };
    
    TestCase cases[] = {
        // Both SP
        {1, 8, 1, SP,  1, 8, 1, SP,  "[1,8,1,SP]+[1,8,1,SP] add"},
        // One has W=1
        {1, 8, 1, SP,  1, 8, 1, 1,   "[1,8,1,SP]+[1,8,1,1] add"},
        // N>1 with SP
        {4, 8, 1, SP,  4, 8, 1, SP,  "[4,8,1,SP]+[4,8,1,SP] add"},
        // N>1 with W=1
        {4, 8, 1, 1,   4, 8, 1, 1,   "[4,8,1,1]+[4,8,1,1] add"},
        // Different N
        {8, 8, 1, SP,  1, 8, 1, SP,  "[8,8,1,SP]+[1,8,1,SP] add (broadcast N)"},
        // H=SP instead of W
        {1, 8, SP, 1,  1, 8, SP, 1,  "[1,8,SP,1]+[1,8,SP,1] add"},
    };
    
    int n_pass = 0, n_total = 0;
    for (auto& tc : cases) {
        n_total++;
        printf("  %-50s ", tc.desc);
        fflush(stdout);
        
        char mil[2048];
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [%d, %d, %d, %d]> a, tensor<fp16, [%d, %d, %d, %d]> b) {\n"
            "        tensor<fp16, [%d, %d, %d, %d]> y = add(x = a, y = b)[name = tensor<string, []>(\"add\")];\n"
            "    } -> (y);\n"
            "}\n",
            tc.n1, tc.c1, tc.h1, tc.w1,
            tc.n2, tc.c2, tc.h2, tc.w2,
            // Output shape: broadcast result
            (tc.n1 > tc.n2 ? tc.n1 : tc.n2),
            (tc.c1 > tc.c2 ? tc.c1 : tc.c2),
            (tc.h1 > tc.h2 ? tc.h1 : tc.h2),
            (tc.w1 > tc.w2 ? tc.w1 : tc.w2));
        
        size_t s1 = (size_t)tc.n1 * tc.c1 * tc.h1 * tc.w1 * sizeof(uint16_t);
        size_t s2 = (size_t)tc.n2 * tc.c2 * tc.h2 * tc.w2 * sizeof(uint16_t);
        int on = tc.n1 > tc.n2 ? tc.n1 : tc.n2;
        int oc = tc.c1 > tc.c2 ? tc.c1 : tc.c2;
        int oh = tc.h1 > tc.h2 ? tc.h1 : tc.h2;
        int ow = tc.w1 > tc.w2 ? tc.w1 : tc.w2;
        size_t so = (size_t)on * oc * oh * ow * sizeof(uint16_t);
        
        size_t in_sizes[2] = {s1, s2};
        ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &so);
        if (!k) { printf("compile FAIL\n"); continue; }
        
        // Write ones to both surfaces
        int n1_ch = tc.n1 * tc.c1 * tc.h1;
        int n2_ch = tc.n2 * tc.c2 * tc.h2;
        // Just zero everything and write a few values
        uint16_t* buf1 = (uint16_t*)calloc(s1/2, sizeof(uint16_t));
        uint16_t* buf2 = (uint16_t*)calloc(s2/2, sizeof(uint16_t));
        // Write 1.0 to first position
        buf1[0] = f32_to_f16(1.0f);
        buf2[0] = f32_to_f16(2.0f);
        ane_write_surface_raw(k, 0, buf1, s1);
        ane_write_surface_raw(k, 1, buf2, s2);
        
        int o_ch = on * oc * oh;
        float* y = (float*)calloc(o_ch * ow, sizeof(float));
        float* outptrs[] = {y};
        int out_chs[] = {o_ch};
        bool ok = ane_eval_raw_outputs(k, outptrs, out_chs);
        
        if (ok) { printf("EVAL OK  y[0]=%.2f\n", y[0]); n_pass++; }
        else    { printf("EVAL FAIL\n"); }
        
        ane_free(k); free(buf1); free(buf2); free(y);
    }
    
    printf("  %d/%d passed\n", n_pass, n_total);
    return n_pass > 0;
}

// ===== Test C: Dense IOSurface for [D,D,1,1] =====
// Maybe ANE expects dense packing (no SP padding) for tensors with W=1
static bool test_dense_weight_surface() {
    printf("\n=== Test C: Dense IOSurface for [D,D,1,1] weight ===\n");
    int D = 8;
    
    char mil[2048];
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
    
    // Dense: D*D*2 bytes for W
    size_t in_sizes[2] = {
        (size_t)D * SP * sizeof(uint16_t),   // x
        (size_t)D * D * sizeof(uint16_t)     // W dense
    };
    size_t out_size = (size_t)D * SP * sizeof(uint16_t);
    
    ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &out_size);
    if (!k) { printf("  Compile FAILED\n"); return false; }
    printf("  Compiled OK (dense W surface = %zu bytes)\n", (size_t)D*D*2);
    
    uint16_t* x_fp16 = (uint16_t*)calloc(D, sizeof(uint16_t));
    for (int i = 0; i < D; i++) x_fp16[i] = f32_to_f16((float)(i+1));
    ane_write_surface_strided(k, 0, x_fp16, D);
    
    // Dense weight: identity matrix
    uint16_t* w_fp16 = (uint16_t*)calloc(D*D, sizeof(uint16_t));
    for (int o = 0; o < D; o++)
        for (int i = 0; i < D; i++)
            w_fp16[o*D+i] = f32_to_f16((o==i) ? 1.0f : 0.0f);
    ane_write_surface_raw(k, 1, w_fp16, D*D*sizeof(uint16_t));
    
    float* y = (float*)calloc(D, sizeof(float));
    float* outptrs[] = {y};
    int out_chs[] = {D};
    bool ok = ane_eval_raw_outputs(k, outptrs, out_chs);
    
    if (ok) {
        printf("  EVAL OK! y = [");
        for (int i = 0; i < D; i++) printf("%.2f%s", y[i], i<D-1?", ":"");
        printf("]\n");
    } else {
        printf("  EVAL FAILED (dense)\n");
    }
    
    ane_free(k); free(x_fp16); free(w_fp16); free(y);
    return ok;
}

int main() {
    void* pool = objc_autoreleasePoolPush();
    g_verbose = false;
    ane_init();
    if (!ane_available()) { fprintf(stderr, "ANE not available!\n"); return 1; }
    ane_set_persist_cache(false);
    
    bool a = test_weight_as_flat_sp();
    test_shape_combos();
    test_dense_weight_surface();
    
    printf("\ncompiles=%d\n", ane_compile_count());
    objc_autoreleasePoolPop(pool);
    return 0;
}
