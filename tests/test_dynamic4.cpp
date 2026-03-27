// test_dynamic4.cpp — Fresh investigation of dynamic weight conv failure
// Tests three hypotheses:
//  1) Do multi-input programs actually EVAL correctly? (add/mul with 2 IOSurface inputs)
//  2) Can mul+reduce_sum do manual matvec on ANE? (bypass conv weight path)
//  3) Does the weightsBuffer parameter on ANERequest enable runtime weight injection?
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

// ===== Test 1: Multi-input EVAL — does add(a,b) produce correct results? =====
static bool test_multi_input_add() {
    printf("=== Test 1: Multi-input EVAL (add two inputs) ===\n");
    
    int C = 64;
    char mil[2048];
    snprintf(mil, sizeof(mil),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> a, tensor<fp16, [1, %d, 1, %d]> b) {\n"
        "        tensor<fp16, [1, %d, 1, %d]> y = add(x = a, y = b)[name = tensor<string, []>(\"add\")];\n"
        "    } -> (y);\n"
        "}\n", C, SP, C, SP, C, SP);
    
    size_t sz = (size_t)C * SP * sizeof(uint16_t);
    size_t in_sizes[2] = {sz, sz};
    size_t out_size = sz;
    
    ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &out_size);
    if (!k) { printf("  Compile FAILED\n"); return false; }
    printf("  Compiled OK (nInputs=%d, nOutputs=%d)\n", ane_get_n_inputs(k), ane_get_n_outputs(k));
    
    // Prepare inputs
    float* a = (float*)calloc(C, sizeof(float));
    float* b = (float*)calloc(C, sizeof(float));
    float* y = (float*)calloc(C, sizeof(float));
    float* y_ref = (float*)calloc(C, sizeof(float));
    for (int i = 0; i < C; i++) {
        a[i] = (float)(i + 1) / C;
        b[i] = (float)(C - i) / C;
        y_ref[i] = a[i] + b[i]; // should all be ~1.0
    }
    
    float* inputs[] = {a, b};
    int in_ch[] = {C, C};
    float* outputs[] = {y};
    int out_ch[] = {C};
    
    bool ok = ane_eval_multi(k, inputs, in_ch, outputs, out_ch);
    if (!ok) {
        printf("  EVAL FAILED!\n");
        ane_free(k); free(a); free(b); free(y); free(y_ref);
        return false;
    }
    
    float mad = max_abs_diff(y, y_ref, C);
    printf("  EVAL OK! max_abs_diff = %.6f %s\n", mad, mad < 0.01f ? "PASS" : "FAIL");
    printf("  y[0..4] = [%.4f, %.4f, %.4f, %.4f, %.4f] (expect ~1.0)\n",
           y[0], y[1], y[2], y[3], y[4]);
    
    ane_free(k); free(a); free(b); free(y); free(y_ref);
    return mad < 0.01f;
}

// ===== Test 2: Multi-input EVAL — mul(a,b) =====
static bool test_multi_input_mul() {
    printf("\n=== Test 2: Multi-input EVAL (mul two inputs) ===\n");
    
    int C = 64;
    char mil[2048];
    snprintf(mil, sizeof(mil),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> a, tensor<fp16, [1, %d, 1, %d]> b) {\n"
        "        tensor<fp16, [1, %d, 1, %d]> y = mul(x = a, y = b)[name = tensor<string, []>(\"mul\")];\n"
        "    } -> (y);\n"
        "}\n", C, SP, C, SP, C, SP);
    
    size_t sz = (size_t)C * SP * sizeof(uint16_t);
    size_t in_sizes[2] = {sz, sz};
    size_t out_size = sz;
    
    ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &out_size);
    if (!k) { printf("  Compile FAILED\n"); return false; }
    
    float* a = (float*)calloc(C, sizeof(float));
    float* b = (float*)calloc(C, sizeof(float));
    float* y = (float*)calloc(C, sizeof(float));
    float* y_ref = (float*)calloc(C, sizeof(float));
    for (int i = 0; i < C; i++) {
        a[i] = 2.0f;
        b[i] = (float)(i + 1) / C;
        y_ref[i] = a[i] * b[i];
    }
    
    float* inputs[] = {a, b};
    int in_ch[] = {C, C};
    float* outputs[] = {y};
    int out_ch[] = {C};
    
    bool ok = ane_eval_multi(k, inputs, in_ch, outputs, out_ch);
    if (!ok) { printf("  EVAL FAILED!\n"); ane_free(k); free(a); free(b); free(y); free(y_ref); return false; }
    
    float mad = max_abs_diff(y, y_ref, C);
    printf("  EVAL OK! max_abs_diff = %.6f %s\n", mad, mad < 0.01f ? "PASS" : "FAIL");
    printf("  y[0..4] = [%.4f, %.4f, %.4f, %.4f, %.4f]\n", y[0], y[1], y[2], y[3], y[4]);
    printf("  ref[0..4] = [%.4f, %.4f, %.4f, %.4f, %.4f]\n", y_ref[0], y_ref[1], y_ref[2], y_ref[3], y_ref[4]);
    
    ane_free(k); free(a); free(b); free(y); free(y_ref);
    return mad < 0.01f;
}

// ===== Test 3: mul + reduce_sum manual matvec =====
// W:[D,D,1,1] * x:[1,D,1,SP] broadcasts to [D,D,1,SP], reduce_sum axis=1 -> [D,1,1,SP]
// reshape to [1,D,1,SP] = output
static bool test_mul_reduce_sum_matvec() {
    printf("\n=== Test 3: mul + reduce_sum manual matvec ===\n");
    
    int D = 8; // small first
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
    
    // IOSurface sizes:
    // x: [1, D, 1, SP] = D * SP * 2 bytes
    // W: [D, D, 1, 1] — critical question: what size?
    // The MIL says [D,D,1,1], innermost dim W=1.
    // ANE may or may not pad W=1 to SP. Let's try BOTH.
    
    // Attempt A: W surface = D*D*2 bytes (dense, no SP padding)
    printf("  Attempt A: W as dense [%d,%d,1,1] = %zu bytes\n", D, D, (size_t)D*D*2);
    {
        size_t in_sizes[2] = {
            (size_t)D * SP * sizeof(uint16_t),  // x
            (size_t)D * D * sizeof(uint16_t)     // W dense
        };
        size_t out_size = (size_t)D * SP * sizeof(uint16_t);
        
        ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &out_size);
        if (!k) {
            printf("    Compile FAILED\n");
        } else {
            printf("    Compiled OK\n");
            
            // Write x as SP-strided
            float* x = (float*)calloc(D, sizeof(float));
            for (int i = 0; i < D; i++) x[i] = (float)(i + 1);
            
            // Write W as dense fp16 into IOSurface 1
            uint16_t* W_fp16 = (uint16_t*)calloc(D * D, sizeof(uint16_t));
            float W_f32[64]; // D*D max
            for (int o = 0; o < D; o++)
                for (int i = 0; i < D; i++) {
                    float v = (o == i) ? 1.0f : 0.0f; // identity matrix
                    W_f32[o * D + i] = v;
                    W_fp16[o * D + i] = f32_to_f16(v);
                }
            
            // Write x to input 0 (strided)
            float* inputs[] = {x, nullptr};
            int in_ch[] = {D, 0};
            float* y = (float*)calloc(D, sizeof(float));
            float* outputs[] = {y};
            int out_ch[] = {D};
            
            // Write W raw to input 1
            ane_write_surface_raw(k, 1, W_fp16, D * D * sizeof(uint16_t));
            
            // Write x via strided write
            uint16_t* x_fp16 = (uint16_t*)calloc(D, sizeof(uint16_t));
            for (int i = 0; i < D; i++) x_fp16[i] = f32_to_f16(x[i]);
            ane_write_surface_strided(k, 0, x_fp16, D);
            
            // Now eval (using ane_eval_multi but with pre-written surfaces)
            // Actually we need a raw eval. Let's write all surfaces manually then eval.
            // ane_eval_multi overwrites the surfaces. Use it with dummy data for input 0
            // and skip input 1 write... 
            // Better: let's just use ane_eval_multi for the x input and pre-write W.
            // But ane_eval_multi writes ALL inputs. We need a raw eval function.
            // Let me just use ane_eval_multi with the W data as "channels" = D*D
            // But that would SP-stride every element, which is wrong for dense W.
            
            // OK — let me try: write both surfaces manually, then call eval_multi 
            // with 0 channels (so it doesn't overwrite them), but that would memset to 0.
            // This API needs a "just eval" function. Let me write it differently.
            printf("    (Need raw eval — testing with ane_dynamic_conv_eval pattern)\n");
            printf("    Skipping eval for now, testing hypothesis B instead\n");
            
            ane_free(k);
            free(x); free(W_fp16); free(y); free(x_fp16);
        }
    }
    
    // Attempt B: W surface = D*D*SP*2 bytes (SP-padded, like activation tensors)
    printf("  Attempt B: W with SP padding [%d,%d,1,SP] = %zu bytes\n", D, D, (size_t)D*D*SP*2);
    {
        size_t in_sizes[2] = {
            (size_t)D * SP * sizeof(uint16_t),      // x
            (size_t)D * D * SP * sizeof(uint16_t)    // W with SP padding
        };
        size_t out_size = (size_t)D * SP * sizeof(uint16_t);
        
        ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &out_size);
        if (!k) {
            printf("    Compile FAILED\n");
        } else {
            printf("    Compiled OK\n");
            
            // For W:[D,D,1,1] with SP padding, each element W[o,i,0,0] 
            // would be at offset (o * D * 1 * SP + i * 1 * SP + 0) * 2
            // = (o * D * SP + i * SP) * 2
            // But wait — the tensor shape is [D,D,1,1], dims are N=D, C=D, H=1, W=1.
            // SP pads the innermost dim W. So W[n,c,h,w=0] at offset (n*C*H*SP + c*H*SP + h*SP + w)*2
            // = (n * D * SP + c * SP) * 2
            // Total: D * D * SP * 2 bytes
            
            // But mul broadcasts W:[D,D,1,1] * x:[1,D,1,SP] -> [D,D,1,SP]
            // For this to work, x needs to broadcast along dim0 (N: 1->D)
            // and W needs to broadcast along dim3 (W: 1->SP)
            // ANE may handle the broadcast internally... but the IOSurface still 
            // needs the data in the right layout.
            
            // For W, element [o,i,0,0] should be readable at position 0 of the 
            // innermost dim. With SP padding, that's base[(o*D*SP + i*SP) * sizeof(fp16)]
            
            uint16_t* W_fp16 = (uint16_t*)calloc(D * D * SP, sizeof(uint16_t));
            for (int o = 0; o < D; o++)
                for (int i = 0; i < D; i++)
                    W_fp16[o * D * SP + i * SP] = f32_to_f16((o == i) ? 1.0f : 0.0f);
            
            ane_write_surface_raw(k, 1, W_fp16, D * D * SP * sizeof(uint16_t));
            
            // Write x with SP stride
            uint16_t* x_fp16 = (uint16_t*)calloc(D * SP, sizeof(uint16_t));
            for (int i = 0; i < D; i++) x_fp16[i * SP] = f32_to_f16((float)(i + 1));
            ane_write_surface_raw(k, 0, x_fp16, D * SP * sizeof(uint16_t));
            
            // Eval
            float* y = (float*)calloc(D, sizeof(float));
            float* dummy_inputs[] = {nullptr, nullptr};
            int dummy_in_ch[] = {0, 0};
            float* outptrs[] = {y};
            int out_chs[] = {D};
            
            // We've pre-written the surfaces. Use ane_eval_multi but it will overwrite!
            // We need a function that just evals without writing inputs.
            // For now, let me hack: write inputs via ane_eval_multi's write path
            // by passing the correct float data.
            float x_f32[8];
            for (int i = 0; i < D; i++) x_f32[i] = (float)(i + 1);
            
            // Problem: ane_eval_multi writes inputs as SP-strided fp16 from float,
            // but for W we need the special layout. Can't use ane_eval_multi for mixed layouts.
            // Need to add a raw eval that doesn't touch inputs.
            printf("    (Need raw eval without input overwrite — adding to API)\n");
            
            ane_free(k);
            free(W_fp16); free(x_fp16); free(y);
        }
    }
    
    return false; // need raw eval
}

int main() {
    void* pool = objc_autoreleasePoolPush();
    g_verbose = false;
    ane_init();
    if (!ane_available()) { fprintf(stderr, "ANE not available!\n"); return 1; }
    ane_set_persist_cache(false);
    
    bool t1 = test_multi_input_add();
    bool t2 = test_multi_input_mul();
    test_mul_reduce_sum_matvec();
    
    printf("\n=== Summary ===\n");
    printf("  Multi-input add eval: %s\n", t1 ? "PASS" : "FAIL");
    printf("  Multi-input mul eval: %s\n", t2 ? "PASS" : "FAIL");
    printf("  compiles=%d\n", ane_compile_count());
    
    objc_autoreleasePoolPop(pool);
    return 0;
}
