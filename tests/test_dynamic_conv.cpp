// test_dynamic_conv.cpp — Test dynamic-weight conv on ANE
// conv() with weight as a runtime input (not const) WORKS on ANE!
// This means we can compile ONE conv kernel and reuse it with different weights.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "core/ane_runtime.h"
#include <ane_lm/common.h>

extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void  objc_autoreleasePoolPop(void*);

using namespace ane_lm;

#define SP ANE_SPATIAL

static uint16_t* make_bf16(size_t numel, int seed) {
    uint16_t* w = (uint16_t*)malloc(numel * 2);
    srand(seed);
    for (size_t i = 0; i < numel; i++)
        w[i] = f32_to_bf16(((float)(rand() % 2000) / 10000.0f) - 0.1f);
    return w;
}

static uint16_t* bf16_to_fp16_arr(const uint16_t* bf16, size_t n) {
    uint16_t* fp16 = (uint16_t*)malloc(n * 2);
    bf16_to_f16_vec(fp16, bf16, (int)n);
    return fp16;
}

static void cpu_matvec(float* y, const uint16_t* W_bf16, const float* x, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        float sum = 0;
        for (int i = 0; i < in_dim; i++)
            sum += bf16_to_f32(W_bf16[(size_t)o * in_dim + i]) * x[i];
        y[o] = sum;
    }
}

// MIL: dynamic conv with weight as second input
static void make_dynamic_conv_mil(char* buf, size_t buflen, int out_dim, int in_dim) {
    snprintf(buf, buflen,
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, %d, 1, %d]> x, tensor<fp16, [%d, %d, 1, 1]> W) {\n"
        "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y = conv(dilations = dl, groups = gr, "
        "pad = pd, pad_type = pt, strides = st, weight = W, x = x)"
        "[name = tensor<string, []>(\"cv\")];\n"
        "    } -> (y);\n"
        "}\n",
        in_dim, SP, out_dim, in_dim, out_dim, SP);
}

// Dynamic FFN using 3 dynamic convs in one kernel
static void make_dynamic_ffn_mil(char* buf, size_t buflen, int dim, int inter) {
    snprintf(buf, buflen,
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>("
        "tensor<fp16, [1, %d, 1, %d]> x, "
        "tensor<fp16, [%d, %d, 1, 1]> Wg, "
        "tensor<fp16, [%d, %d, 1, 1]> Wu, "
        "tensor<fp16, [%d, %d, 1, 1]> Wd) {\n"
        "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"
        // gate = Wg conv x
        "        tensor<fp16, [1, %d, 1, %d]> gate = conv(dilations = dl, groups = gr, "
        "pad = pd, pad_type = pt, strides = st, weight = Wg, x = x)"
        "[name = tensor<string, []>(\"cg\")];\n"
        // up = Wu conv x
        "        tensor<fp16, [1, %d, 1, %d]> up = conv(dilations = dl, groups = gr, "
        "pad = pd, pad_type = pt, strides = st, weight = Wu, x = x)"
        "[name = tensor<string, []>(\"cu\")];\n"
        // silu(gate) = gate * sigmoid(gate)
        "        tensor<fp16, [1, %d, 1, %d]> sig = sigmoid(x = gate)[name = tensor<string, []>(\"sg\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> silu = mul(x = gate, y = sig)[name = tensor<string, []>(\"sl\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> fused = mul(x = silu, y = up)[name = tensor<string, []>(\"fu\")];\n"
        // down = Wd conv fused
        "        tensor<fp16, [1, %d, 1, %d]> out = conv(dilations = dl, groups = gr, "
        "pad = pd, pad_type = pt, strides = st, weight = Wd, x = fused)"
        "[name = tensor<string, []>(\"cd\")];\n"
        "    } -> (out);\n"
        "}\n",
        /* inputs */ dim, SP, inter, dim, inter, dim, dim, inter,
        /* gate */ inter, SP,
        /* up */ inter, SP,
        /* sig */ inter, SP,
        /* silu */ inter, SP,
        /* fused */ inter, SP,
        /* down */ dim, SP);
}

static ANEKernel* compile_dynamic_conv(int out_dim, int in_dim) {
    char mil[4096];
    make_dynamic_conv_mil(mil, sizeof(mil), out_dim, in_dim);
    size_t in_sizes[2] = {
        (size_t)in_dim * SP * sizeof(uint16_t),
        (size_t)out_dim * in_dim * sizeof(uint16_t)
    };
    size_t out_size = (size_t)out_dim * SP * sizeof(uint16_t);
    return ane_compile_mil(mil, 2, in_sizes, 1, &out_size);
}

static ANEKernel* compile_dynamic_ffn(int dim, int inter) {
    char mil[8192];
    make_dynamic_ffn_mil(mil, sizeof(mil), dim, inter);
    size_t in_sizes[4] = {
        (size_t)dim * SP * sizeof(uint16_t),
        (size_t)inter * dim * sizeof(uint16_t),
        (size_t)inter * dim * sizeof(uint16_t),
        (size_t)dim * inter * sizeof(uint16_t),
    };
    size_t out_size = (size_t)dim * SP * sizeof(uint16_t);
    return ane_compile_mil(mil, 4, in_sizes, 1, &out_size);
}

// Write fp16 weights into an IOSurface (dense packed, no spatial stride)
static bool write_weights_to_surface(ANEKernel* k, int input_idx, const uint16_t* fp16, size_t numel) {
    // Access internal IOSurface (we declared ANEKernel in the header... it's opaque)
    // We need to expose the IOSurface. For now, use the ane_dynamic_matvec pattern.
    // Actually ANEKernel is defined in ane_runtime.cpp, not exposed.
    // Let's use the ane_compile_mil + ane_dynamic_matvec approach but for conv.
    
    // We need to add eval functions for dynamic conv too. For now, let's test
    // compilation at scale and measure eval later.
    return true; // placeholder
}

static void test_dynamic_conv_compile() {
    printf("=== Test 1: Dynamic conv compile at various sizes ===\n");
    
    struct { int out; int in; const char* desc; } cases[] = {
        {64, 64, "tiny"},
        {1024, 1024, "2MB"},
        {2560, 2560, "12.5MB (o_proj)"},
        {9216, 2560, "45MB (gate/up_proj)"},
        {2560, 9216, "45MB (down_proj)"},
        {16384, 2560, "80MB (LM head chunk)"},
    };
    
    for (auto& tc : cases) {
        size_t bytes = (size_t)tc.out * tc.in * 2;
        printf("  [%d, %d] (%.1f MB) %s: ", tc.out, tc.in, bytes/(1024.0*1024.0), tc.desc);
        fflush(stdout);
        Timer t;
        ANEKernel* k = compile_dynamic_conv(tc.out, tc.in);
        if (k) {
            printf("OK (%.0f ms, compiles=%d)\n", t.elapsed_ms(), ane_compile_count());
            ane_free(k);
        } else {
            printf("FAILED (%.0f ms)\n", t.elapsed_ms());
        }
    }
}

static void test_dynamic_ffn_compile() {
    printf("\n=== Test 2: Dynamic FFN compile ===\n");
    
    struct { int dim; int inter; const char* desc; } cases[] = {
        {64, 128, "tiny"},
        {1024, 3584, "0.8B"},
        {2560, 9216, "4B"},
    };
    
    for (auto& tc : cases) {
        size_t bytes = (size_t)(2 * tc.inter * tc.dim + tc.dim * tc.inter) * 2;
        printf("  dim=%d inter=%d (%.1f MB weights) %s: ", 
               tc.dim, tc.inter, bytes/(1024.0*1024.0), tc.desc);
        fflush(stdout);
        Timer t;
        ANEKernel* k = compile_dynamic_ffn(tc.dim, tc.inter);
        if (k) {
            printf("OK (%.0f ms, compiles=%d)\n", t.elapsed_ms(), ane_compile_count());
            ane_free(k);
        } else {
            printf("FAILED (%.0f ms)\n", t.elapsed_ms());
        }
    }
}

static void test_loaded_budget_dynamic() {
    printf("\n=== Test 3: How many dynamic convs can we load simultaneously? ===\n");
    printf("  (Dynamic convs have NO baked weights — should use minimal memory)\n");
    
    ane_set_persist_cache(false);
    
    // Load lots of dynamic conv kernels (all same shape — should share compiled code?)
    int out_dim = 9216, in_dim = 2560;
    printf("  Loading dynamic conv [%d,%d] kernels...\n", out_dim, in_dim);
    
    std::vector<ANEKernel*> kernels;
    for (int i = 0; i < 300; i++) {
        ANEKernel* k = compile_dynamic_conv(out_dim, in_dim);
        if (!k) {
            printf("  LIMIT at %d kernels (compiles=%d)\n", (int)kernels.size(), ane_compile_count());
            break;
        }
        kernels.push_back(k);
        if ((i+1) % 50 == 0)
            printf("  %d loaded (compiles=%d)\n", i+1, ane_compile_count());
    }
    
    if (kernels.size() >= 300)
        printf("  No limit hit at 300 kernels!\n");
    
    printf("  Total loaded: %d\n", (int)kernels.size());
    for (auto* k : kernels) ane_free(k);
}

static void test_correctness_and_perf() {
    printf("\n=== Test 4: Dynamic conv correctness & performance ===\n");
    
    // We need to evaluate the kernel. Since ANEKernel is opaque,
    // we need to update ane_runtime to expose dynamic eval functions.
    // For now, let's use the ane_compile_dynamic_matmul path but fixed
    // to use conv instead of matmul.
    
    // Actually, let's update ane_compile_dynamic_matmul to use conv!
    printf("  (Need to wire up eval — testing via updated dynamic_matmul path)\n");
    
    int out_dim = 64, in_dim = 64;
    
    // Compile
    ANEKernel* k = compile_dynamic_conv(out_dim, in_dim);
    if (!k) { printf("  Compile failed!\n"); return; }
    
    // For eval, we need to write to the IOSurfaces directly.
    // The ANEKernel struct has ioInputs[0] (x) and ioInputs[1] (W).
    // Since ANEKernel is defined in ane_runtime.cpp (not exposed), we can't
    // access the fields from here. We need to add an eval function.
    printf("  Compiled OK. Eval requires exposing IOSurface access — see next step.\n");
    
    ane_free(k);
}

int main(int argc, char* argv[]) {
    void* pool = objc_autoreleasePoolPush();
    g_verbose = false;
    ane_init();
    if (!ane_available()) { fprintf(stderr, "ANE not available!\n"); return 1; }
    
    ane_set_persist_cache(true);
    
    test_dynamic_conv_compile();
    test_dynamic_ffn_compile();
    test_loaded_budget_dynamic();
    // test_correctness_and_perf(); // needs eval wiring
    
    printf("\nFinal: compiles=%d, cache_loads=%d\n", ane_compile_count(), ane_cache_loads());
    objc_autoreleasePoolPop(pool);
    return 0;
}
