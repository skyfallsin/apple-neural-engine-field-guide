// test_dynamic2.cpp — Test various MIL formulations to find what ANE accepts
// for multi-input / dynamic-weight kernels
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "core/ane_runtime.h"
#include <ane_lm/common.h>

extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void  objc_autoreleasePoolPop(void*);

using namespace ane_lm;

#define SP ANE_SPATIAL

static bool try_compile(const char* label, const char* mil, 
                         int n_in, size_t* in_sizes, int n_out, size_t* out_sizes) {
    printf("  %-50s ", label);
    fflush(stdout);
    ANEKernel* k = ane_compile_mil(mil, n_in, in_sizes, n_out, out_sizes);
    if (k) {
        printf("OK (compiles=%d)\n", ane_compile_count());
        ane_free(k);
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

int main() {
    void* pool = objc_autoreleasePoolPush();
    g_verbose = false;
    ane_init();
    if (!ane_available()) { fprintf(stderr, "ANE not available!\n"); return 1; }
    ane_set_persist_cache(false);
    
    printf("=== Testing MIL formulations on ANE ===\n\n");
    
    char mil[8192];
    size_t in_sizes[8], out_sizes[4];
    
    // Test 1: Two 4D inputs, simple add (does multi-input work at all?)
    {
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, 64, 1, %d]> a, tensor<fp16, [1, 64, 1, %d]> b) {\n"
            "        tensor<fp16, [1, 64, 1, %d]> y = add(x = a, y = b)[name = tensor<string, []>(\"add\")];\n"
            "    } -> (y);\n"
            "}\n", SP, SP, SP);
        in_sizes[0] = in_sizes[1] = 64 * SP * 2;
        out_sizes[0] = 64 * SP * 2;
        try_compile("Two 4D inputs + add", mil, 2, in_sizes, 1, out_sizes);
    }
    
    // Test 2: Two 4D inputs, elementwise mul
    {
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, 64, 1, %d]> a, tensor<fp16, [1, 64, 1, %d]> b) {\n"
            "        tensor<fp16, [1, 64, 1, %d]> y = mul(x = a, y = b)[name = tensor<string, []>(\"mul\")];\n"
            "    } -> (y);\n"
            "}\n", SP, SP, SP);
        in_sizes[0] = in_sizes[1] = 64 * SP * 2;
        out_sizes[0] = 64 * SP * 2;
        try_compile("Two 4D inputs + mul", mil, 2, in_sizes, 1, out_sizes);
    }
    
    // Test 3: matmul with 4D tensors (no reshape)
    // matmul supports 4D: [B, C, M, K] @ [B, C, K, N] = [B, C, M, N]
    {
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, 1, 1, 64]> x, tensor<fp16, [1, 1, 64, 64]> W) {\n"
            "        tensor<fp16, [1, 1, 1, 64]> y = matmul(x = x, y = W)[name = tensor<string, []>(\"mm\")];\n"
            "    } -> (y);\n"
            "}\n");
        in_sizes[0] = 1 * 1 * 1 * 64 * 2;
        in_sizes[1] = 1 * 1 * 64 * 64 * 2;
        out_sizes[0] = 1 * 1 * 1 * 64 * 2;
        try_compile("matmul 4D [1,1,1,64]@[1,1,64,64]", mil, 2, in_sizes, 1, out_sizes);
    }
    
    // Test 4: matmul with 2D tensors only (no reshape from 4D)
    {
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, 64]> x, tensor<fp16, [64, 64]> W) {\n"
            "        tensor<fp16, [1, 64]> y = matmul(x = x, y = W)[name = tensor<string, []>(\"mm\")];\n"
            "    } -> (y);\n"
            "}\n");
        in_sizes[0] = 1 * 64 * 2;
        in_sizes[1] = 64 * 64 * 2;
        out_sizes[0] = 1 * 64 * 2;
        try_compile("matmul 2D [1,64]@[64,64]", mil, 2, in_sizes, 1, out_sizes);
    }
    
    // Test 5: linear op (high-level matmul)
    {
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, 64]> x, tensor<fp16, [64, 64]> W) {\n"
            "        tensor<fp16, [1, 64]> y = linear(x = x, weight = W)[name = tensor<string, []>(\"lin\")];\n"
            "    } -> (y);\n"
            "}\n");
        in_sizes[0] = 1 * 64 * 2;
        in_sizes[1] = 64 * 64 * 2;
        out_sizes[0] = 1 * 64 * 2;
        try_compile("linear 2D [1,64] weight=[64,64]", mil, 2, in_sizes, 1, out_sizes);
    }
    
    // Test 6: conv with weight as SECOND input (not const)
    // In MIL, conv weight param MUST be const. But let's try.
    {
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, 64, 1, %d]> x, tensor<fp16, [64, 64, 1, 1]> W) {\n"
            "        tensor<string, []> pt = const()[name = tensor<string, []>(\"pt\"), val = tensor<string, []>(\"valid\")];\n"
            "        tensor<int32, [2]> st = const()[name = tensor<string, []>(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
            "        tensor<int32, [4]> pd = const()[name = tensor<string, []>(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
            "        tensor<int32, [2]> dl = const()[name = tensor<string, []>(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
            "        tensor<int32, []> gr = const()[name = tensor<string, []>(\"gr\"), val = tensor<int32, []>(1)];\n"
            "        tensor<fp16, [1, 64, 1, %d]> y = conv(dilations = dl, groups = gr, "
            "pad = pd, pad_type = pt, strides = st, weight = W, x = x)"
            "[name = tensor<string, []>(\"cv\")];\n"
            "    } -> (y);\n"
            "}\n", SP, SP);
        in_sizes[0] = 64 * SP * 2;
        in_sizes[1] = 64 * 64 * 2;
        out_sizes[0] = 64 * SP * 2;
        try_compile("conv with input weight (not const)", mil, 2, in_sizes, 1, out_sizes);
    }
    
    // Test 7: reshape of 4D input to 2D (does reshape on inputs work?)
    {
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, 64, 1, %d]> x) {\n"
            "        tensor<int32, [2]> s = const()[name = tensor<string, []>(\"s\"), val = tensor<int32, [2]>([64, %d])];\n"
            "        tensor<fp16, [64, %d]> y = reshape(x = x, shape = s)[name = tensor<string, []>(\"rs\")];\n"
            "    } -> (y);\n"
            "}\n", SP, SP, SP);
        in_sizes[0] = 64 * SP * 2;
        out_sizes[0] = 64 * SP * 2;
        try_compile("reshape [1,64,1,SP] -> [64,SP]", mil, 1, in_sizes, 1, out_sizes);
    }
    
    // Test 8: einsum for matvec
    {
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, 64]> x, tensor<fp16, [64, 64]> W) {\n"
            "        tensor<string, []> eq = const()[name = tensor<string, []>(\"eq\"), val = tensor<string, []>(\"ij,jk->ik\")];\n"
            "        tensor<fp16, [1, 64]> y = einsum(values = (x, W), equation = eq)[name = tensor<string, []>(\"es\")];\n"
            "    } -> (y);\n"
            "}\n");
        in_sizes[0] = 1 * 64 * 2;
        in_sizes[1] = 64 * 64 * 2;
        out_sizes[0] = 1 * 64 * 2;
        try_compile("einsum ij,jk->ik 2D", mil, 2, in_sizes, 1, out_sizes);
    }
    
    // Test 9: Can we even have a 2nd input that's [C, C, 1, 1] (weight-shaped)?
    {
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, 64, 1, %d]> x, tensor<fp16, [64, 64, 1, 1]> W) {\n"
            "        tensor<fp16, [1, 64, 1, %d]> y = add(x = x, y = x)[name = tensor<string, []>(\"add\")];\n"
            "    } -> (y);\n"
            "}\n", SP, SP);
        in_sizes[0] = 64 * SP * 2;
        in_sizes[1] = 64 * 64 * 2;
        out_sizes[0] = 64 * SP * 2;
        try_compile("2 inputs (different shapes), ignore W", mil, 2, in_sizes, 1, out_sizes);
    }
    
    // Test 10: Three inputs (x + a + b)
    {
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, 64, 1, %d]> x, "
            "tensor<fp16, [1, 64, 1, %d]> a, "
            "tensor<fp16, [1, 64, 1, %d]> b) {\n"
            "        tensor<fp16, [1, 64, 1, %d]> t = add(x = x, y = a)[name = tensor<string, []>(\"a1\")];\n"
            "        tensor<fp16, [1, 64, 1, %d]> y = add(x = t, y = b)[name = tensor<string, []>(\"a2\")];\n"
            "    } -> (y);\n"
            "}\n", SP, SP, SP, SP, SP);
        in_sizes[0] = in_sizes[1] = in_sizes[2] = 64 * SP * 2;
        out_sizes[0] = 64 * SP * 2;
        try_compile("3 inputs (x+a+b), all same shape", mil, 3, in_sizes, 1, out_sizes);
    }
    
    // Test 11: mul + reduce_sum to do dot product (manual matvec)
    // W is [out, in, 1, 1], x is [1, in, 1, SP]
    // For each out channel: sum over in of W[o,i]*x[0,i,0,:]
    // This needs broadcasting: W[out,in,1,1] * x[1,in,1,SP] -> [out,in,1,SP]
    // Then reduce_sum over axis=1 -> [out,1,1,SP] -> reshape to [1,out,1,SP]
    {
        int D = 8; // small for test
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
            "}\n", D, SP, D, D, D, D, SP, D, SP, D, SP, D, SP);
        in_sizes[0] = D * SP * 2;
        in_sizes[1] = D * D * 2;
        out_sizes[0] = D * SP * 2;
        try_compile("mul+reduce_sum matvec (8x8)", mil, 2, in_sizes, 1, out_sizes);
    }
    
    printf("\nFinal: compiles=%d\n", ane_compile_count());
    objc_autoreleasePoolPop(pool);
    return 0;
}
