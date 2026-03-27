#include <cstdio>
#include "core/ane_runtime.h"
#include <ane_lm/common.h>
extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void  objc_autoreleasePoolPop(void*);
using namespace ane_lm;
#define SP ANE_SPATIAL

int main() {
    void* pool = objc_autoreleasePoolPush();
    g_verbose = false;
    ane_init();
    ane_set_persist_cache(false);

    // Exact same as test12: mul [1,2,1,SP]*[4,2,1,SP]
    char mil[2048];
    snprintf(mil, sizeof(mil),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, 2, 1, %d]> a, tensor<fp16, [4, 2, 1, %d]> b) {\n"
        "        tensor<fp16, [4, 2, 1, %d]> y = mul(x = a, y = b)[name = tensor<string, []>(\"op\")];\n"
        "    } -> (y);\n"
        "}\n", SP, SP, SP);
    
    size_t s1 = 1*2*1*SP*2;  // 128
    size_t s2 = 4*2*1*SP*2;  // 512
    size_t so = 4*2*1*SP*2;  // 512
    size_t in_sizes[2] = {s1, s2};
    
    printf("Compiling mul [1,2,1,%d]*[4,2,1,%d] -> [4,2,1,%d]...\n", SP, SP, SP);
    printf("  in_sizes: %zu, %zu  out_size: %zu\n", s1, s2, so);
    
    ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &so);
    if (!k) { printf("COMPILE FAIL\n"); return 1; }
    printf("Compiled OK\n");
    
    uint16_t v2 = f32_to_f16(2.0f), v3 = f32_to_f16(3.0f);
    ane_write_surface_raw(k, 0, &v2, 2);
    ane_write_surface_raw(k, 1, &v3, 2);
    
    float y[8] = {};
    float* op[] = {y};
    int oc[] = {1};
    bool ok = ane_eval_raw_outputs(k, op, oc);
    printf("Result: %s y[0]=%.1f (expected 6.0)\n", ok?"PASS":"FAIL", y[0]);
    
    ane_free(k);
    objc_autoreleasePoolPop(pool);
    return ok ? 0 : 1;
}
