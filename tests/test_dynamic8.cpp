// test_dynamic8.cpp — Systematic shape/broadcast investigation
// Finding: [1,C,1,SP]+[1,C,1,SP] works, broadcast on N works, but what about C broadcast?
// And what's the minimum surface size?
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

static bool test_case(const char* desc,
                      int n1, int c1, int h1, int w1,
                      int n2, int c2, int h2, int w2,
                      const char* op) {
    printf("  %-55s ", desc);
    fflush(stdout);
    
    // Output shape from broadcast
    int on = n1 > n2 ? n1 : n2;
    int oc = c1 > c2 ? c1 : c2;
    int oh = h1 > h2 ? h1 : h2;
    int ow = w1 > w2 ? w1 : w2;
    
    char mil[2048];
    snprintf(mil, sizeof(mil),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [%d, %d, %d, %d]> a, tensor<fp16, [%d, %d, %d, %d]> b) {\n"
        "        tensor<fp16, [%d, %d, %d, %d]> y = %s(x = a, y = b)[name = tensor<string, []>(\"op\")];\n"
        "    } -> (y);\n"
        "}\n",
        n1, c1, h1, w1, n2, c2, h2, w2, on, oc, oh, ow, op);
    
    // IOSurface sizes: N*C*H*W * sizeof(fp16)
    size_t s1 = (size_t)n1 * c1 * h1 * w1 * sizeof(uint16_t);
    size_t s2 = (size_t)n2 * c2 * h2 * w2 * sizeof(uint16_t);
    size_t so = (size_t)on * oc * oh * ow * sizeof(uint16_t);
    
    size_t in_sizes[2] = {s1, s2};
    
    ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &so);
    if (!k) { printf("COMPILE FAIL\n"); return false; }
    
    // Write 1.0 to first position of each surface
    uint16_t one = f32_to_f16(1.0f);
    uint16_t two = f32_to_f16(2.0f);
    ane_write_surface_raw(k, 0, &one, sizeof(uint16_t));
    ane_write_surface_raw(k, 1, &two, sizeof(uint16_t));
    
    float y[64] = {};
    float* outptrs[] = {y};
    int out_chs[] = {1}; // just read first channel
    bool ok = ane_eval_raw_outputs(k, outptrs, out_chs);
    
    if (ok) printf("EVAL OK  y[0]=%.1f\n", y[0]);
    else    printf("EVAL FAIL\n");
    
    ane_free(k);
    return ok;
}

int main() {
    void* pool = objc_autoreleasePoolPush();
    g_verbose = false;
    ane_init();
    if (!ane_available()) { fprintf(stderr, "ANE not available!\n"); return 1; }
    ane_set_persist_cache(false);
    
    printf("=== Systematic shape investigation for multi-input ANE eval ===\n\n");
    
    // Group 1: Both same shape, W=SP
    printf("--- Both same shape, W=SP ---\n");
    test_case("[1,8,1,SP]+[1,8,1,SP] add",  1,8,1,SP, 1,8,1,SP, "add");
    test_case("[1,1,1,SP]+[1,1,1,SP] add",  1,1,1,SP, 1,1,1,SP, "add");
    test_case("[2,4,1,SP]+[2,4,1,SP] add",  2,4,1,SP, 2,4,1,SP, "add");
    test_case("[8,8,1,SP]+[8,8,1,SP] add",  8,8,1,SP, 8,8,1,SP, "add");
    
    // Group 2: Broadcast on N only
    printf("\n--- Broadcast on N, W=SP ---\n");
    test_case("[1,8,1,SP]+[4,8,1,SP] add (N:1->4)",  1,8,1,SP, 4,8,1,SP, "add");
    test_case("[8,8,1,SP]+[1,8,1,SP] add (N:1->8)",  8,8,1,SP, 1,8,1,SP, "add");
    
    // Group 3: Broadcast on C only
    printf("\n--- Broadcast on C, W=SP ---\n");
    test_case("[1,1,1,SP]+[1,8,1,SP] add (C:1->8)",  1,1,1,SP, 1,8,1,SP, "add");
    test_case("[1,8,1,SP]+[1,1,1,SP] add (C:1->8)",  1,8,1,SP, 1,1,1,SP, "add");
    test_case("[1,1,1,SP]+[1,8,1,SP] mul (C:1->8)",  1,1,1,SP, 1,8,1,SP, "mul");
    
    // Group 4: W != SP (should all fail based on previous findings)
    printf("\n--- W != SP (expected to fail) ---\n");
    test_case("[1,8,1,1]+[1,8,1,1] add (W=1)",  1,8,1,1, 1,8,1,1, "add");
    test_case("[1,8,1,16]+[1,8,1,16] add (W=16)",  1,8,1,16, 1,8,1,16, "add");
    test_case("[1,8,1,64]+[1,8,1,64] add (W=64)",  1,8,1,64, 1,8,1,64, "add");
    
    // Group 5: H != 1 with W=SP
    printf("\n--- H != 1, W=SP ---\n");
    test_case("[1,8,2,SP]+[1,8,2,SP] add (H=2)",  1,8,2,SP, 1,8,2,SP, "add");
    test_case("[1,4,4,SP]+[1,4,4,SP] add (H=4)",  1,4,4,SP, 1,4,4,SP, "add");
    
    // Group 6: Large C with W=SP
    printf("\n--- Large C, W=SP ---\n");
    test_case("[1,64,1,SP]+[1,64,1,SP] add (C=64)",  1,64,1,SP, 1,64,1,SP, "add");
    test_case("[1,256,1,SP]+[1,256,1,SP] add (C=256)",  1,256,1,SP, 1,256,1,SP, "add");
    test_case("[1,1024,1,SP]+[1,1024,1,SP] add (C=1k)",  1,1024,1,SP, 1,1024,1,SP, "add");
    
    printf("\ncompiles=%d\n", ane_compile_count());
    objc_autoreleasePoolPop(pool);
    return 0;
}
