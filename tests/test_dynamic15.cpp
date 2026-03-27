// test_dynamic15.cpp — Is C=2 the problem? Minimal test
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "core/ane_runtime.h"
#include <ane_lm/common.h>
extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void  objc_autoreleasePoolPop(void*);
using namespace ane_lm;
#define SP ANE_SPATIAL

static bool test(const char* desc, int n1,int c1,int n2,int c2, const char* op) {
    printf("  %-55s ", desc); fflush(stdout);
    int on=n1>n2?n1:n2, oc=c1>c2?c1:c2;
    char mil[2048];
    snprintf(mil, sizeof(mil),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [%d,%d,1,%d]> a, tensor<fp16, [%d,%d,1,%d]> b) {\n"
        "        tensor<fp16, [%d,%d,1,%d]> y = %s(x = a, y = b)[name = tensor<string, []>(\"op\")];\n"
        "    } -> (y);\n"
        "}\n", n1,c1,SP, n2,c2,SP, on,oc,SP, op);
    size_t s1=(size_t)n1*c1*SP*2, s2=(size_t)n2*c2*SP*2;
    size_t so=(size_t)on*oc*SP*2;
    size_t in_sizes[2]={s1,s2};
    ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &so);
    if (!k) { printf("COMPILE FAIL\n"); return false; }
    uint16_t v2=f32_to_f16(2.0f), v3=f32_to_f16(3.0f);
    ane_write_surface_raw(k, 0, &v2, 2);
    ane_write_surface_raw(k, 1, &v3, 2);
    float y[4]={}; float*op_arr[]={y}; int och[]={1};
    bool ok = ane_eval_raw_outputs(k, op_arr, och);
    if (ok) printf("PASS y[0]=%.1f\n", y[0]);
    else printf("FAIL\n");
    ane_free(k); return ok;
}

int main() {
    void* pool = objc_autoreleasePoolPush();
    g_verbose = false;
    ane_init();
    if (!ane_available()) { fprintf(stderr, "ANE not available!\n"); return 1; }
    ane_set_persist_cache(false);
    
    printf("=== C dimension investigation ===\n\n");
    
    // Vary C while keeping N broadcast
    printf("--- mul with N broadcast, varying C ---\n");
    test("[1,1,1,SP]*[4,1,1,SP] mul C=1", 1,1, 4,1, "mul");
    test("[1,2,1,SP]*[4,2,1,SP] mul C=2", 1,2, 4,2, "mul");
    test("[1,3,1,SP]*[4,3,1,SP] mul C=3", 1,3, 4,3, "mul");
    test("[1,4,1,SP]*[4,4,1,SP] mul C=4", 1,4, 4,4, "mul");
    test("[1,8,1,SP]*[4,8,1,SP] mul C=8", 1,8, 4,8, "mul");
    test("[1,16,1,SP]*[4,16,1,SP] mul C=16", 1,16, 4,16, "mul");
    test("[1,32,1,SP]*[4,32,1,SP] mul C=32", 1,32, 4,32, "mul");
    
    printf("\n--- mul same shape, varying C ---\n");
    test("[4,1,1,SP]*[4,1,1,SP] mul C=1", 4,1, 4,1, "mul");
    test("[4,2,1,SP]*[4,2,1,SP] mul C=2", 4,2, 4,2, "mul");
    test("[4,3,1,SP]*[4,3,1,SP] mul C=3", 4,3, 4,3, "mul");
    
    printf("\n--- mul N=1, varying C ---\n");
    test("[1,1,1,SP]*[1,1,1,SP] mul", 1,1, 1,1, "mul");
    test("[1,2,1,SP]*[1,2,1,SP] mul", 1,2, 1,2, "mul");
    test("[1,3,1,SP]*[1,3,1,SP] mul", 1,3, 1,3, "mul");
    
    printf("\n--- Vary N with C=2 ---\n");
    test("[1,2,1,SP]*[1,2,1,SP] N=1", 1,2, 1,2, "mul");
    test("[1,2,1,SP]*[2,2,1,SP] N=1v2", 1,2, 2,2, "mul");
    test("[2,2,1,SP]*[2,2,1,SP] N=2", 2,2, 2,2, "mul");
    test("[1,2,1,SP]*[3,2,1,SP] N=1v3", 1,2, 3,2, "mul");
    test("[1,2,1,SP]*[4,2,1,SP] N=1v4", 1,2, 4,2, "mul");
    test("[1,2,1,SP]*[8,2,1,SP] N=1v8", 1,2, 8,2, "mul");
    
    printf("\n  compiles=%d\n", ane_compile_count());
    objc_autoreleasePoolPop(pool);
    return 0;
}
