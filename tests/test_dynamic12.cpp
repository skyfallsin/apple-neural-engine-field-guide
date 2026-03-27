// test_dynamic12.cpp — Minimal mul investigation
// Finding which exact configurations of mul work vs fail
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

static bool test(const char* desc, int n1,int c1,int h1,int w1,
                 int n2,int c2,int h2,int w2, const char* op) {
    printf("  %-55s ", desc); fflush(stdout);
    int on=n1>n2?n1:n2, oc=c1>c2?c1:c2, oh=h1>h2?h1:h2, ow=w1>w2?w1:w2;
    char mil[2048];
    snprintf(mil, sizeof(mil),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [%d,%d,%d,%d]> a, tensor<fp16, [%d,%d,%d,%d]> b) {\n"
        "        tensor<fp16, [%d,%d,%d,%d]> y = %s(x = a, y = b)[name = tensor<string, []>(\"op\")];\n"
        "    } -> (y);\n"
        "}\n", n1,c1,h1,w1, n2,c2,h2,w2, on,oc,oh,ow, op);
    size_t s1=(size_t)n1*c1*h1*w1*2, s2=(size_t)n2*c2*h2*w2*2;
    size_t so=(size_t)on*oc*oh*ow*2;
    size_t in_sizes[2]={s1,s2};
    ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &so);
    if (!k) { printf("COMPILE FAIL\n"); return false; }
    // Write 2.0 and 3.0 at position 0
    uint16_t v2=f32_to_f16(2.0f), v3=f32_to_f16(3.0f);
    ane_write_surface_raw(k, 0, &v2, 2);
    ane_write_surface_raw(k, 1, &v3, 2);
    float y[64]={};
    float* op_arr[]={y}; int och[]={1};
    bool ok = ane_eval_raw_outputs(k, op_arr, och);
    if (ok) printf("EVAL OK  y[0]=%.1f\n", y[0]);
    else    printf("EVAL FAIL\n");
    ane_free(k); return ok;
}

int main() {
    void* pool = objc_autoreleasePoolPush();
    g_verbose = false;
    ane_init();
    if (!ane_available()) { fprintf(stderr, "ANE not available!\n"); return 1; }
    ane_set_persist_cache(false);
    
    printf("=== Minimal mul vs add investigation ===\n\n");
    
    printf("--- add: same shape ---\n");
    test("[1,1,1,SP]+[1,1,1,SP] add", 1,1,1,SP, 1,1,1,SP, "add");
    test("[1,4,1,SP]+[1,4,1,SP] add", 1,4,1,SP, 1,4,1,SP, "add");
    test("[4,4,1,SP]+[4,4,1,SP] add", 4,4,1,SP, 4,4,1,SP, "add");
    
    printf("\n--- mul: same shape ---\n");
    test("[1,1,1,SP]*[1,1,1,SP] mul", 1,1,1,SP, 1,1,1,SP, "mul");
    test("[1,4,1,SP]*[1,4,1,SP] mul", 1,4,1,SP, 1,4,1,SP, "mul");
    test("[4,4,1,SP]*[4,4,1,SP] mul", 4,4,1,SP, 4,4,1,SP, "mul");
    test("[2,2,1,SP]*[2,2,1,SP] mul", 2,2,1,SP, 2,2,1,SP, "mul");
    
    printf("\n--- mul: C broadcast (N=1) ---\n");
    test("[1,1,1,SP]*[1,4,1,SP] mul Cbcast", 1,1,1,SP, 1,4,1,SP, "mul");
    test("[1,4,1,SP]*[1,1,1,SP] mul Cbcast", 1,4,1,SP, 1,1,1,SP, "mul");
    
    printf("\n--- mul: N broadcast ---\n");
    test("[1,4,1,SP]*[4,4,1,SP] mul Nbcast", 1,4,1,SP, 4,4,1,SP, "mul");
    test("[4,4,1,SP]*[1,4,1,SP] mul Nbcast", 4,4,1,SP, 1,4,1,SP, "mul");
    test("[1,1,1,SP]*[4,1,1,SP] mul Nbcast", 1,1,1,SP, 4,1,1,SP, "mul");
    
    printf("\n--- add: N broadcast (for comparison) ---\n");
    test("[1,4,1,SP]+[4,4,1,SP] add Nbcast", 1,4,1,SP, 4,4,1,SP, "add");
    
    printf("\n--- mul: H>1 ---\n");
    test("[1,1,2,SP]*[1,1,2,SP] mul H=2", 1,1,2,SP, 1,1,2,SP, "mul");
    test("[1,4,2,SP]*[1,4,2,SP] mul H=2", 1,4,2,SP, 1,4,2,SP, "mul");
    test("[1,1,2,SP]*[1,4,2,SP] mul H=2 Cbcast", 1,1,2,SP, 1,4,2,SP, "mul");
    
    printf("\n--- single input mul (x*x) ---\n");
    {
        printf("  %-55s ", "[4,4,1,SP] x*x (single input)"); fflush(stdout);
        char mil[2048];
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [4, 4, 1, %d]> x) {\n"
            "        tensor<fp16, [4, 4, 1, %d]> y = mul(x = x, y = x)[name = tensor<string, []>(\"m\")];\n"
            "    } -> (y);\n"
            "}\n", SP, SP);
        size_t sz = 4*4*SP*2;
        ANEKernel* k = ane_compile_mil(mil, 1, &sz, 1, &sz);
        if (!k) { printf("COMPILE FAIL\n"); }
        else {
            float vals[16]; for(int i=0;i<16;i++) vals[i]=2.0f;
            float* ip[]={vals}; int ic[]={16};
            float out[16]; float* op[]={out}; int oc[]={16};
            bool ok = ane_eval_multi(k, ip, ic, op, oc);
            if (ok) printf("EVAL OK y[0]=%.1f\n", out[0]);
            else printf("EVAL FAIL\n");
            ane_free(k);
        }
    }
    
    printf("\n  compiles=%d\n", ane_compile_count());
    objc_autoreleasePoolPop(pool);
    return 0;
}
