// test_dynamic14.cpp — Why does mul fail in test13 but pass in test12?
// Hypothesis: ane_eval_multi vs ane_write_surface_raw + ane_eval_raw_outputs
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

int main() {
    void* pool = objc_autoreleasePoolPush();
    g_verbose = false;
    ane_init();
    if (!ane_available()) { fprintf(stderr, "ANE not available!\n"); return 1; }
    ane_set_persist_cache(false);
    
    printf("=== Write method comparison ===\n\n");
    
    // All tests: mul [1,2,1,SP]*[4,2,1,SP] -> [4,2,1,SP]
    const char* mil_fmt =
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, 2, 1, %d]> x, tensor<fp16, [4, 2, 1, %d]> W) {\n"
        "        tensor<fp16, [4, 2, 1, %d]> y = mul(x = W, y = x)[name = tensor<string, []>(\"m\")];\n"
        "    } -> (y);\n"
        "}\n";
    char mil[2048];
    snprintf(mil, sizeof(mil), mil_fmt, SP, SP, SP);
    
    size_t xsz = 1*2*SP*2;  // 128 bytes
    size_t wsz = 4*2*SP*2;  // 512 bytes  
    size_t osz = 4*2*SP*2;  // 512 bytes
    size_t in_sizes[2] = {xsz, wsz};
    
    // Test A: write_surface_raw with MINIMAL data (like test12)
    {
        printf("A: write_surface_raw minimal (2 bytes each)... ");
        ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &osz);
        if (!k) { printf("COMPILE FAIL\n"); }
        else {
            uint16_t v2 = f32_to_f16(2.0f), v3 = f32_to_f16(3.0f);
            ane_write_surface_raw(k, 0, &v2, 2);
            ane_write_surface_raw(k, 1, &v3, 2);
            float y[8]={}; float*op[]={y}; int oc[]={1};
            bool ok = ane_eval_raw_outputs(k, op, oc);
            printf("%s y[0]=%.1f\n", ok?"PASS":"FAIL", y[0]);
            ane_free(k);
        }
    }
    
    // Test B: eval_multi (like test13)
    {
        printf("B: eval_multi (2ch + 8ch SP-strided)... ");
        ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &osz);
        if (!k) { printf("COMPILE FAIL\n"); }
        else {
            float x[2]={3,5}, w[8]={1,2,2,4,3,6,4,8};
            float*ip[]={x,w}; int ic[]={2,8};
            float out[8]; float*op[]={out}; int oc[]={8};
            bool ok = ane_eval_multi(k, ip, ic, op, oc);
            printf("%s out=[%.1f,%.1f,%.1f,%.1f]\n", ok?"PASS":"FAIL", out[0],out[1],out[2],out[3]);
            ane_free(k);
        }
    }
    
    // Test C: write_surface_raw with FULL data (properly zero-padded at SP stride)
    {
        printf("C: write_surface_raw full (SP-strided manual)... ");
        ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &osz);
        if (!k) { printf("COMPILE FAIL\n"); }
        else {
            // x: [1,2,1,SP] = 2 channels, SP values each. Only pos 0 has data.
            uint16_t* xbuf = (uint16_t*)calloc(xsz/2, sizeof(uint16_t));
            xbuf[0] = f32_to_f16(3.0f);       // ch0, pos0
            xbuf[SP] = f32_to_f16(5.0f);      // ch1, pos0
            ane_write_surface_raw(k, 0, xbuf, xsz);
            
            // W: [4,2,1,SP] = 4*2=8 channels. 
            uint16_t* wbuf = (uint16_t*)calloc(wsz/2, sizeof(uint16_t));
            float wvals[8] = {1,2,2,4,3,6,4,8};
            for (int i = 0; i < 8; i++) wbuf[(size_t)i*SP] = f32_to_f16(wvals[i]);
            ane_write_surface_raw(k, 1, wbuf, wsz);
            
            float out[8]={}; float*op[]={out}; int oc[]={8};
            bool ok = ane_eval_raw_outputs(k, op, oc);
            printf("%s out=[%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f]\n",
                   ok?"PASS":"FAIL", out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
            // Expected: W[n,c]*x[0,c] at pos 0
            // out[0]=1*3=3, out[1]=2*5=10, out[2]=2*3=6, out[3]=4*5=20, etc.
            if (ok) {
                float exp[8] = {1*3.f, 2*5.f, 2*3.f, 4*5.f, 3*3.f, 6*5.f, 4*3.f, 8*5.f};
                printf("  Expected: [%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f]\n",
                       exp[0],exp[1],exp[2],exp[3],exp[4],exp[5],exp[6],exp[7]);
            }
            free(xbuf); free(wbuf);
            ane_free(k);
        }
    }
    
    // Test D: Same but tile+mul (the failing combo)
    {
        printf("\nD: tile+mul with write_surface_raw... ");
        char mil2[4096];
        snprintf(mil2, sizeof(mil2),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, 2, 1, %d]> x, tensor<fp16, [4, 2, 1, %d]> W) {\n"
            "        tensor<int32, [4]> reps = const()[name = tensor<string, []>(\"rp\"), val = tensor<int32, [4]>([4, 1, 1, 1])];\n"
            "        tensor<fp16, [4, 2, 1, %d]> xt = tile(x = x, reps = reps)[name = tensor<string, []>(\"t\")];\n"
            "        tensor<fp16, [4, 2, 1, %d]> y = mul(x = W, y = xt)[name = tensor<string, []>(\"m\")];\n"
            "    } -> (y);\n"
            "}\n", SP, SP, SP, SP);
        
        ANEKernel* k = ane_compile_mil(mil2, 2, in_sizes, 1, &osz);
        if (!k) { printf("COMPILE FAIL\n"); }
        else {
            uint16_t* xbuf = (uint16_t*)calloc(xsz/2, sizeof(uint16_t));
            xbuf[0] = f32_to_f16(3.0f);
            xbuf[SP] = f32_to_f16(5.0f);
            ane_write_surface_raw(k, 0, xbuf, xsz);
            
            uint16_t* wbuf = (uint16_t*)calloc(wsz/2, sizeof(uint16_t));
            float wvals[8] = {1,2,2,4,3,6,4,8};
            for (int i = 0; i < 8; i++) wbuf[(size_t)i*SP] = f32_to_f16(wvals[i]);
            ane_write_surface_raw(k, 1, wbuf, wsz);
            
            float out[8]={}; float*op[]={out}; int oc[]={8};
            bool ok = ane_eval_raw_outputs(k, op, oc);
            printf("%s\n", ok?"PASS":"FAIL");
            if (ok) {
                printf("  out=[%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f]\n",
                       out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
            }
            free(xbuf); free(wbuf);
            ane_free(k);
        }
    }
    
    // Test E: mul with N-broadcast, full data write
    {
        printf("\nE: N-bcast mul [1,2,1,SP]*[4,2,1,SP] full write... ");
        ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &osz);
        if (!k) { printf("COMPILE FAIL\n"); }
        else {
            uint16_t* xbuf = (uint16_t*)calloc(xsz/2, sizeof(uint16_t));
            xbuf[0] = f32_to_f16(3.0f);
            xbuf[SP] = f32_to_f16(5.0f);
            ane_write_surface_raw(k, 0, xbuf, xsz);
            
            uint16_t* wbuf = (uint16_t*)calloc(wsz/2, sizeof(uint16_t));
            for (int i = 0; i < 8; i++) wbuf[(size_t)i*SP] = f32_to_f16((float)(i+1));
            ane_write_surface_raw(k, 1, wbuf, wsz);
            
            float out[8]={}; float*op[]={out}; int oc[]={8};
            bool ok = ane_eval_raw_outputs(k, op, oc);
            printf("%s\n", ok?"PASS":"FAIL");
            if (ok) printf("  out=[%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f]\n",
                           out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]);
            free(xbuf); free(wbuf);
            ane_free(k);
        }
    }
    
    printf("\n  compiles=%d\n", ane_compile_count());
    objc_autoreleasePoolPop(pool);
    return 0;
}
