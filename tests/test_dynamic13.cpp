// test_dynamic13.cpp — Isolate which operation combination fails
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

static bool run_test(const char* desc, const char* mil_text,
                     int n_in, size_t* in_sizes, size_t out_size,
                     float** in_data, int* in_chs,
                     float* out_data, int out_ch) {
    printf("%-50s ", desc); fflush(stdout);
    ANEKernel* k = ane_compile_mil(mil_text, n_in, in_sizes, 1, &out_size);
    if (!k) { printf("COMPILE FAIL\n"); return false; }
    float* op[] = {out_data}; int oc[] = {out_ch};
    bool ok = ane_eval_multi(k, in_data, in_chs, op, oc);
    if (ok) {
        printf("EVAL OK  out[0..3]=[%.2f %.2f %.2f %.2f]\n",
               out_data[0], out_ch>1?out_data[1]:0, out_ch>2?out_data[2]:0, out_ch>3?out_data[3]:0);
    } else printf("EVAL FAIL\n");
    ane_free(k); return ok;
}

int main() {
    void* pool = objc_autoreleasePoolPush();
    g_verbose = false;
    ane_init();
    if (!ane_available()) { fprintf(stderr, "ANE not available!\n"); return 1; }
    ane_set_persist_cache(false);
    
    printf("=== Isolating failure: tile, mul+reduce, tile+mul, tile+mul+reduce ===\n\n");
    
    // Test 1: tile alone
    {
        char mil[2048];
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, 2, 1, %d]> x) {\n"
            "        tensor<int32, [4]> r = const()[name = tensor<string, []>(\"r\"), val = tensor<int32, [4]>([4, 1, 1, 1])];\n"
            "        tensor<fp16, [4, 2, 1, %d]> y = tile(x = x, reps = r)[name = tensor<string, []>(\"t\")];\n"
            "    } -> (y);\n"
            "}\n", SP, SP);
        size_t isz = 2*SP*2;
        float in_data[2] = {7.0f, 11.0f};
        float out[8]; float* ip[] = {in_data}; int ic[] = {2};
        run_test("tile [1,2,1,SP]->[4,2,1,SP]", mil, 1, &isz, 4*2*SP*2, ip, ic, out, 8);
    }
    
    // Test 2: mul(a,b) where both have N>1 (same shape)
    {
        char mil[2048];
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [4, 2, 1, %d]> a, tensor<fp16, [4, 2, 1, %d]> b) {\n"
            "        tensor<fp16, [4, 2, 1, %d]> y = mul(x = a, y = b)[name = tensor<string, []>(\"m\")];\n"
            "    } -> (y);\n"
            "}\n", SP, SP, SP);
        size_t sz = 4*2*SP*2;
        size_t in_sizes[2] = {sz, sz};
        float a[8]={1,2,3,4,5,6,7,8}, b[8]={2,2,2,2,2,2,2,2};
        float out[8]; float* ip[]={a,b}; int ic[]={8,8};
        run_test("mul [4,2,1,SP]*[4,2,1,SP] (same)", mil, 2, in_sizes, sz, ip, ic, out, 8);
    }
    
    // Test 3: mul + reduce_sum (both inputs same shape)
    {
        char mil[4096];
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [4, 2, 1, %d]> a, tensor<fp16, [4, 2, 1, %d]> b) {\n"
            "        tensor<fp16, [4, 2, 1, %d]> prod = mul(x = a, y = b)[name = tensor<string, []>(\"m\")];\n"
            "        tensor<int32, [1]> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, [1]>([1])];\n"
            "        tensor<bool, []> kd = const()[name = tensor<string, []>(\"kd\"), val = tensor<bool, []>(true)];\n"
            "        tensor<fp16, [4, 1, 1, %d]> y = reduce_sum(x = prod, axes = ax, keep_dims = kd)"
            "[name = tensor<string, []>(\"rs\")];\n"
            "    } -> (y);\n"
            "}\n", SP, SP, SP, SP);
        size_t sz = 4*2*SP*2;
        size_t in_sizes[2] = {sz, sz};
        size_t osz = 4*1*SP*2;
        float a[8]={1,2,3,4,5,6,7,8}, b[8]={2,2,2,2,2,2,2,2};
        float out[4]; float* ip[]={a,b}; int ic[]={8,8};
        run_test("mul+reduce [4,2,1,SP]*[4,2,1,SP]->reduce", mil, 2, in_sizes, osz, ip, ic, out, 4);
    }
    
    // Test 4: tile + mul (the exact combination from Strategy A)
    {
        char mil[4096];
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, 2, 1, %d]> x, tensor<fp16, [4, 2, 1, %d]> W) {\n"
            "        tensor<int32, [4]> reps = const()[name = tensor<string, []>(\"rp\"), val = tensor<int32, [4]>([4, 1, 1, 1])];\n"
            "        tensor<fp16, [4, 2, 1, %d]> xt = tile(x = x, reps = reps)[name = tensor<string, []>(\"t\")];\n"
            "        tensor<fp16, [4, 2, 1, %d]> y = mul(x = W, y = xt)[name = tensor<string, []>(\"m\")];\n"
            "    } -> (y);\n"
            "}\n", SP, SP, SP, SP);
        size_t xsz = 2*SP*2, wsz = 4*2*SP*2, osz = 4*2*SP*2;
        size_t in_sizes[2] = {xsz, wsz};
        float x[2]={3,5}, w[8]={1,2,2,4,3,6,4,8};
        float out[8]; float* ip[]={x,w}; int ic[]={2,8};
        run_test("tile+mul [1,2,1,SP] tile4 * [4,2,1,SP]", mil, 2, in_sizes, osz, ip, ic, out, 8);
    }
    
    // Test 5: tile + mul + reduce_sum (full pipeline)
    {
        char mil[4096];
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, 2, 1, %d]> x, tensor<fp16, [4, 2, 1, %d]> W) {\n"
            "        tensor<int32, [4]> reps = const()[name = tensor<string, []>(\"rp\"), val = tensor<int32, [4]>([4, 1, 1, 1])];\n"
            "        tensor<fp16, [4, 2, 1, %d]> xt = tile(x = x, reps = reps)[name = tensor<string, []>(\"t\")];\n"
            "        tensor<fp16, [4, 2, 1, %d]> prod = mul(x = W, y = xt)[name = tensor<string, []>(\"m\")];\n"
            "        tensor<int32, [1]> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, [1]>([1])];\n"
            "        tensor<bool, []> kd = const()[name = tensor<string, []>(\"kd\"), val = tensor<bool, []>(true)];\n"
            "        tensor<fp16, [4, 1, 1, %d]> y = reduce_sum(x = prod, axes = ax, keep_dims = kd)"
            "[name = tensor<string, []>(\"rs\")];\n"
            "    } -> (y);\n"
            "}\n", SP, SP, SP, SP, SP);
        size_t xsz = 2*SP*2, wsz = 4*2*SP*2, osz = 4*1*SP*2;
        size_t in_sizes[2] = {xsz, wsz};
        float x[2]={3,5}, w[8]={1,2,2,4,3,6,4,8};
        float out[4]; float* ip[]={x,w}; int ic[]={2,8};
        run_test("tile+mul+reduce (full pipeline)", mil, 2, in_sizes, osz, ip, ic, out, 4);
    }
    
    // Test 6: C-broadcast mul [1,1,2,SP]*[1,4,2,SP]
    {
        char mil[2048];
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, 1, 2, %d]> x, tensor<fp16, [1, 4, 2, %d]> W) {\n"
            "        tensor<fp16, [1, 4, 2, %d]> y = mul(x = W, y = x)[name = tensor<string, []>(\"m\")];\n"
            "    } -> (y);\n"
            "}\n", SP, SP, SP);
        size_t xsz = 1*2*SP*2, wsz = 4*2*SP*2, osz = 4*2*SP*2;
        size_t in_sizes[2] = {xsz, wsz};
        float x[2]={3,5}, w[8]={1,2,2,4,3,6,4,8};
        float out[8]; float* ip[]={x,w}; int ic[]={2,8};
        run_test("C-bcast mul [1,1,2,SP]*[1,4,2,SP]", mil, 2, in_sizes, osz, ip, ic, out, 8);
    }
    
    // Test 7: N-broadcast mul with MATCHING iosurface sizes 
    // The IOSurface for [1,4,1,SP] is 4*SP*2 = 256 bytes
    // The IOSurface for [4,4,1,SP] is 16*SP*2 = 1024 bytes  
    // Maybe the issue was wrong IOSurface size in test_dynamic11?
    {
        char mil[2048];
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, 2, 1, %d]> x, tensor<fp16, [4, 2, 1, %d]> W) {\n"
            "        tensor<fp16, [4, 2, 1, %d]> y = mul(x = W, y = x)[name = tensor<string, []>(\"m\")];\n"
            "    } -> (y);\n"
            "}\n", SP, SP, SP);
        size_t xsz = 1*2*SP*2, wsz = 4*2*SP*2, osz = 4*2*SP*2;
        size_t in_sizes[2] = {xsz, wsz};
        float x[2]={3,5}, w[8]={1,2,2,4,3,6,4,8};
        float out[8]; float* ip[]={x,w}; int ic[]={2,8};
        run_test("N-bcast mul [1,2,1,SP]*[4,2,1,SP]", mil, 2, in_sizes, osz, ip, ic, out, 8);
    }
    
    printf("\n  compiles=%d\n", ane_compile_count());
    objc_autoreleasePoolPop(pool);
    return 0;
}
