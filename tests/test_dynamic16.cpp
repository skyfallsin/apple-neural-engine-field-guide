// test_dynamic16.cpp — tile+mul failure isolation
// The tile op output feeding into mul causes eval failure.
// Let's find what about the tile output is wrong.
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    
    printf("=== tile+op combination testing ===\n\n");
    
    // 1. tile + add (does this work?)
    {
        printf("1. tile+add: "); fflush(stdout);
        char mil[2048];
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, 2, 1, %d]> x, tensor<fp16, [4, 2, 1, %d]> W) {\n"
            "        tensor<int32, [4]> r = const()[name = tensor<string, []>(\"r\"), val = tensor<int32, [4]>([4, 1, 1, 1])];\n"
            "        tensor<fp16, [4, 2, 1, %d]> xt = tile(x = x, reps = r)[name = tensor<string, []>(\"t\")];\n"
            "        tensor<fp16, [4, 2, 1, %d]> y = add(x = W, y = xt)[name = tensor<string, []>(\"a\")];\n"
            "    } -> (y);\n"
            "}\n", SP, SP, SP, SP);
        size_t xsz=2*SP*2, wsz=4*2*SP*2;
        size_t in_sizes[2]={xsz, wsz};
        ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &wsz);
        if (!k) { printf("COMPILE FAIL\n"); }
        else {
            uint16_t v2=f32_to_f16(2.0f), v3=f32_to_f16(3.0f);
            ane_write_surface_raw(k, 0, &v2, 2);
            ane_write_surface_raw(k, 1, &v3, 2);
            float y[1]={}; float*op[]={y}; int oc[]={1};
            bool ok = ane_eval_raw_outputs(k, op, oc);
            printf("%s y[0]=%.1f\n", ok?"PASS":"FAIL", y[0]);
            ane_free(k);
        }
    }
    
    // 2. tile + mul (different tile sizes)
    for (int tile_n : {2, 3, 4, 8}) {
        printf("2. tile(%d)+mul: ", tile_n); fflush(stdout);
        char mil[2048];
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, 2, 1, %d]> x, tensor<fp16, [%d, 2, 1, %d]> W) {\n"
            "        tensor<int32, [4]> r = const()[name = tensor<string, []>(\"r\"), val = tensor<int32, [4]>([%d, 1, 1, 1])];\n"
            "        tensor<fp16, [%d, 2, 1, %d]> xt = tile(x = x, reps = r)[name = tensor<string, []>(\"t\")];\n"
            "        tensor<fp16, [%d, 2, 1, %d]> y = mul(x = W, y = xt)[name = tensor<string, []>(\"m\")];\n"
            "    } -> (y);\n"
            "}\n", SP, tile_n, SP, tile_n, tile_n, SP, tile_n, SP);
        size_t xsz=2*SP*2, wsz=(size_t)tile_n*2*SP*2;
        size_t in_sizes[2]={xsz, wsz};
        ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &wsz);
        if (!k) { printf("COMPILE FAIL\n"); continue; }
        uint16_t v2=f32_to_f16(2.0f), v3=f32_to_f16(3.0f);
        ane_write_surface_raw(k, 0, &v2, 2);
        ane_write_surface_raw(k, 1, &v3, 2);
        float y[1]={}; float*op[]={y}; int oc[]={1};
        bool ok = ane_eval_raw_outputs(k, op, oc);
        printf("%s y[0]=%.1f\n", ok?"PASS":"FAIL", y[0]);
        ane_free(k);
    }
    
    // 3. tile on C dim instead of N (tile [1,2,1,SP] with reps=[1,4,1,1] -> [1,8,1,SP])
    {
        printf("3. tile(C)+mul: "); fflush(stdout);
        char mil[2048];
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, 2, 1, %d]> x, tensor<fp16, [1, 8, 1, %d]> W) {\n"
            "        tensor<int32, [4]> r = const()[name = tensor<string, []>(\"r\"), val = tensor<int32, [4]>([1, 4, 1, 1])];\n"
            "        tensor<fp16, [1, 8, 1, %d]> xt = tile(x = x, reps = r)[name = tensor<string, []>(\"t\")];\n"
            "        tensor<fp16, [1, 8, 1, %d]> y = mul(x = W, y = xt)[name = tensor<string, []>(\"m\")];\n"
            "    } -> (y);\n"
            "}\n", SP, SP, SP, SP);
        size_t xsz=2*SP*2, wsz=8*SP*2;
        size_t in_sizes[2]={xsz, wsz};
        ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &wsz);
        if (!k) { printf("COMPILE FAIL\n"); }
        else {
            uint16_t v2=f32_to_f16(2.0f), v3=f32_to_f16(3.0f);
            ane_write_surface_raw(k, 0, &v2, 2);
            ane_write_surface_raw(k, 1, &v3, 2);
            float y[1]={}; float*op[]={y}; int oc[]={1};
            bool ok = ane_eval_raw_outputs(k, op, oc);
            printf("%s y[0]=%.1f\n", ok?"PASS":"FAIL", y[0]);
            ane_free(k);
        }
    }
    
    // 4. expand_dims + concat instead of tile (manual tile)
    {
        printf("4. concat-based tile+mul: "); fflush(stdout);
        char mil[4096];
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, 2, 1, %d]> x, tensor<fp16, [4, 2, 1, %d]> W) {\n"
            // concat x 4 times along axis 0 to get [4, 2, 1, SP]
            "        tensor<bool, []> ci = const()[name = tensor<string, []>(\"ci\"), val = tensor<bool, []>(false)];\n"
            "        tensor<int32, []> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, []>(0)];\n"
            "        tensor<fp16, [4, 2, 1, %d]> xt = concat(values = (x, x, x, x), axis = ax, interleave = ci)"
            "[name = tensor<string, []>(\"c\")];\n"
            "        tensor<fp16, [4, 2, 1, %d]> y = mul(x = W, y = xt)[name = tensor<string, []>(\"m\")];\n"
            "    } -> (y);\n"
            "}\n", SP, SP, SP, SP);
        size_t xsz=2*SP*2, wsz=4*2*SP*2;
        size_t in_sizes[2]={xsz, wsz};
        ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &wsz);
        if (!k) { printf("COMPILE FAIL\n"); }
        else {
            uint16_t v2=f32_to_f16(2.0f), v3=f32_to_f16(3.0f);
            ane_write_surface_raw(k, 0, &v2, 2);
            ane_write_surface_raw(k, 1, &v3, 2);
            float y[1]={}; float*op[]={y}; int oc[]={1};
            bool ok = ane_eval_raw_outputs(k, op, oc);
            printf("%s y[0]=%.1f\n", ok?"PASS":"FAIL", y[0]);
            ane_free(k);
        }
    }
    
    // 5. Just use N-broadcast mul directly (no tile) — verify it works here
    {
        printf("5. Direct N-bcast mul (no tile): "); fflush(stdout);
        char mil[2048];
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, 2, 1, %d]> x, tensor<fp16, [4, 2, 1, %d]> W) {\n"
            "        tensor<fp16, [4, 2, 1, %d]> y = mul(x = W, y = x)[name = tensor<string, []>(\"m\")];\n"
            "    } -> (y);\n"
            "}\n", SP, SP, SP);
        size_t xsz=2*SP*2, wsz=4*2*SP*2;
        size_t in_sizes[2]={xsz, wsz};
        ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &wsz);
        if (!k) { printf("COMPILE FAIL\n"); }
        else {
            uint16_t v2=f32_to_f16(2.0f), v3=f32_to_f16(3.0f);
            ane_write_surface_raw(k, 0, &v2, 2);
            ane_write_surface_raw(k, 1, &v3, 2);
            float y[1]={}; float*op[]={y}; int oc[]={1};
            bool ok = ane_eval_raw_outputs(k, op, oc);
            printf("%s y[0]=%.1f\n", ok?"PASS":"FAIL", y[0]);
            ane_free(k);
        }
    }
    
    // 6. N-bcast mul + reduce (no tile)
    {
        printf("6. N-bcast mul+reduce (no tile): "); fflush(stdout);
        char mil[4096];
        snprintf(mil, sizeof(mil),
            "program(1.0)\n"
            "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
            "{\n"
            "    func main<ios16>(tensor<fp16, [1, 2, 1, %d]> x, tensor<fp16, [4, 2, 1, %d]> W) {\n"
            "        tensor<fp16, [4, 2, 1, %d]> prod = mul(x = W, y = x)[name = tensor<string, []>(\"m\")];\n"
            "        tensor<int32, [1]> ax = const()[name = tensor<string, []>(\"ax\"), val = tensor<int32, [1]>([1])];\n"
            "        tensor<bool, []> kd = const()[name = tensor<string, []>(\"kd\"), val = tensor<bool, []>(true)];\n"
            "        tensor<fp16, [4, 1, 1, %d]> y = reduce_sum(x = prod, axes = ax, keep_dims = kd)"
            "[name = tensor<string, []>(\"rs\")];\n"
            "    } -> (y);\n"
            "}\n", SP, SP, SP, SP);
        size_t xsz=2*SP*2, wsz=4*2*SP*2, osz=4*SP*2;
        size_t in_sizes[2]={xsz, wsz};
        ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &osz);
        if (!k) { printf("COMPILE FAIL\n"); }
        else {
            uint16_t v2=f32_to_f16(2.0f), v3=f32_to_f16(3.0f);
            ane_write_surface_raw(k, 0, &v2, 2);
            ane_write_surface_raw(k, 1, &v3, 2);
            float y[4]={}; float*op[]={y}; int oc[]={4};
            bool ok = ane_eval_raw_outputs(k, op, oc);
            printf("%s y=[%.1f,%.1f,%.1f,%.1f]\n", ok?"PASS":"FAIL", y[0],y[1],y[2],y[3]);
            ane_free(k);
        }
    }
    
    printf("\n  compiles=%d\n", ane_compile_count());
    objc_autoreleasePoolPop(pool);
    return 0;
}
