// test_ane_limits3.cpp — Test if unloading kernels reclaims compile budget
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "core/ane_runtime.h"
#include <ane_lm/common.h>

extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void  objc_autoreleasePoolPop(void*);

using namespace ane_lm;

static uint16_t* make_bf16(size_t numel, int seed) {
    uint16_t* w = (uint16_t*)malloc(numel * 2);
    srand(seed);
    for (size_t i = 0; i < numel; i++)
        w[i] = f32_to_bf16(((float)(rand() % 1000) / 10000.0f) - 0.05f);
    return w;
}

// Test 1: Compile FFN, free it, compile another. Does the budget reset?
static void test_compile_free_recompile() {
    printf("\n=== TEST: Compile-free-recompile cycle (135MB FFN) ===\n");
    ane_set_persist_cache(false);
    
    int dim = 2560, inter = 9216;
    int max = 60;
    
    for (int i = 0; i < max; i++) {
        uint16_t* g = make_bf16((size_t)inter * dim, i * 3);
        uint16_t* u = make_bf16((size_t)inter * dim, i * 3 + 1);
        uint16_t* d = make_bf16((size_t)dim * inter, i * 3 + 2);
        
        ANEKernel* k = ane_compile_fused_ffn(g, u, d, dim, inter);
        free(g); free(u); free(d);
        
        if (!k) {
            printf("  FAILED at iteration %d (compiles=%d)\n", i + 1, ane_compile_count());
            break;
        }
        
        // Immediately free (unload from ANE)
        ane_free(k);
        
        if ((i + 1) % 5 == 0)
            printf("  Iteration %d: compile+free OK (compiles=%d)\n", i + 1, ane_compile_count());
    }
}

// Test 2: Keep half loaded, free half, see what happens
static void test_mixed_loaded() {
    printf("\n=== TEST: 4B model sim — compile all, keep loaded ===\n");
    printf("  (Like real model: all kernels must stay loaded for inference)\n");
    ane_set_persist_cache(false);
    
    int dim = 2560, inter = 9216;
    int lin_qkv = 6144, lin_val = 4096;
    int full_q = 5120, full_kv = 2560;
    
    std::vector<ANEKernel*> loaded;
    bool failed = false;
    
    int layer_types[] = {
        0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1,
        0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1
    };
    
    for (int L = 0; L < 32 && !failed; L++) {
        bool is_full = (layer_types[L] == 1);
        
        // first_proj
        ANEKernel* fp;
        if (!is_full) {
            uint16_t* wa = make_bf16((size_t)lin_qkv * dim, L * 10);
            uint16_t* wb = make_bf16((size_t)lin_val * dim, L * 10 + 1);
            fp = ane_compile_fused_2(wa, lin_qkv, wb, lin_val, dim);
            free(wa); free(wb);
        } else {
            uint16_t* wa = make_bf16((size_t)full_q * dim, L * 10);
            uint16_t* wb = make_bf16((size_t)full_kv * dim, L * 10 + 1);
            uint16_t* wc = make_bf16((size_t)full_kv * dim, L * 10 + 2);
            fp = ane_compile_fused_3(wa, full_q, wb, full_kv, wc, full_kv, dim);
            free(wa); free(wb); free(wc);
        }
        if (!fp) { printf("  L%d first_proj FAIL (compiles=%d, loaded=%d)\n", L, ane_compile_count(), (int)loaded.size()); failed=true; break; }
        loaded.push_back(fp);
        
        // o_proj
        int o_in = is_full ? full_q : lin_val;
        uint16_t* wo = make_bf16((size_t)dim * o_in, L * 10 + 3);
        ANEKernel* op = ane_compile_matmul(wo, dim, o_in);
        free(wo);
        if (!op) { printf("  L%d o_proj FAIL (compiles=%d, loaded=%d)\n", L, ane_compile_count(), (int)loaded.size()); failed=true; break; }
        loaded.push_back(op);
        
        // FFN
        uint16_t* g = make_bf16((size_t)inter * dim, L * 10 + 4);
        uint16_t* u = make_bf16((size_t)inter * dim, L * 10 + 5);
        uint16_t* d = make_bf16((size_t)dim * inter, L * 10 + 6);
        ANEKernel* ffn = ane_compile_fused_ffn(g, u, d, dim, inter);
        free(g); free(u); free(d);
        if (!ffn) { printf("  L%d ffn FAIL (compiles=%d, loaded=%d)\n", L, ane_compile_count(), (int)loaded.size()); failed=true; break; }
        loaded.push_back(ffn);
        
        printf("  Layer %d/32 OK (compiles=%d, loaded=%d, %s)\n",
               L+1, ane_compile_count(), (int)loaded.size(), is_full ? "full" : "lin");
    }
    
    printf("  Result: %d layers, %d kernels loaded, %d compiles\n",
           (int)loaded.size() / 3, (int)loaded.size(), ane_compile_count());
    
    for (auto* k : loaded) ane_free(k);
}

// Test 3: Compile with cache — do cached loads avoid the limit?
static void test_cached_loads() {
    printf("\n=== TEST: Cached loads vs compile limit ===\n");
    printf("  Phase 1: Compile 10 FFN kernels with cache enabled\n");
    ane_set_persist_cache(true);
    
    int dim = 2560, inter = 9216;
    
    // Compile 10, cache them
    std::vector<ANEKernel*> batch1;
    for (int i = 0; i < 10; i++) {
        uint16_t* g = make_bf16((size_t)inter * dim, 1000 + i * 3);
        uint16_t* u = make_bf16((size_t)inter * dim, 1000 + i * 3 + 1);
        uint16_t* d = make_bf16((size_t)dim * inter, 1000 + i * 3 + 2);
        ANEKernel* k = ane_compile_fused_ffn(g, u, d, dim, inter);
        free(g); free(u); free(d);
        if (!k) { printf("  Phase 1 failed at %d!\n", i); break; }
        batch1.push_back(k);
    }
    printf("  Phase 1: compiled %d, compiles=%d, cache_loads=%d\n",
           (int)batch1.size(), ane_compile_count(), ane_cache_loads());
    
    // Free them all
    for (auto* k : batch1) ane_free(k);
    batch1.clear();
    printf("  Freed all phase 1 kernels\n");
    
    // Now reload same kernels — should come from cache
    printf("  Phase 2: Reload same 10 kernels from cache\n");
    for (int i = 0; i < 10; i++) {
        uint16_t* g = make_bf16((size_t)inter * dim, 1000 + i * 3);
        uint16_t* u = make_bf16((size_t)inter * dim, 1000 + i * 3 + 1);
        uint16_t* d = make_bf16((size_t)dim * inter, 1000 + i * 3 + 2);
        ANEKernel* k = ane_compile_fused_ffn(g, u, d, dim, inter);
        free(g); free(u); free(d);
        if (!k) { printf("  Phase 2 failed at %d!\n", i); break; }
        batch1.push_back(k);
    }
    printf("  Phase 2: loaded %d, compiles=%d, cache_loads=%d\n",
           (int)batch1.size(), ane_compile_count(), ane_cache_loads());
    
    // Now compile MORE new kernels — does cache load budget count toward limit?
    printf("  Phase 3: Compile MORE unique kernels on top of cached\n");
    for (int i = 0; i < 30; i++) {
        uint16_t* g = make_bf16((size_t)inter * dim, 5000 + i * 3);
        uint16_t* u = make_bf16((size_t)inter * dim, 5000 + i * 3 + 1);
        uint16_t* d = make_bf16((size_t)dim * inter, 5000 + i * 3 + 2);
        ANEKernel* k = ane_compile_fused_ffn(g, u, d, dim, inter);
        free(g); free(u); free(d);
        if (!k) {
            printf("  Phase 3 failed at new kernel %d (compiles=%d, cache_loads=%d, total_loaded=%d)\n",
                   i + 1, ane_compile_count(), ane_cache_loads(), (int)(batch1.size() + i));
            break;
        }
        batch1.push_back(k);
        if ((i + 1) % 5 == 0)
            printf("  Phase 3: +%d new (compiles=%d, cache_loads=%d, total_loaded=%d)\n",
                   i + 1, ane_compile_count(), ane_cache_loads(), (int)batch1.size());
    }
    
    for (auto* k : batch1) ane_free(k);
}

int main(int argc, char* argv[]) {
    void* pool = objc_autoreleasePoolPush();
    g_verbose = false;
    ane_init();
    if (!ane_available()) { fprintf(stderr, "ANE not available!\n"); return 1; }
    
    if (argc < 2) {
        printf("Usage:\n");
        printf("  test_ane_limits3 cycle    — compile-free-recompile\n");
        printf("  test_ane_limits3 loaded   — all kernels loaded simultaneously\n");
        printf("  test_ane_limits3 cached   — cached loads vs compile limit\n");
        return 1;
    }
    
    if (strcmp(argv[1], "cycle") == 0) test_compile_free_recompile();
    else if (strcmp(argv[1], "loaded") == 0) test_mixed_loaded();
    else if (strcmp(argv[1], "cached") == 0) test_cached_loads();
    
    printf("\nFinal: compiles=%d, cache_loads=%d\n", ane_compile_count(), ane_cache_loads());
    objc_autoreleasePoolPop(pool);
    return 0;
}
