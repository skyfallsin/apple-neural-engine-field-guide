// test_dynamic7.cpp — Matvec using ONLY W=SP tensors (the ANE hardware constraint)
//
// KEY INSIGHT: ANE runtime inputs MUST have innermost dim W=SP=32.
// This means we cannot have weight tensors shaped [out,in,1,1] as inputs.
// ALL inputs must be [..., SP].
//
// APPROACH: Column-wise matvec
// Instead of standard y = W @ x where W is [out, in]:
// 1. Pack the weight matrix column-by-column as SP-strided channels
// 2. Multiply each weight column by the corresponding x element (scalar broadcast)
// 3. Sum the results
//
// Concretely for y = W @ x, y[o] = sum_i W[o,i] * x[i]:
// - W stored as: for each column i, W[:,i] is a vector of length out_dim
//   Reshape: W_col[i] = [1, out_dim, 1, SP] with W[:,i] at position 0 of SP
// - x[i] is broadcast-multiplied with W_col[i]
//
// But we can't have in_dim separate inputs! We need to pack it.
//
// BETTER APPROACH: Use the spatial dimension SP for the reduction.
// For small in_dim <= SP:
//   x: [1, 1, 1, SP] with x[i] at position i (use first in_dim positions)
//   W: [1, out_dim, 1, SP] with W[o,i] at position i for channel o
//   mul: [1, out_dim, 1, SP] — each channel c has W[c,:] * x[:]
//   reduce_sum axis=3 -> [1, out_dim, 1, 1]... but output needs W=SP too
//
// For in_dim > SP, tile in_dim across channels: 
//   W: [1, out_dim * tiles, 1, SP]
//   Do tiled multiply + accumulate
//
// Let's start simple: in_dim <= SP
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

static float max_abs_diff(const float* a, const float* b, int n) {
    float mx = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static void cpu_matvec(float* y, const float* W, const float* x, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        double sum = 0;
        for (int i = 0; i < in_dim; i++)
            sum += (double)W[o * in_dim + i] * (double)x[i];
        y[o] = (float)sum;
    }
}

// ===== Approach 1: Use spatial dim for inner product, in_dim <= SP =====
// x packed in W dimension: [1, 1, 1, SP] — x[i] at spatial position i
// W packed: [1, out, 1, SP] — W[o,i] at channel o, spatial position i
// mul: [1, out, 1, SP] (elementwise)
// reduce_sum axis=3: [1, out, 1, 1] — need to get back to SP
// But output can't be [1, out, 1, 1]! It needs W=SP.
// 
// FIX: Don't reduce. Instead, after mul, we have the products in spatial positions.
// We need a horizontal sum across the spatial dim. reduce_sum along W dim should work
// IF the output gets broadcast back to SP somehow... 
// Actually, let's just not reduce — use reduce_sum with keep_dims=true to get [1,out,1,1]
// then broadcast-add with a zero [1,out,1,SP] to expand back.
// OR: output the [1,out,1,1] and accept the small IOSurface for the output?
// No — the output also needs W=SP for reading!
//
// Alternative: reduce_sum with keep_dims=true gives [1,out,1,1], then 
// tile/reshape to [1,out,1,SP]. Let's try.
static bool test_spatial_matvec() {
    printf("=== Spatial matvec (in_dim <= SP) ===\n");
    
    int OUT = 8, IN = 8; // IN <= SP
    
    // Try: mul + reduce_sum(axis=3, keep_dims=true) to get [1,OUT,1,1]
    // Then add a zeros tensor to broadcast to SP? No, the output IS [1,OUT,1,1].
    // But we proved [*,*,1,1] outputs fail too... or do outputs have different rules?
    // The working tests had output [*,*,1,SP]. Let's check if output [*,*,1,1] works.
    
    // Actually let me think differently. The reduce_sum result is [1,OUT,1,1].
    // If we reshape it to [1,OUT,1,SP] (repeating the value? no, reshape doesn't replicate).
    // We can't go from [1,OUT,1,1] to [1,OUT,1,SP] with reshape (different element count).
    //
    // Better: keep the products in spatial positions and DON'T reduce.
    // Instead, the host reads from ALL SP positions and sums on CPU.
    // For in_dim=8 with SP=32, positions 0-7 have products, 8-31 are zero.
    // CPU sums positions 0..in_dim-1 for each channel.
    // Output: [1, OUT, 1, SP] — host reads and sums.
    
    printf("  Strategy: mul only, host-side reduction\n");
    char mil[2048];
    snprintf(mil, sizeof(mil),
        "program(1.0)\n"
        "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n"
        "{\n"
        "    func main<ios16>(tensor<fp16, [1, 1, 1, %d]> x, tensor<fp16, [1, %d, 1, %d]> W) {\n"
        "        tensor<fp16, [1, %d, 1, %d]> y = mul(x = x, y = W)[name = tensor<string, []>(\"m\")];\n"
        "    } -> (y);\n"
        "}\n",
        SP, OUT, SP,
        OUT, SP);
    
    size_t in_sizes[2] = {
        (size_t)1 * SP * sizeof(uint16_t),       // x: [1,1,1,SP]
        (size_t)OUT * SP * sizeof(uint16_t)       // W: [1,OUT,1,SP]
    };
    size_t out_size = (size_t)OUT * SP * sizeof(uint16_t);
    
    ANEKernel* k = ane_compile_mil(mil, 2, in_sizes, 1, &out_size);
    if (!k) { printf("  Compile FAILED\n"); return false; }
    printf("  Compiled OK\n");
    
    // Prepare data: x vector and W matrix
    float x_f32[SP] = {};
    float W_f32[8 * SP] = {}; // OUT * SP
    
    // x = [1, 2, 3, ..., 8, 0, 0, ..., 0] in spatial positions
    for (int i = 0; i < IN; i++) x_f32[i] = (float)(i + 1);
    
    // W: identity matrix packed into spatial positions
    // W[o,i] at channel o, spatial position i
    for (int o = 0; o < OUT; o++)
        for (int i = 0; i < IN; i++)
            W_f32[o * SP + i] = (o == i) ? 1.0f : 0.0f;
    
    // Write x: [1,1,1,SP] — data goes into spatial positions directly
    // The IOSurface has 1 channel, so data is at positions 0..SP-1
    uint16_t* x_fp16 = (uint16_t*)calloc(SP, sizeof(uint16_t));
    for (int i = 0; i < SP; i++) x_fp16[i] = f32_to_f16(x_f32[i]);
    ane_write_surface_raw(k, 0, x_fp16, SP * sizeof(uint16_t));
    
    // Write W: [1,OUT,1,SP] — OUT channels, each with SP spatial values
    // Channel o at offset o*SP
    uint16_t* w_fp16 = (uint16_t*)calloc(OUT * SP, sizeof(uint16_t));
    for (int o = 0; o < OUT; o++)
        for (int i = 0; i < SP; i++)
            w_fp16[o * SP + i] = f32_to_f16(W_f32[o * SP + i]);
    ane_write_surface_raw(k, 1, w_fp16, OUT * SP * sizeof(uint16_t));
    
    // Eval
    // Output [1,OUT,1,SP]: products at each spatial position
    // We read the full SP-wide output for each channel, then sum positions 0..IN-1 on CPU
    float* y_products = (float*)calloc(OUT * SP, sizeof(float));
    
    // Read output manually (all SP positions per channel)
    float* outptrs[] = {y_products};
    // We want to read OUT channels, each SP wide
    // ane_eval_raw_outputs reads at SP stride (positions 0, SP, 2*SP, ...)
    // But we want the full spatial data for each channel.
    // Let me read the raw surface instead.
    
    bool ok = false;
    // Dispatch eval
    {
        // Use internal eval
        // Actually, ane_eval_raw_outputs reads SP-strided (position 0 of each channel).
        // For this test we need ALL spatial positions. Let me just read the raw surface.
        // But we can't access k->ioOutputs from here... 
        // 
        // Wait — for the identity matrix case, output[o, pos_i] = (o==i) ? x[i] : 0
        // After host-side reduction: y[o] = x[o] (identity)
        // So position 0 of each channel should have x[0]*W[o,0].
        // For identity: y_products[o*SP + 0] = (o==0) ? x[0] : 0
        // ane_eval_raw_outputs reads position 0 of each channel — not the full vector.
        
        // I need to read ALL spatial positions. Let me add a raw output read function.
        // For now, just test if eval succeeds at all:
        int out_chs[] = {OUT};
        ok = ane_eval_raw_outputs(k, outptrs, out_chs);
    }
    
    if (!ok) { printf("  EVAL FAILED!\n"); }
    else {
        printf("  EVAL OK!\n");
        // y_products has position 0 of each channel (SP-strided read)
        // For identity matrix: y_products[o] = x[0] * W[o,0] = (o==0 ? 1 : 0)
        printf("  Channel 0 pos 0 (SP-strided): [");
        for (int o = 0; o < OUT; o++) printf("%.2f%s", y_products[o], o<OUT-1?", ":"");
        printf("]\n");
        printf("  Expected for identity@[1..8]: pos0 only has x[0]*W[o,0]\n");
        printf("  Need full spatial read to sum across positions...\n");
    }
    
    ane_free(k);
    free(x_fp16); free(w_fp16); free(y_products);
    return ok;
}

// ===== Approach 2: Tiled matvec — pack in_dim into channel+spatial =====
// For general in_dim = tiles * SP:
// x: [1, tiles, 1, SP] — x values packed across channels and spatial
// W: [tiles, out_dim, 1, SP] — W[o,t*SP+s] at [t, o, 0, s]  
// Wait, this is getting complex. Let me try the simplest thing that could work first.
//
// Actually the simplest approach: for matvec y = W @ x where in_dim = T*SP:
// x: [1, T, 1, SP] where x[t*SP + s] is at channel t, spatial s
// W: [1, out*T, 1, SP] where for output o and tile t, W[o, t*SP+s] is at channel (o*T+t), spatial s
// mul broadcasts x[1,T,1,SP] with W[1,out*T,1,SP]... doesn't broadcast! Different C.
// 
// Need to think more carefully. Let me just add a raw output reader first.

int main() {
    void* pool = objc_autoreleasePoolPush();
    g_verbose = false;
    ane_init();
    if (!ane_available()) { fprintf(stderr, "ANE not available!\n"); return 1; }
    ane_set_persist_cache(false);
    
    test_spatial_matvec();
    
    printf("\ncompiles=%d\n", ane_compile_count());
    objc_autoreleasePoolPop(pool);
    return 0;
}
