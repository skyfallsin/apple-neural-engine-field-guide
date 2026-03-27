// test_ane_matmul.cpp — Test if ANE supports dynamic (non-constant) weight inputs via matmul
// If this works, we can compile ONE FFN kernel and reuse across all 32 layers
// by writing different weight data into the input IOSurfaces each time.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <IOSurface/IOSurface.h>
#include "core/ane_runtime.h"
#include <ane_lm/common.h>

extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void  objc_autoreleasePoolPop(void*);

// We need access to ane_compile_raw internals — replicate the critical parts
// For this test we'll use a minimal approach: generate MIL with matmul,
// compile through the existing infrastructure by adding a new function

namespace ane_lm {

// Forward-declare the internal compile function (defined in ane_runtime.cpp)
// We'll duplicate the MIL-gen + compile pattern instead

#define SP ANE_SPATIAL
#define MIL_HEADER \
    "program(1.0)\n" \
    "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-version\", \"3505.4.1\"}})]\n" \
    "{\n"

} // namespace ane_lm

using namespace ane_lm;

static uint16_t* make_bf16(size_t numel, int seed) {
    uint16_t* w = (uint16_t*)malloc(numel * 2);
    srand(seed);
    for (size_t i = 0; i < numel; i++)
        w[i] = f32_to_bf16(((float)(rand() % 1000) / 10000.0f) - 0.05f);
    return w;
}

// Test 1: Can matmul with two runtime inputs compile on ANE?
// Simple: y = matmul(x, W) where both x and W are inputs (no constants)
// x: [1, in_dim, 1, 1] reshaped, W: [in_dim, out_dim, 1, 1]
// Actually MIL matmul operates on 2D+ tensors. Let's try the natural form:
// x = [1, in_dim], W = [in_dim, out_dim] → y = [1, out_dim]
// But ANE expects 4D tensors: [N, C, H, W]
// So: x = [1, 1, 1, in_dim], W = [1, in_dim, 1, out_dim], y = matmul(x, W) = [1, 1, 1, out_dim]
// Or use the ANE layout: x = [1, in_dim, 1, SP], W = [1, in_dim, 1, out_dim]
// matmul would need compatible shapes...

// Let's try multiple MIL formulations and see which one ANE accepts:

static void test_matmul_2d() {
    printf("\n=== Test: matmul with 2D tensors ===\n");
    // y = matmul(x, W)
    // x: [1, dim], W: [dim, out_dim], y: [1, out_dim]
    int in_dim = 64, out_dim = 64;
    
    char buf[4096];
    int n = snprintf(buf, sizeof(buf),
        MIL_HEADER
        "    func main<ios16>(tensor<fp16, [1, %d]> x, tensor<fp16, [%d, %d]> W) {\n"
        "        tensor<fp16, [1, %d]> y = matmul(x = x, y = W)[name = tensor<string, []>(\"mm\")];\n"
        "    } -> (y);\n"
        "}\n",
        in_dim, in_dim, out_dim, out_dim);
    
    printf("  MIL: matmul([1,%d], [%d,%d]) → [1,%d]\n", in_dim, in_dim, out_dim, out_dim);
    
    // Compile using ane_compile_matmul's approach but with custom MIL
    // We'll use the simpler approach: just try to compile a matmul kernel
    // For now, let's try via the existing conv-based approach but with TWO inputs
    printf("  (Cannot test directly without exposing ane_compile_raw — testing via wrapper)\n");
    printf("  Skipping for now, trying conv-with-input-weights approach\n");
}

// Test 2: What if we make a conv kernel where weight comes from a second input?
// MIL conv requires `weight` to be a constant. But can we use a `reshape` trick?
// E.g.: input W_flat as [1, out*in, 1, 1], reshape to [out, in, 1, 1], use in conv?
// NO — reshape output fed to conv's `weight` parameter must be constant at compile time.

// Test 3: Linear (which maps to einsum/matmul internally)
// MIL `linear` op: y = matmul(x, W^T) + b
// If W is an input tensor (not const), does it compile?

// Test 4: The real test — we need to expose ane_compile_raw.
// Let's add a test function directly in ane_runtime that we can call.

// Actually, the cleanest approach: write a small helper that calls ane_compile_raw
// with our custom MIL. We already have the full ObjC machinery in ane_runtime.cpp.
// Let's add a test entry point there.

// For now, let's test something simpler but informative:
// Can we make a kernel with MULTIPLE INPUTS (not just one)?
// The existing chunked FFN already uses [1, 2*dim, 1, SP] as a packed input.
// But that's packing data into one IOSurface. True multi-input means multiple IOSurfaces.

// The ANERequest already supports multiple inputs/outputs (nInputs/nOutputs).
// So let's compile a kernel with 2 inputs via the raw API.

// We need to modify ane_runtime to expose this. Let's do it differently:
// Just test whether a kernel with matmul(input1, input2) compiles by adding
// a new function to ane_runtime.

int main(int argc, char* argv[]) {
    void* pool = objc_autoreleasePoolPush();
    g_verbose = true;
    ane_init();
    if (!ane_available()) { fprintf(stderr, "ANE not available!\n"); return 1; }
    
    printf("ANE initialized. Testing dynamic weight approaches.\n\n");
    
    // Summary of what we know:
    printf("=== FINDINGS SO FAR ===\n");
    printf("1. Compile limit is NOT a count — it's a loaded-model MEMORY limit\n");
    printf("   - 229 tiny (8KB) kernels: OK (all loaded simultaneously)\n");
    printf("   - 50 large (~70MB avg) kernels: LIMIT\n");
    printf("   - 15 FFN (135MB) kernels: LIMIT\n");
    printf("   - compile+free cycle: UNLIMITED (60+ tested)\n");
    printf("   - Estimated budget: ~3.5GB loaded simultaneously\n");
    printf("\n");
    printf("2. Per-kernel timing at 4B dims (dim=2560, inter=9216):\n");
    printf("   - FFN (135MB):   compile=450ms, cache_load=140ms, eval=2ms, free=6ms\n");
    printf("   - fused_2 (50MB): compile=139ms, cache_load=39ms, eval=0.7ms, free=4ms\n");
    printf("   - matmul (20MB):  compile=45ms, cache_load=17ms, eval=0.5ms, free=3ms\n");
    printf("   - LM head (80MB): compile=191ms, cache_load=76ms, eval=1ms, free=8ms\n");
    printf("\n");
    printf("3. 4B model needs: 32×3 layer + 16 LM head = 112 kernels\n");
    printf("   Total weight data: ~6.6GB → exceeds ~3.5GB loaded limit\n");
    printf("\n");
    printf("4. Per-layer swap approach: 32×(140+2+6)ms = 4.7s per token → 0.2 tok/s\n");
    printf("   (load+eval+free FFN each layer, keep attn loaded)\n");
    printf("\n");
    
    printf("=== APPROACH TO TEST: Dynamic weight matmul ===\n");
    printf("Goal: compile 1 FFN kernel, reuse across 32 layers\n");
    printf("Method: use matmul() op with runtime input tensors for weights\n");
    printf("Cost: memcpy 135MB weights per layer = ~4.5ms × 32 = 144ms overhead\n");
    printf("Plus eval: 32 × 2ms = 64ms\n");
    printf("Total: ~208ms per token → ~4.8 tok/s (before LM head)\n");
    printf("\nNeed to test if ANE accepts matmul with dynamic (non-constant) inputs.\n");
    printf("This requires adding a test function to ane_runtime.cpp.\n");
    
    objc_autoreleasePoolPop(pool);
    return 0;
}
