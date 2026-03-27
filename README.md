# Apple Neural Engine — Reverse-Engineering Field Guide

Undocumented hardware constraints, IOSurface layout rules, MIL programming patterns, and behavioral quirks of Apple's Neural Engine on Apple Silicon, discovered through systematic testing via private `AppleNeuralEngine.framework` APIs.

Apple doesn't publish any of this. Every LLM inference engine on Mac (llama.cpp, MLX, MLC-LLM) ignores the Neural Engine entirely. This guide documents what we've learned getting [LLM inference running directly on ANE](https://github.com/skyfallsin/ANE-LM).

---

## Table of Contents

- [IOSurface Layout](#iosurface-layout)
- [Operations on Runtime Tensors](#operations-on-runtime-tensors)
- [The `tile` Poison](#the-tile-poison)
- [Dynamic-Weight Matrix-Vector Multiply](#dynamic-weight-matrix-vector-multiply)
- [Why Conv with Dynamic Weights Fails](#why-conv-with-dynamic-weights-fails)
- [Compile Budget and Caching](#compile-budget-and-caching)
- [Dispatch Overhead](#dispatch-overhead)
- [MIL Programming Reference](#mil-programming-reference)
- [Chunked FFN for Large Models](#chunked-ffn-for-large-models)
- [Methodology](#methodology)

---

## IOSurface Layout

The ANE communicates with the host CPU through IOSurface buffers. Tensors use a 4D `[N, C, H, W]` layout with a hardware-imposed spatial dimension.

### The SP=32 Rule

**All runtime input IOSurfaces must have W (innermost dimension) ≥ 32.**

This is the single most important constraint. We call this value `SP` (spatial). W=16 or W=1 causes silent eval failure — the kernel compiles fine, the eval call returns, but the output is garbage or the status code is non-zero.

```
SP = 32  (ANE spatial dimension, minimum W for any runtime IOSurface)
```

This appears to be the minimum granularity of the ANE's vector processing unit. It processes 32 FP16 values at a time along the W axis.

### Memory Layout

Data is FP16 with stride SP=32 on the W axis:

```
Tensor [N=1, C=2560, H=1, W=32]:
  - 2560 channels × 32 FP16 values = 163,840 values
  - Size: 2560 × 32 × 2 bytes = 160 KB
```

For a vector of logical dimension D (e.g. hidden_size=2560), you pack it as:
- `T = ceil(D / SP)` channels, each with SP=32 values
- Tensor shape: `[1, T, 1, SP]`
- Last channel is zero-padded if D is not a multiple of 32

The `H` dimension is typically 1 for vector/matmul operations. `N` and `C` can vary freely.

### Writing Data to IOSurfaces

IOSurfaces are locked for CPU access, written in FP16, then unlocked before ANE dispatch. The ANE reads directly from the IOSurface's backing memory — there's no separate copy step.

For strided writes (logical channels mapped to SP-strided layout), each channel occupies SP×2 bytes in the surface, regardless of how many values are logically meaningful.

---

## Operations on Runtime Tensors

"Runtime tensors" means IOSurface inputs — data that varies per eval, as opposed to constant weights baked into the compiled MIL program.

| Operation | Status | Notes |
|-----------|--------|-------|
| `add(a, b)` | ✅ Works | Same-shape and N-broadcast both work reliably |
| `mul(a, b)` | ✅ Works (same-shape) | Same-shape elementwise multiply works perfectly for all tested sizes up to 2560×2560 |
| `mul(a, b)` | ⚠️ Unreliable (N-broadcast) | Works in isolation, but may produce incorrect results when composed with other ops in multi-operation MIL programs |
| `reduce_sum(axis)` | ✅ Works | Tested on axis=1 (C dimension). Correctly sums across channels |
| `reshape` | ✅ Works | Even changing the N dimension works — contradicts our early hypothesis that N was immutable for runtime tensors |
| `tile` | ☠️ Poisons ANE | Compiles and may eval correctly, but corrupts global state. See [The `tile` Poison](#the-tile-poison) |
| `conv` (dynamic weights) | ❌ Silent failure | Compiles but weights are silently ignored at eval. See [Why Conv Fails](#why-conv-with-dynamic-weights-fails) |
| `slice_by_size` | ✅ Works | Used in fused multi-output projections (Q/K/V split) |
| `silu` / `sigmoid` | ✅ Works | Activation functions work as constant-free unary ops |

### Broadcasting Rules

- **N-axis broadcast**: `[1, C, H, W]` op `[N, C, H, W]` — works for `add`, unreliable for `mul` in complex programs
- **C-axis broadcast**: Not tested extensively; avoid if possible
- **Same-shape**: Always works. When in doubt, make both inputs identical shape

---

## The `tile` Poison

**The `tile` MIL operation corrupts global ANE state.** This is the most dangerous undocumented behavior we found.

### What Happens

1. You compile a MIL program containing `tile`
2. You evaluate it — it may even produce correct output
3. Every subsequent ANE kernel evaluation **in the same process** fails with status `0x1d`
4. Previously-working kernels that have nothing to do with `tile` now fail
5. The corruption persists until process exit

### How We Discovered It

Through systematic isolation testing:

```
Step 1: Eval program A (mul+reduce_sum) → ✅ works
Step 2: Eval program B (tile+mul)        → ✅ works (output looks correct)
Step 3: Eval program A again             → ❌ fails (status 0x1d)
Step 4: Compile fresh program C          → ❌ fails
Step 5: Nothing works until process exit
```

We verified this by testing every combination:
- `tile` + `add` → poisons
- `tile` + `mul` → poisons
- `tile` alone → poisons (even if the tile output isn't used by another op)
- Programs without `tile` before/after → never poison

### Workaround

**Never use `tile` in any MIL program that will run on ANE.** Perform tiling on the CPU before writing to IOSurfaces. This is exactly what the [dynamic matvec strategy](#dynamic-weight-matrix-vector-multiply) does — CPU-side memcpy to repeat the input vector.

---

## Dynamic-Weight Matrix-Vector Multiply

The key research contribution: a working method for `y = W @ x` where the weight matrix `W` is a runtime input, not compiled into the kernel. This enables weight-swapping without recompilation.

### Why It's Hard

ANE's native conv operation is the natural way to do matmul. But conv reads weights from a dedicated internal hardware bus that's populated from the BLOBFILE (compiled weights) — not from runtime IOSurface inputs. You can declare a conv with a runtime weight input, it compiles, but the weight data is silently ignored at eval time.

### Working Strategy: CPU-Side Tiling

Instead of conv, use elementwise `mul` + `reduce_sum`:

```
x:       [1, T, 1, SP]              — input vector (T = ceil(in_dim/SP))
W:       [out_dim, T, 1, SP]        — weight matrix (one row per N slice)

Step 1 (CPU): Tile x into x_tiled [out_dim, T, 1, SP]
              (memcpy the same T*SP values into each of the out_dim N slices)

Step 2 (ANE): mul(x_tiled, W)       → [out_dim, T, 1, SP]
              (same-shape multiply, no broadcasting needed)

Step 3 (ANE): reduce_sum(axis=1)    → [out_dim, 1, 1, SP]
              (sum across T chunks — partial dot products)

Step 4 (CPU): Sum the 32 (SP) values per output channel → final y[out_dim]
```

Both inputs are the same shape, so we avoid N-broadcast (which is unreliable for `mul`). The CPU-side tiling is just memcpy — fast and simple.

### Performance (2560×2560 — Qwen3-4B hidden size)

| Component | Time |
|-----------|------|
| W memcpy to IOSurface | 0.27ms |
| x tile + write to IOSurface | 0.39ms |
| ANE eval (mul + reduce_sum) | 1.00ms |
| **Total dynamic matvec** | **1.65ms** |
| Const-weight conv (baseline) | 0.32ms |
| Const-weight compile time | ~60ms |

**Dynamic is ~5× slower per eval** than const-weight conv, but has **zero recompile cost on weight swap** (vs ~60ms per weight set for const-weight). Break-even is ~1 weight swap per 5 evals.

### Use Cases

- **LoRA adapter hot-swapping** — switch adapters without 60ms×N_layers recompile
- **Mixture-of-experts routing** — load selected expert weights at runtime
- **Speculative decoding** — swap between draft and target model weights
- **Any scenario where compile time dominates** — dynamic matvec trades per-eval speed for instant weight changes

### Correctness

Tested against CPU FP16 reference implementation at all sizes from 8×32 up to 2560×2560. Maximum absolute error is within FP16 accumulation tolerance (~1e-3 for large dimensions).

---

## Why Conv with Dynamic Weights Fails

This is worth understanding because it's the obvious first approach everyone will try.

ANE's conv unit has two data paths:
1. **Activation path** — reads from IOSurface inputs (runtime data)
2. **Weight path** — reads from a dedicated internal weight bus, populated at compile time from BLOBFILE data

When you declare a conv in MIL with a runtime (non-constant) weight input:
- The MIL compiler accepts it
- ANE compilation succeeds
- At eval time, the weight IOSurface data is **written but never read by the conv unit**
- The conv unit reads from the weight bus, which contains whatever was there from compilation (likely zeros or stale data)
- Output is garbage with no error code

This is a fundamental hardware architecture constraint, not a software bug. The conv unit's weight path is physically separate from the IOSurface data path.

---

## Compile Budget and Caching

### Kernel Limits

There's a per-process limit on simultaneously loaded ANE kernels. The exact number depends on kernel size, but for reference:
- Qwen3-4B loads 226 kernels (216 layer + 10 LM head) — within budget
- Qwen3.5-0.8B loads 72 layer kernels — well within budget
- Freeing kernels (`ane_free`) reclaims budget — you can compile → eval → free in a loop for models that exceed the limit

### Persistent Cache

The ANE framework supports persistent compile caching:
- Compiled ANE programs are stored on disk (keyed by MIL program text + weight data hash)
- Cache load is **~10× faster** than fresh compilation
- Qwen3-4B: ~28s first run → ~8s cached
- Cache is per-user, survives process restarts
- Different models don't collide (different weight data → different cache keys)

### Compile Times

| Kernel Type | Compile Time | Notes |
|-------------|-------------|-------|
| Single matmul (2560×2560) | ~60ms | One weight matrix |
| Fused QKV (3 matrices) | ~100ms | Q+K+V in one kernel |
| Fused SwiGLU FFN | ~150ms | gate+up+silu+mul+down |
| Chunked FFN (per chunk) | ~80ms | Partial intermediate dim |
| Cache load (any) | ~5-10ms | ~10× faster than compile |

---

## Dispatch Overhead

This section explains why ANE inference at 4B scale tops out at ~6 tok/s despite the hardware being capable of much more.

### The Numbers

```
Qwen3-4B generation: ~6.2 tok/s
Time per token:      ~161ms
ANE dispatches/token: 216
Time per dispatch:   ~0.75ms average
```

Each dispatch involves:
1. Lock input IOSurface for CPU write access
2. Convert FP32 → FP16 and write to surface (with SP-strided layout)
3. Unlock surface
4. Submit ANE program for execution
5. Wait for ANE completion
6. Lock output IOSurface for CPU read access
7. Read FP16 output and convert to FP32
8. Unlock surface

The actual ANE compute time for a 2560×2560 matmul at ~15 TOPS FP16 is **~0.0004ms**. That's less than 0.1% of the dispatch time. The other 99.9% is CPU↔ANE coordination overhead.

### Implications

- **Quantization won't help much** — INT8/INT4 reduces memory transfer slightly but dispatch count stays the same
- **Fusing more ops into fewer kernels** is the highest-leverage optimization — going from 6 dispatches/layer to 2-3 would nearly double throughput
- **Moving attention to ANE** would eliminate the biggest source of mid-layer CPU work and its associated dispatch round-trips
- **Batch prefill** (multiple tokens per dispatch) amortizes dispatch cost over more useful work — this is where ANE could really shine, processing a batch of tokens in parallel through its vector units

---

## MIL Programming Reference

ANE-LM compiles MIL (Machine Learning Intermediate Language) programs via the private `Espresso` and `ANECompiler` frameworks. Here are practical patterns that work.

### Basic Structure

```
program(1.0)
[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{"coremlc-version", "3505.4.1"}})]
{
    func main<ios16>(tensor<fp16, [1, 80, 1, 32]> x) {
        // operations...
        return out;
    }
}
```

- The `buildInfo` dict with `coremlc-version` is **required**. Without it, the parser fails. The value `"3505.4.1"` works across macOS 13-15; the exact version doesn't appear to matter.
- `ios16` deployment target works for all patterns described here. Higher targets may enable additional ops.
- All input/output tensors must be FP16 with W dimension ≥ SP (32).

### Fused Multi-Output Projections

Compute Q, K, V projections in one kernel by concatenating weight matrices:

```
func main<ios16>(tensor<fp16, [1, T, 1, SP]> x) {
    // Combined QKV weight: [q_dim + k_dim + v_dim, T, 1, SP]
    const combined_w = const(...);
    y = conv(x, combined_w, ...);
    q = slice_by_size(y, begin=[0,0,0,0], size=[1, q_dim, 1, SP]);
    k = slice_by_size(y, begin=[0,q_dim,0,0], size=[1, k_dim, 1, SP]);
    v = slice_by_size(y, begin=[0,q_dim+k_dim,0,0], size=[1, v_dim, 1, SP]);
    return q, k, v;
}
```

This turns 3 ANE dispatches into 1.

### Fused SwiGLU FFN

The entire FFN block as one kernel:

```
func main<ios16>(tensor<fp16, [1, T, 1, SP]> x) {
    const gate_w = const(...);  // [inter, T, 1, SP]
    const up_w = const(...);    // [inter, T, 1, SP]
    const down_w = const(...);  // [dim, inter_T, 1, SP]

    gate = conv(x, gate_w, ...);
    up = conv(x, up_w, ...);
    activated = silu(gate);
    gated = mul(activated, up);
    out = conv(gated, down_w, ...);
    return out;
}
```

5 operations, 1 dispatch.

### Chunked FFN

When intermediate_size is too large for a single kernel (ANE has finite internal buffer/register space), split along the intermediate dimension:

```
Chunk 0: gate_w[0:chunk], up_w[0:chunk] → partial down_proj → accum[0]
Chunk 1: gate_w[chunk:2*chunk], up_w[chunk:2*chunk] → partial down_proj → accum[1]
...
Final: sum(accum[0..N]) on CPU
```

Each chunk is a separate ANE kernel. For Qwen3-4B (inter=9728), 4 chunks of 2432 each.

### Dynamic-Weight Matvec Kernel

```
func main<ios16>(
    tensor<fp16, [N, T, 1, SP]> x_tiled,
    tensor<fp16, [N, T, 1, SP]> W
) {
    product = mul(x_tiled, W);
    reduced = reduce_sum(product, axes=[1], keep_dims=true);
    return reduced;
}
```

Where N=out_dim, T=ceil(in_dim/SP). Both inputs same shape — no broadcasting.

---

## Chunked FFN for Large Models

Models with large intermediate dimensions (e.g. Qwen3-4B with inter=9728) exceed ANE's single-kernel capacity. The chunked approach:

1. **Auto-detect**: `ane_ffn_chunk_count(dim, inter)` determines if chunking is needed and how many chunks
2. **Compile**: Each chunk handles `inter/N` of the intermediate dimension. Gate and up projections are sliced, down projection produces partial output
3. **Eval**: Chunks are evaluated sequentially, partial outputs accumulated on CPU
4. **Fallback**: If single-kernel compilation fails (returns null), automatically retry with chunking

This is what enabled Qwen3-4B — the original codebase only had single-kernel FFN and couldn't handle inter > ~4096.

---

## Methodology

These findings come from [25 targeted test programs](tests/), each isolating a specific hypothesis about ANE behavior. The full progression is documented in [`tests/README.md`](tests/README.md), but the key arc:

1. **Can conv accept dynamic weights?** → No. Hardware weight bus, not IOSurface path.
2. **What ops work on runtime inputs?** → add, mul, reduce_sum, reshape. Not conv weights, not tile.
3. **Why do some inputs fail silently?** → IOSurface W dimension must be ≥ 32 (SP).
4. **Can we do matvec without conv?** → Yes: mul + reduce_sum.
5. **Why is mul unreliable with broadcasting?** → tile poisons global ANE state.
6. **Working solution?** → CPU-side tiling (memcpy) + same-shape mul + reduce_sum. 1.65ms at 2560×2560.

Each test was designed to answer one specific question, with clear pass/fail criteria and CPU reference implementations for correctness verification.

---

## Related Projects

- **[ANE-LM](https://github.com/skyfallsin/ANE-LM)** — LLM inference on ANE using these findings. Runs Qwen3-4B at ~6 tok/s on the Neural Engine.
- **[johnmai-dev/ANE-LM](https://github.com/johnmai-dev/ANE-LM)** — Original ANE inference project (Qwen3/3.5 support, ANE runtime, safetensors loader)
- **[maderix/ANE](https://github.com/maderix/ANE)** — Neural network training on ANE via reverse-engineered APIs

## License

MIT
