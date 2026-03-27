# Apple Neural Engine — Reverse-Engineering Field Guide

Hardware constraints, IOSurface layout rules, MIL programming patterns, and behavioral quirks of Apple's Neural Engine on Apple Silicon, discovered through systematic experimentation via private `AppleNeuralEngine.framework` APIs.

Apple publishes none of this information. Every LLM inference engine on Mac (llama.cpp, MLX, MLC-LLM) bypasses the Neural Engine entirely. This guide documents what we learned getting [LLM inference running directly on ANE](https://github.com/skyfallsin/ANE-LM), structured as a series of hypotheses, experiments, and measured results. All findings are reproducible via the [25 test programs](tests/) in this repo.

**⚠️ Experimental:** These findings are based on testing a single machine (see below) against undocumented private APIs. Hardware behavior may vary across chips, OS versions, and future updates. Treat this as a collection of empirical observations, not a specification.

**Hardware tested:** MacBook Pro, Apple M3 Max (128 GB), macOS 26.3.1.

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
  - [Human-AI Collaboration](#human-ai-collaboration)

---

## IOSurface Layout

The ANE communicates with the host CPU through IOSurface buffers. Tensors use a 4D `[N, C, H, W]` layout with a hardware-imposed spatial dimension.

### The SP=32 Constraint

**Hypothesis:** The ANE's vector unit has a minimum processing granularity along the W (innermost) dimension.

**Experiment:** We compiled identical MIL programs with varying W dimensions for runtime input IOSurfaces and evaluated them against CPU reference outputs ([`test_dynamic5.cpp`](tests/test_dynamic5.cpp), [`test_dynamic6.cpp`](tests/test_dynamic6.cpp)).

**Observation:** All runtime input IOSurfaces require W ≥ 32. We refer to this value as `SP` (spatial). Kernels with W=16 or W=1 on runtime inputs compile without error and the eval call returns, but the output contains garbage or the eval returns a non-zero status code. There is no warning or error at compile time.

```
SP = 32  (minimum W for any runtime IOSurface input)
```

This 32-wide minimum is consistent with a SIMD vector unit that processes 32 FP16 values per cycle along the W axis.

### Memory Layout

Data is stored as FP16 with stride SP=32 on the W axis:

```
Tensor [N=1, C=2560, H=1, W=32]:
  - 2560 channels × 32 FP16 values = 163,840 values
  - Size: 2560 × 32 × 2 bytes = 160 KB
```

For a vector of logical dimension D (e.g. hidden_size=2560), we pack it as:
- `T = ceil(D / SP)` channels, each with SP=32 values
- Tensor shape: `[1, T, 1, SP]`
- The last channel is zero-padded if D is not a multiple of 32

The `H` dimension is typically 1 for vector/matmul operations. `N` and `C` can vary freely.

### Writing Data to IOSurfaces

IOSurfaces are locked for CPU access, written in FP16, then unlocked before ANE dispatch. The ANE reads directly from the IOSurface's backing memory — there is no intermediate copy. For strided writes (logical channels mapped to SP-strided layout), each channel occupies SP×2 bytes in the surface regardless of how many values are logically meaningful.

---

## Operations on Runtime Tensors

"Runtime tensors" refers to IOSurface inputs — data that varies per eval, as opposed to constant weights baked into the compiled MIL program. We tested each MIL operation systematically to determine which ones correctly read from runtime IOSurfaces ([`test_mil_variants.cpp`](tests/test_mil_variants.cpp), [`test_dynamic_conv.cpp`](tests/test_dynamic_conv.cpp), [`test_dynamic4.cpp`](tests/test_dynamic4.cpp), [`test_dynamic10.cpp`](tests/test_dynamic10.cpp)).

| Operation | Status | Observation |
|-----------|--------|-------------|
| `add(a, b)` | ✅ Works | Same-shape and N-broadcast both produced correct results in all tested configurations |
| `mul(a, b)` | ✅ Works (same-shape) | Elementwise multiply produced correct output for all tested sizes up to 2560×2560 ([`test_matvec_final.cpp`](tests/test_matvec_final.cpp)) |
| `mul(a, b)` | ⚠️ Unreliable (N-broadcast) | Produced correct results in isolation, but returned incorrect output when composed with other ops in multi-operation MIL programs ([`test_dynamic10.cpp`](tests/test_dynamic10.cpp)) |
| `reduce_sum(axis)` | ✅ Works | Tested on axis=1 (C dimension). Correctly summed across channels and matched CPU reference |
| `reshape` | ✅ Works | Even changing the N dimension succeeded — this contradicted our initial hypothesis that N was immutable for runtime tensors ([`test_dynamic10.cpp`](tests/test_dynamic10.cpp)) |
| `tile` | ☠️ Poisons ANE | Compiles and may eval correctly on first call, but corrupts global ANE state. See [The `tile` Poison](#the-tile-poison) |
| `conv` (dynamic weights) | ❌ Silent failure | Compiles but weight data from IOSurface is silently ignored at eval. See [Why Conv Fails](#why-conv-with-dynamic-weights-fails) |
| `slice_by_size` | ✅ Works | Used in fused multi-output projections (Q/K/V split) |
| `silu` / `sigmoid` | ✅ Works | Activation functions work as constant-free unary ops |

### Broadcasting Behavior

We tested several broadcasting patterns across operations:

- **N-axis broadcast** (`[1, C, H, W]` op `[N, C, H, W]`): Produced correct results for `add`. For `mul`, results were correct in single-op programs but unreliable when composed with other operations ([`test_dynamic10.cpp`](tests/test_dynamic10.cpp), [`test_dynamic12.cpp`](tests/test_dynamic12.cpp)).
- **C-axis broadcast**: Not tested extensively. We avoided it in production code.
- **Same-shape**: Correct in every case we tested. When an operation can be expressed either way, same-shape inputs are the safer choice.

---

## The `tile` Poison

**Hypothesis:** Failures observed in multi-operation MIL programs with `mul` might be caused by a specific operation rather than by broadcasting itself.

**Experiment:** We systematically isolated which operation caused the failures by testing combinations of ops in separate programs and evaluating them in sequence ([`test_dynamic13.cpp`](tests/test_dynamic13.cpp), [`test_dynamic14.cpp`](tests/test_dynamic14.cpp), [`test_dynamic16.cpp`](tests/test_dynamic16.cpp)).

**Observation:** The `tile` MIL operation corrupts global ANE state. The corruption persists for the lifetime of the process.

### Reproduction Sequence

```
Step 1: Eval program A (mul + reduce_sum) → ✅ correct output
Step 2: Eval program B (contains tile)     → ✅ may produce correct output
Step 3: Eval program A again               → ❌ status 0x1d, incorrect output
Step 4: Compile and eval fresh program C   → ❌ status 0x1d
Step 5: All subsequent evals fail          → ❌ until process exit
```

We verified this was specific to `tile` by testing every combination ([`test_dynamic16.cpp`](tests/test_dynamic16.cpp)):
- `tile` + `add` → subsequent evals poisoned
- `tile` + `mul` → subsequent evals poisoned
- `tile` alone (output unused by another op) → subsequent evals poisoned
- Programs without `tile`, in any order → no poisoning observed

Previously-working kernels that contain no `tile` operation begin failing after a `tile`-containing program is evaluated. The only recovery we found is process exit.

### Workaround

We perform all tiling on the CPU before writing to IOSurfaces. This is what the [dynamic matvec strategy](#dynamic-weight-matrix-vector-multiply) does — `memcpy` repeats the input vector into the IOSurface rather than using a `tile` op in the MIL program.

---

## Dynamic-Weight Matrix-Vector Multiply

**Problem:** Compute `y = W @ x` where the weight matrix `W` is a runtime input (not compiled into the kernel), enabling weight-swapping without recompilation.

**Why it's hard:** The obvious approach — `conv` with a runtime weight input — compiles but silently ignores the weight data at eval time (see [Why Conv Fails](#why-conv-with-dynamic-weights-fails)). We needed an alternative path that uses only operations proven to work with runtime IOSurface inputs.

### Working Strategy: CPU-Side Tiling

After eliminating `conv` (hardware constraint) and `tile` (global state corruption), we arrived at a strategy using elementwise `mul` + `reduce_sum` with CPU-side input preparation:

```
x:       [1, T, 1, SP]              — input vector (T = ceil(in_dim/SP))
W:       [out_dim, T, 1, SP]        — weight matrix (one row per N slice)

Step 1 (CPU): Tile x into x_tiled [out_dim, T, 1, SP]
              memcpy the same T×SP values into each of the out_dim N slices

Step 2 (ANE): mul(x_tiled, W)       → [out_dim, T, 1, SP]
              Same-shape multiply — no broadcasting needed

Step 3 (ANE): reduce_sum(axis=1)    → [out_dim, 1, 1, SP]
              Sum across T chunks (partial dot products)

Step 4 (CPU): Sum the 32 (SP) values per output channel → final y[out_dim]
```

Both ANE inputs have identical shapes, avoiding N-broadcast `mul` (which we observed to be unreliable in multi-op programs). The CPU-side tiling is a `memcpy` — fast and simple.

### Measured Performance (2560×2560 — Qwen3-4B hidden size)

From [`test_matvec_final.cpp`](tests/test_matvec_final.cpp):

| Component | Time |
|-----------|------|
| W memcpy to IOSurface | 0.27ms |
| x tile + write to IOSurface | 0.39ms |
| ANE eval (mul + reduce_sum) | 1.00ms |
| **Total dynamic matvec** | **1.65ms** |
| Const-weight conv (baseline) | 0.32ms |
| Const-weight compile time | ~60ms |

Dynamic matvec is ~5× slower per eval than const-weight conv, but incurs zero recompile cost on weight swap (vs ~60ms per weight set for const-weight). The break-even point is approximately 1 weight swap per 5 evals.

### Correctness Verification

We tested against a CPU FP16 reference implementation at sizes from 8×32 up to 2560×2560 ([`test_matvec_final.cpp`](tests/test_matvec_final.cpp)). Maximum absolute error was within FP16 accumulation tolerance (~1e-3 for large dimensions).

### Potential Applications

- **LoRA adapter hot-swapping** — switch adapters without 60ms × N_layers recompile
- **Mixture-of-experts routing** — load selected expert weights at runtime
- **Speculative decoding** — swap between draft and target model weights
- **Any scenario where compile time dominates** — dynamic matvec trades per-eval latency for instant weight changes

---

## Why Conv with Dynamic Weights Fails

**Hypothesis:** ANE's `conv` unit reads weights through a separate hardware path from runtime IOSurface data.

**Experiment:** We declared a `conv` in MIL with a non-constant (runtime) weight input, compiled it, wrote weight data to the input IOSurface, and compared the eval output against a CPU reference ([`test_ane_matmul.cpp`](tests/test_ane_matmul.cpp), [`test_dynamic_conv.cpp`](tests/test_dynamic_conv.cpp)).

**Observation:** The MIL compiler accepted the program. ANE compilation succeeded. At eval time, the weight IOSurface data was written but never read by the conv unit. Output did not match the CPU reference and appeared to contain stale or zero data from the weight bus. No error code was returned.

We interpret this as two separate data paths in the conv hardware:
1. **Activation path** — reads from IOSurface inputs (runtime data)
2. **Weight path** — reads from a dedicated internal bus populated at compile time from BLOBFILE data

When the weight input is declared as runtime rather than constant, the compiler does not reject it, but the hardware still reads from the weight bus at eval time. This appears to be a fundamental hardware architecture constraint rather than a software limitation.

---

## Compile Budget and Caching

### Kernel Limits

**Experiment:** We compiled increasing numbers of ANE kernels in a single process to find the limit, then tested whether freeing kernels reclaimed budget ([`test_ane_limits.cpp`](tests/test_ane_limits.cpp), [`test_ane_limits2.cpp`](tests/test_ane_limits2.cpp), [`test_ane_limits3.cpp`](tests/test_ane_limits3.cpp)).

**Observations:**
- A per-process limit on simultaneously loaded ANE kernels exists. The exact number varies with kernel size (larger kernels consume more budget).
- Freeing kernels via `ane_free` reclaims budget. A compile → eval → free loop allows models that exceed the simultaneous limit to run.
- Qwen3-4B loads 226 kernels (216 layer + 10 LM head chunks) and fits within budget on tested hardware.

### Persistent Cache

**Experiment:** We measured compile time on first run vs subsequent runs for the same MIL programs ([`test_ane_limits4.cpp`](tests/test_ane_limits4.cpp)).

**Observations:**
- The ANE framework maintains a persistent on-disk compile cache, keyed by MIL program text + weight data hash.
- Cache load is approximately 10× faster than fresh compilation.
- Qwen3-4B: ~28s first run → ~8s cached init (226 kernels).
- The cache survives process restarts. Different models produce different cache keys (different weight data), so they do not collide.

### Measured Compile Times

| Kernel Type | Compile Time | Notes |
|-------------|-------------|-------|
| Single matmul (2560×2560) | ~60ms | One weight matrix |
| Fused QKV (3 matrices) | ~100ms | Q+K+V in one kernel |
| Fused SwiGLU FFN | ~150ms | gate+up+silu+mul+down |
| Chunked FFN (per chunk) | ~80ms | Partial intermediate dim |
| Cache load (any) | ~5–10ms | ~10× faster than fresh compile |

---

## Dispatch Overhead

**Problem:** Qwen3-4B generates at ~6.2 tok/s on ANE. We measured where the time goes to understand the bottleneck.

### Measured Breakdown

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

**Key observation:** The actual ANE compute time for a 2560×2560 matmul at ~15 TOPS FP16 is approximately 0.0004ms. This is less than 0.1% of the measured dispatch time. The remaining 99.9% is CPU↔ANE coordination overhead (IOSurface locking, data format conversion, dispatch submission, completion waiting).

### Implications for Optimization

These measurements suggest the following priorities for improving throughput:

- **Fusing more ops into fewer kernels** has the highest leverage — reducing dispatches/layer from 6 to 2–3 would nearly double throughput
- **Moving attention to ANE** would eliminate the largest source of mid-layer CPU work and its associated dispatch round-trips
- **Batch prefill** (multiple tokens per dispatch) would amortize dispatch cost over more useful work
- **Quantization** (INT8/INT4) would reduce memory transfer somewhat but would not reduce dispatch count, so the benefit is limited by the overhead-dominated profile

---

## MIL Programming Reference

ANE-LM compiles MIL (Machine Learning Intermediate Language) programs via the private `Espresso` and `ANECompiler` frameworks. Below are patterns we verified to work through testing.

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

- The `buildInfo` dict with `coremlc-version` is required — without it, the MIL parser fails. We used `"3505.4.1"` across macOS 13–15; the exact version string did not appear to matter.
- `ios16` deployment target worked for all patterns described here. Higher targets may enable additional operations but were not tested.
- All input/output tensors must be FP16 with W ≥ SP (32).

### Fused Multi-Output Projections

We compute Q, K, V projections in a single kernel by concatenating weight matrices and slicing the output:

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

This reduces 3 ANE dispatches to 1.

### Fused SwiGLU FFN

The entire FFN block compiled and evaluated correctly as a single kernel:

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

When `intermediate_size` exceeds what a single kernel can hold (ANE has finite internal buffer space), we split along the intermediate dimension:

```
Chunk 0: gate_w[0:chunk], up_w[0:chunk] → partial down_proj → accum[0]
Chunk 1: gate_w[chunk:2*chunk], up_w[chunk:2*chunk] → partial down_proj → accum[1]
...
Final: sum(accum[0..N]) on CPU
```

Each chunk is a separate ANE kernel. For Qwen3-4B (intermediate_size=9728), we used 4 chunks of 2432 each.

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

Where N=out_dim, T=ceil(in_dim/SP). Both inputs have the same shape — no broadcasting.

---

## Chunked FFN for Large Models

Models with large intermediate dimensions (e.g. Qwen3-4B with intermediate_size=9728) exceeded what we could compile into a single ANE kernel. We implemented an automatic chunking strategy:

1. **Auto-detect**: `ane_ffn_chunk_count(dim, inter)` determines whether chunking is needed and how many chunks to use
2. **Compile**: Each chunk handles `inter/N` of the intermediate dimension. Gate and up projections are sliced; the down projection produces a partial output
3. **Eval**: Chunks are evaluated sequentially, with partial outputs accumulated on CPU
4. **Fallback**: If single-kernel compilation fails (returns null), the system automatically retries with chunking

This is what enabled Qwen3-4B inference on ANE — the original ANE-LM codebase only supported single-kernel FFN and could not handle intermediate_size values above ~4096.

---

## Methodology

These findings come from [25 targeted test programs](tests/), each designed to isolate a single hypothesis about ANE behavior. Every test includes CPU reference implementations for correctness verification and clear pass/fail criteria. The full progression is documented in [`tests/README.md`](tests/README.md).

The investigation followed this arc:

1. **Can conv accept dynamic weights?** → Compiled but weights were silently ignored at eval. We concluded the hardware reads weights from a dedicated bus, not from IOSurface inputs. ([`test_ane_matmul.cpp`](tests/test_ane_matmul.cpp))
2. **Which MIL ops work with runtime inputs?** → `add`, `mul`, `reduce_sum`, `reshape` produced correct results. `conv` weights and `tile` did not. ([`test_mil_variants.cpp`](tests/test_mil_variants.cpp), [`test_dynamic4.cpp`](tests/test_dynamic4.cpp))
3. **Why do some inputs fail silently?** → We traced this to the IOSurface W dimension: all runtime inputs require W ≥ 32 (SP). ([`test_dynamic6.cpp`](tests/test_dynamic6.cpp))
4. **Can mul + reduce_sum implement matvec?** → Yes, with same-shape inputs. ([`test_dynamic7.cpp`](tests/test_dynamic7.cpp))
5. **Why is mul unreliable with N-broadcasting?** → The `tile` operation was corrupting global ANE state. After removing `tile`, N-broadcast `mul` failures disappeared in our single-op tests, but we chose same-shape inputs for production reliability. ([`test_dynamic16.cpp`](tests/test_dynamic16.cpp))
6. **Working solution?** → CPU-side tiling (`memcpy`) + same-shape `mul` + `reduce_sum`. Measured 1.65ms at 2560×2560, verified correct at all tested sizes. ([`test_matvec_final.cpp`](tests/test_matvec_final.cpp))

### Human-AI Collaboration

This research was conducted as a collaboration between a human researcher and Claude Opus 4.6 (Anthropic), working through [pi](https://github.com/mariozechner/pi-coding-agent), a terminal-based coding agent. pi gives Claude direct access to the terminal — it writes files, compiles, runs tests, and reads output without any human copy-pasting in the loop. The workflow operated as a tight cycle:

1. **Human** steered the investigation — identified the next question, provided domain intuition, and course-corrected when results were ambiguous
2. **Claude** wrote the C++ test program, compiled it, ran it, read the output, analyzed the results, updated the working model of ANE behavior, and proposed the next hypothesis
3. Repeat

A key challenge was managing the tension between context window size and model output quality. LLMs produce better code and reasoning when they have full context of prior experiments, but the context window is finite and quality degrades as it fills. The human researcher managed this actively — deciding when to start fresh sessions, which prior findings to summarize into the prompt vs. leave as file references, and structuring each test program to be self-contained with clear pass/fail output so Claude could read a result and reason about it without needing the full history re-explained. The 25 test files served as external memory that both participants could reference, keeping the investigation coherent across session boundaries.

Neither participant could have done this alone efficiently. The human had the intuition for which questions mattered, could recognize when results pointed toward a hardware constraint vs. a software bug, and kept the context budget under control. Claude could rapidly generate correct C++ against an undocumented API, execute the full write-compile-run-analyze cycle autonomously, and reason about what each result implied for the hardware model.

The entire investigation — from "can conv accept dynamic weights?" to a working 2560×2560 dynamic matvec — took a single evening. Dead ends (tile, N-broadcast mul, dynamic conv) were identified and discarded within 1–2 test iterations rather than hours of manual debugging.

---

## Related Projects

- **[ANE-LM](https://github.com/skyfallsin/ANE-LM)** — LLM inference on ANE using these findings. Runs Qwen3-4B at ~6.2 tok/s on the Neural Engine.
- **[johnmai-dev/ANE-LM](https://github.com/johnmai-dev/ANE-LM)** — Original ANE inference project (Qwen3/3.5 support, ANE runtime, safetensors loader)
- **[maderix/ANE](https://github.com/maderix/ANE)** — Neural network training on ANE via reverse-engineered APIs

## License

MIT
