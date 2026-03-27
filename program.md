# ANE autoresearch

Autonomous experimentation to maximize LLM inference tok/s on Apple Neural Engine.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — same loop (modify → measure → keep/discard), adapted for ANE hardware reverse-engineering and inference optimization.

Two repos are involved:
- **This repo** (`apple-neural-engine-field-guide`) — where new ANE capabilities are discovered via standalone test programs.
- **[ANE-LM](../ANE-LM/)** — where proven findings are applied to the inference engine to improve tok/s.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar27`). The branch `autoresearch/<tag>` must not already exist in either repo.
2. **Create the branch** in both repos:
   ```bash
   cd ../apple-neural-engine-field-guide && git checkout -b autoresearch/<tag>
   cd ../ANE-LM && git checkout -b autoresearch/<tag>
   ```
3. **Read for full context**:
   - This file (`program.md`) — experiment instructions and constraints.
   - `tests/README.md` — the 25 existing test programs and what they found.
   - `README.md` — full field guide with all known ANE constraints.
   - `../ANE-LM/models/llm/qwen3.cpp` — the model forward pass (primary optimization target).
   - `../ANE-LM/core/ane_runtime.h` — ANE kernel compilation and eval API.
   - `../ANE-LM/core/ane_runtime.cpp` — MIL program generation, IOSurface management, dispatch.
   - `../ANE-LM/core/cpu_ops.h` / `cpu_ops.cpp` — CPU operations (RMSNorm, RoPE, attention).
4. **Verify the model exists**: Check that `/tmp/Qwen3-4B/` contains safetensors + config.json.
5. **Build ANE-LM**: `cd ../ANE-LM && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(sysctl -n hw.ncpu)`
6. **Initialize results.tsv** in this repo with the header row. Run baseline to record starting tok/s.
7. **Confirm and go**.

## The metric

**Maximize generation tok/s.** Currently ~6.2 tok/s for Qwen3-4B. This is the tokens-per-second reported during autoregressive decoding.

Run the benchmark:

```bash
cd ../ANE-LM
./build/ane-lm generate --model /tmp/Qwen3-4B --prompt "Explain the theory of general relativity in detail, covering spacetime curvature, the equivalence principle, gravitational time dilation, and experimental confirmations" --max-tokens 200 2>&1 | tail -1
```

This prints:
```
Generation: 200 tokens, 6.234 tokens-per-sec
```

**Always generate 200 tokens** for consistency. Run 2-3 times when results are close (within ~0.3 tok/s) and take the median.

## Two-phase workflow

### Phase 1: Discover (this repo)

Write standalone test programs in `tests/` to probe untested ANE capabilities. Each test is a self-contained C++ program that:
1. Compiles a MIL program with the operation under test
2. Evaluates it with known inputs
3. Compares output against a CPU reference implementation
4. Reports pass/fail with timing data

Build and run a test:
```bash
cd ../ANE-LM
# Add the test file to CMakeLists.txt as a new executable, or compile directly:
clang++ -std=c++17 -O2 \
  -I../ANE-LM -I../ANE-LM/include \
  ../apple-neural-engine-field-guide/tests/test_new_op.cpp \
  ../ANE-LM/core/ane_runtime.cpp \
  ../ANE-LM/utils.cpp \
  -framework Foundation -framework IOSurface -framework Accelerate -lobjc \
  -o /tmp/test_new_op && /tmp/test_new_op
```

When a test discovers a working operation, document it in `tests/README.md` and commit to the field guide branch.

### Phase 2: Apply (ANE-LM repo)

Take proven findings from Phase 1 and integrate them into the inference engine:
- Modify `models/llm/qwen3.cpp` — forward pass, dispatch strategy
- Modify `core/ane_runtime.cpp/.h` — new fused kernel types, optimized IOSurface paths
- Modify `core/cpu_ops.cpp/.h` — vectorize with Accelerate/vDSP

After each change:
1. Build: `cmake --build build -j$(sysctl -n hw.ncpu)`
2. Correctness check: `./build/ane-lm generate --model /tmp/Qwen3-4B --prompt "What is 2+2?" --max-tokens 20 2>&1`
3. Benchmark: run the tok/s measurement (see above)
4. Keep or discard based on results

## ANE hardware constraints (hard rules)

These are proven constraints. Violating them produces silent failures or corrupted state.

- **IOSurface W ≥ 32 (SP)**: All runtime input IOSurfaces must have innermost dimension ≥ 32. W=16 or W=1 compiles but produces garbage.
- **Never use `tile` in MIL programs**: Corrupts global ANE state. All subsequent evals fail with status 0x1d until process exit. Tile on CPU via memcpy instead.
- **Conv cannot read dynamic weights**: Conv reads from a dedicated hardware weight bus, not IOSurface inputs. Dynamic weight input compiles but is silently ignored.
- **N-broadcast `mul` is unreliable**: Correct in isolation, incorrect in multi-op MIL programs. Use same-shape inputs.

### Known working operations on runtime tensors

| Operation | Status |
|-----------|--------|
| `add(a, b)` — same-shape or N-broadcast | ✅ |
| `mul(a, b)` — same-shape only | ✅ |
| `reduce_sum(axis)` | ✅ |
| `reshape` | ✅ (even changing N) |
| `slice_by_size` | ✅ |
| `silu`, `sigmoid` | ✅ |
| `conv` with const weights | ✅ |

## Where the time goes (current bottleneck)

```
Qwen3-4B: ~6.2 tok/s, ~161ms per token
216+ ANE dispatches per token
~0.75ms per dispatch average
ANE compute per 2560×2560 matmul: ~0.0004ms (< 0.1% of dispatch time)
99.9% of time is CPU↔ANE coordination overhead
```

Per-layer dispatches:
1. Fused QKV projection (ANE) — Q+K+V in one conv kernel
2. QK-norm + RoPE (CPU)
3. GQA attention + KV cache (CPU)
4. O projection (ANE)
5. Post-attention RMSNorm (CPU)
6. SwiGLU FFN (ANE) — 4 chunked dispatches for Qwen3-4B

## Experiment ideas (ordered by expected impact)

### High impact — discover new ANE capabilities (Phase 1)

1. **Test `softmax` on runtime tensors**: If softmax works, attention (Q·K softmax, attn·V) could move to ANE, eliminating the biggest CPU bottleneck per layer. Write a test: compile MIL with `softmax(x)` where x is runtime IOSurface, compare against CPU softmax.

2. **Test `rsqrt` on runtime tensors**: Needed for fusing RMSNorm into ANE. RMSNorm is `x * rsqrt(mean(x²) + eps) * weight`. If `rsqrt` works on runtime tensors, RMSNorm becomes: `reduce_sum` (for mean of squares) → `rsqrt` → `mul` (scale). All proven ops except `rsqrt`.

3. **Test `matmul` MIL op (distinct from conv)**: MIL has a `matmul` op separate from `conv`. Unknown whether it uses the same hardware weight bus. If `matmul` can read runtime inputs, it would be faster than the `mul` + `reduce_sum` workaround for dynamic matvec.

4. **Test `exp`, `sub`, `div` on runtime tensors**: Building blocks for softmax (`exp(x - max(x)) / sum(exp(x - max(x)))`). Even if native `softmax` doesn't work, these components might.

5. **Test `reduce_max` on runtime tensors**: Needed for numerically stable softmax.

6. **Test `concat` on runtime tensors**: Could avoid separate dispatches for combining outputs.

### High impact — reduce dispatch count (Phase 2)

7. **Fuse O-projection with FFN**: Currently separate dispatches. If both are const-weight conv, they might chain in one MIL program with the post-attention norm expressed as MIL ops (if rsqrt works).

8. **Fuse RMSNorm into ANE kernels**: Eliminate CPU round-trips for normalization. Requires rsqrt discovery from Phase 1.

9. **Reduce FFN chunks from 4 to 2-3**: Try different intermediate dimension splits. Each eliminated chunk saves 36 dispatches.

### Medium impact — faster CPU operations (Phase 2)

10. **Vectorize RMSNorm with vDSP**: Replace scalar loop with `vDSP_svesq` + `vDSP_meanv` + `vrsqrtf` + `vDSP_vmul`.

11. **Vectorize RoPE with Accelerate**: Precomputed cos/sin tables exist (`rope_cos_`, `rope_sin_`). Use `vDSP_vmul` + `vDSP_vsub` + `vDSP_vadd` for the rotation.

12. **Optimize GQA attention with cblas**: Replace scalar Q·K and attn·V with `cblas_sgemv`. Replace scalar softmax with vDSP-based softmax.

13. **Reduce memory copies in forward pass**: Eliminate intermediate buffers where possible.

## Logging results

Log to `results.tsv` in this repo (tab-separated, do NOT commit it):

```
commit	repo	tok_s	status	description
```

1. git commit hash (short, 7 chars)
2. repo: `field-guide` or `ane-lm`
3. tok/s achieved (0.0 for crashes, N/A for Phase 1 discoveries)
4. status: `keep`, `discard`, `crash`, or `discovery`
5. short description

Example:

```
commit	repo	tok_s	status	description
a1b2c3d	ane-lm	6.234	keep	baseline
b2c3d4e	field-guide	N/A	discovery	softmax works on runtime tensors (tested in test_softmax.cpp)
c3d4e5f	ane-lm	7.891	keep	move attention to ANE using softmax discovery
d4e5f6g	field-guide	N/A	discovery	rsqrt fails on runtime tensors (status 0x1d)
e5f6g7h	ane-lm	6.512	keep	vectorize rmsnorm with vDSP
f6g7h8i	ane-lm	6.100	discard	fuse rope into qkv (slower due to dynamic input overhead)
```

## The experiment loop

LOOP FOREVER:

1. **Pick the highest-impact experiment** you haven't tried yet.
2. If it requires testing an untested ANE op → **Phase 1** (field guide repo):
   a. Write the test program in `tests/`.
   b. Build and run it.
   c. If the op works: document in `tests/README.md`, commit to field guide branch, log as `discovery`.
   d. If the op fails or poisons: note the finding, commit the test anyway (negative results are valuable), log as `crash` or `discard`.
   e. If the discovery enables a new fusion → proceed to Phase 2.
3. If it's an optimization using known-working ops → **Phase 2** (ANE-LM repo):
   a. Make the code change in ANE-LM. Commit.
   b. Build: `cmake --build build -j$(sysctl -n hw.ncpu) 2>&1 | tail -5`
   c. Correctness check: verify output isn't garbage.
   d. Benchmark: measure tok/s.
   e. If improved → keep. If not → `git reset --hard HEAD~1`.
   f. Log to results.tsv.
4. Go to 1.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. If you run out of high-impact ideas, move to medium. If those are exhausted, try more exploratory ANE op tests. Every new discovery opens new optimization paths. The loop runs until the human interrupts you.
