# ANE Test Programs

These are the 25 test programs used to reverse-engineer Apple Neural Engine behavior. Each isolates a specific hypothesis with clear pass/fail criteria and CPU reference implementations for correctness verification.

They depend on the ANE runtime from [ANE-LM](https://github.com/skyfallsin/ANE-LM) (`core/ane_runtime.h`).

## Test Progression

The tests tell the story of the investigation — each one builds on findings from the previous.

### Phase 1: ANE Limits & Baseline (tests 1-4)

| File | Question | Finding |
|------|----------|---------|
| `test_ane_limits.cpp` | How many kernels can ANE hold? What's the max kernel size? | Per-process kernel limit exists, varies with kernel size |
| `test_ane_limits2.cpp` | How does compile limit scale with kernel size? | Larger kernels = fewer simultaneous loads |
| `test_ane_limits3.cpp` | Does freeing kernels reclaim compile budget? | Yes — compile → eval → free → compile works |
| `test_ane_limits4.cpp` | How fast is cache load vs compile vs eval? | Cache load ~10× faster than compile |

### Phase 2: Can Conv Accept Dynamic Weights? (test 5)

| File | Question | Finding |
|------|----------|---------|
| `test_ane_matmul.cpp` | Can we pass weight matrices as runtime inputs to conv? | **No** — compiles but weights are silently ignored at eval. Conv reads from dedicated hardware weight bus. |

### Phase 3: What MIL Ops Work with Runtime Inputs? (tests 6-7)

| File | Question | Finding |
|------|----------|---------|
| `test_mil_variants.cpp` | Systematic sweep: which MIL ops accept runtime IOSurface inputs? | add, mul, reduce_sum work; conv weights don't |
| `test_dynamic_conv.cpp` | Dedicated dynamic conv testing — is it really impossible? | Confirmed: conv dynamic weights are a hardware limitation |

### Phase 4: First Dynamic Matvec Attempts (tests 8-11)

| File | Question | Finding |
|------|----------|---------|
| `test_dynamic.cpp` | Can mul+reduce_sum implement matvec? | Promising but shape issues |
| `test_dynamic2.cpp` | Which MIL formulations does ANE accept for multi-input kernels? | Many compile, fewer eval correctly |
| `test_dynamic3.cpp` | Correctness and performance of dynamic conv approach | Conv approach dead, need alternative |
| `test_dynamic4.cpp` | Do multi-input programs actually eval correctly at all? | Yes, add/mul with 2 IOSurface inputs work |

### Phase 5: The SP=32 Discovery (tests 12-14)

| File | Question | Finding |
|------|----------|---------|
| `test_dynamic5.cpp` | mul+reduce_sum as conv replacement — does it work? | Works for same-shape inputs; broadcast on W=1 fails |
| `test_dynamic6.cpp` | **Why do some inputs fail?** Hypothesis: W must equal SP=32 | **Confirmed: all runtime IOSurfaces need W ≥ 32 (SP)** |
| `test_dynamic7.cpp` | Matvec using only W=SP tensors | First working dynamic matvec! |

### Phase 6: Shape & Broadcast Investigation (tests 15-17)

| File | Question | Finding |
|------|----------|---------|
| `test_dynamic8.cpp` | What broadcasting patterns work? C-broadcast? Min surface size? | N-broadcast works for add; C-broadcast untested |
| `test_dynamic9.cpp` | Production-quality dynamic matvec with proper shapes | Working at multiple sizes |
| `test_dynamic10.cpp` | Systematic: reshape on runtime tensors, N-broadcast for mul | Reshape works (even changing N!), but mul N-broadcast unreliable in complex programs |

### Phase 7: Debugging Mul Reliability (tests 18-22)

| File | Question | Finding |
|------|----------|---------|
| `test_dynamic11.cpp` | Working around mul's N-broadcast limitation | CPU-side tiling avoids broadcast entirely |
| `test_dynamic12.cpp` | Minimal mul reproduction — which exact configs fail? | Isolated to multi-op programs with N-broadcast |
| `test_dynamic13.cpp` | Which operation combination causes failure? | tile op involved in all failures |
| `test_dynamic14.cpp` | Is it the eval API or the MIL program? | The MIL program — specifically tile |
| `test_dynamic15.cpp` | Is C=2 the problem? Minimal test | No — C dimension is fine |

### Phase 8: The Tile Poison (test 23)

| File | Question | Finding |
|------|----------|---------|
| `test_dynamic16.cpp` | **Why does tile+mul fail?** | **tile poisons global ANE state.** Any program using tile corrupts all subsequent evals in the process. Status 0x1d. Only recovery is process restart. |

### Phase 9: Working Solution (tests 24-25)

| File | Question | Finding |
|------|----------|---------|
| `test_minimal.cpp` | Minimal working mul+reduce_sum (sanity check post-poison-discovery) | Confirmed: without tile, everything works |
| `test_matvec_final.cpp` | **Production dynamic matvec with CPU-side tiling** | **1.65ms at 2560×2560. Correct at all sizes up to 2560×2560. 5× slower than const-weight conv but zero recompile cost.** |
