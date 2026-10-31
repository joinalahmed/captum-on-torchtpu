# Explainability as a TorchTPU performance lens: Captum on Cloud TPUs, and what it taught us about TorchTPU

*An entry for the #TorchTPUSprint. Everything below was measured on a Cloud TPU v6e-1 in us-east5-b, with torch 2.15.0, torch_tpu 0.2.0, Captum 0.9.0 and transformers 5.22.1 unless stated otherwise. Code, scripts and raw findings: see "Where everything lives" at the end. TODO before publishing: push this project folder to a public repository and link it here.*

---

## TL;DR

- **Captum runs unmodified on TorchTPU.** Every public `captum.attr` method, including the `LLMAttribution` wrappers, works on `torch.device("tpu")`. 27 deterministic methods match CPU to 1e-7 relative error; 11 stochastic ones run and differ only by RNG.
- **One setting matters:** `torch.set_float32_matmul_precision("highest")`. Without it, attributions are off from CPU by about 1e-3 (1e-2 for LayerGradCam). With it, they agree to 1e-7 on small models and 1e-5 on Qwen3-0.6B.
- **Attribution is a sharp stress test for execution modes.** On Qwen3-0.6B, Layer Integrated Gradients with 64 steps takes 0.80 s in strict eager, 0.36 s with DeferAndFuse (one environment variable), and 0.16 s under `torch.compile(backend="tpu")`. CPU takes 4 to 17 s depending on steps.
- **Explainability found real TorchTPU bugs and gaps.** A scalar-buffer leak, a missing `max_unpool2d` lowering, a compiled-mode bug reading CPU view arguments past their buffer, a Captum-side device bug, eight unregistered ops, and 11 stale entries on the bounded-dynamism deny list. All of it is fixed on branches; five PRs are open upstream and five more are pushed and waiting on a GitHub permission hiccup.

---

## 1. Why explainability, and why it is a good lens

The sprint asked for content about TorchTPU's execution modes, XProf and precision. We chose explainability as the vehicle for a practical reason: attribution methods are a compact, repeatable, autograd-heavy workload that most ML teams actually run.

Integrated Gradients interpolates between a baseline and the input in `n_steps` steps, runs forward and backward on every interpolated point, and integrates the gradients. That means:

- many forward and backward passes of the same model with the same shapes (good for a compiler),
- host-side Python scalars in the loop (step sizes, scaling factors) that flow into device ops (a stress test for eager dispatch and caching),
- forward hooks and `torch.autograd.grad` on intermediate layers (a stress test for the autograd integration),
- differences of gradients as the final output (a stress test for numerical precision, because differences amplify relative error).

Also, nobody had done it. Our prior-art research (a report in `reports/`) found that Captum, SHAP, Inseq, TransformerLens, nnsight and pyvene have no TPU support status, no TPU CI, and no TPU issues or PRs. Captum has around 355k downloads per month and CPU and GPU CI only. So the first question was simple: does it just work?

---

## 2. Does it run? Yes, unmodified

We started with a smoke test (`scripts/tpu_smoke.py`) of the primitives attribution depends on, with tiny models so failures would be about the runtime and not about memory:

| Primitive | TPU result |
|---|---|
| `loss.backward()` | pass |
| `torch.autograd.grad` with respect to inputs | pass |
| forward hooks | pass |
| full backward hooks | pass |
| double backward (`create_graph=True`) | pass |
| custom `autograd.Function` | pass |
| `torch.func.grad` | pass |
| `torch.compile(backend="tpu")` plus backward on a view input | pass |
| Captum `IntegratedGradients` | pass |
| Captum `LayerIntegratedGradients` | pass |

12 of 12. `torch.accelerator.current_accelerator()` returns `tpu`, `is_available()` is true, `device_count()` is 1. TorchTPU registers as a first-class accelerator, and Captum never has to know what device it is on.

Then the full method matrix (`scripts/captum_tpu_matrix.py`) over an MLP and a small CNN, every `captum.attr` class, CPU versus TPU with matmul precision "highest":

- **27 OK**: IG, Saliency, InputXGradient, DeepLift, GuidedBackprop, Deconvolution, GuidedGradCam, FeatureAblation, Occlusion, ShapleyValues, LRP, and every Layer* and Neuron* variant including LayerGradCam, LayerLRP, LayerFeatureAblation and NeuronConductance. All within 1e-7 to 1e-8 relative error of CPU.
- **11 RUNS**: GradientShap, ShapleyValueSampling, NoiseTunnel and the *Shap family. They run; they differ from CPU because they sample random noise and the RNG streams differ across devices. Expected.
- **0 DIFF, 0 FAIL.** (The first pass failed Lime and KernelShap for a missing `sklearn` install. Not a TPU problem.)

Finally, the end-to-end LLM path (`scripts/demo_captum_qwen_tpu.py`): `AutoModelForCausalLM.from_pretrained(..., device_map="tpu")` loads Qwen3-0.6B straight onto the device in 2.9 s. `LayerIntegratedGradients` over the embedding layer, `LLMAttribution(FeatureAblation)` and `LLMGradientAttribution(LayerIntegratedGradients)` all work. The HTML heatmap renders.

We also tried the other libraries (`scripts/other_xai_libs_tpu.py`): SHAP's DeepExplainer and GradientExplainer pass, pytorch-grad-cam's GradCAM and ScoreCAM pass, nnsight's trace with `device_map="tpu"` passes. Inseq fails with `ValueError: Unknown device tpu` because of a hard-coded device allow-list (an easy upstream fix). TransformerLens fails on a transformers v5 incompatibility unrelated to TPUs.

---

## 3. Is it right? Yes, with one setting

This is the single most important thing in this post for anyone doing attribution on a TPU.

By default TorchTPU runs float32 matmuls with bfloat16 inputs on the MXU. Model outputs tolerate that fine. Attributions do not, because they are differences of gradients, and differences amplify relative error. Maximum relative error of TPU attribution against CPU attribution, same weights and inputs, small MLP and CNN:

| Method | default precision | `torch.set_float32_matmul_precision("highest")` |
|---|---|---|
| Saliency | 2.2e-3 | 1.7e-7 |
| InputXGradient | 1.7e-3 | 7.0e-8 |
| IntegratedGradients (50 steps) | 2.1e-3 | 4.4e-8 |
| LayerIntegratedGradients | 3.1e-3 | 1.0e-7 |
| LayerGradCam | 1.0e-2 | 2.2e-7 |
| GradientShap | 3.6e-2 | 2.5e-2 (stochastic) |

On a real transformer, Qwen3-0.6B in fp32 with precision "highest", Captum's `LayerIntegratedGradients` CPU versus TPU on the prompt "The capital of France is" with target " Paris" has a maximum relative error of 8.5e-6 and the same top-3 attributed tokens (" France", " of", "The").

The surprising part: **"highest" costs almost nothing for attribution.** On Qwen3-0.6B at 32 and 128 steps, eager warm times were 0.86 s and 1.17 s at "highest" versus 0.88 s and 1.08 s at "high". The workload is not MXU-bound at this size; it is bound by dispatch and host-device traffic, as the profiles in section 5 show. So: set it, always, for attribution.

### 3b. A detour that turned into a correctness lesson

We briefly built our own batched IG (`torchtpu_xai`) on the theory that batching all steps into one fixed-shape call would beat Captum on TPU. Two things happened.

First, Captum already does that. When `internal_batch_size` is None, `IntegratedGradients` expands the input to `n_steps × batch` and runs one forward and backward pass. Our timings were identical to Captum's. The "differentiator" was not one, and we repositioned the project around Captum itself.

Second, our IG disagreed with Captum by 0.71 relative error on Qwen3. That looked like a bug. It was not. Captum defaults to Gauss-Legendre quadrature; ours used right-Riemann. With `method="riemann_right"` and the same steps, the two agreed to the printed digit. The real finding was that **the reference itself had not converged**: Captum on CPU with 256 Gauss-Legendre steps had a convergence delta of 27.3 against a total attribution of 44.4, and the total swung from -26.6 to -8.2 to +44.4 across 32, 128 and 256 steps. Raw-logit target plus pad-token baseline is a poor Integrated Gradients setup for an LLM. The advice that came out of it, and that went into the docs: always request `return_convergence_delta=True` and look at it before trusting an attribution. Not TPU-specific, but TPU made it cheap enough to run the sweep that exposed it.

---

## 4. Is it fast? The execution-mode comparison

TorchTPU has three ways to run a program: strict eager (each op dispatched as it is called), DeferAndFuse (ops are deferred and fused into larger programs, enabled by `TORCH_TPU_DEFER_AND_FUSE=1`), and `torch.compile(backend="tpu")`. Attribution exercises all three in a way a plain forward pass does not.

### Qwen3-0.6B, 53-token prompt, Captum `LayerIntegratedGradients`, precision "high"

| steps | Captum CPU | Captum TPU strict eager | Captum TPU DeferAndFuse | batched IG, `torch.compile` |
|---|---|---|---|---|
| 32 | 4.48 s | 0.880 s | **0.238 s** (first call 26 s) | 0.176 s (first call 68 to 112 s) |
| 128 | 17.04 s | 1.078 s | **0.450 s** (first call 29 s) | OOM (see memory section) |

DeferAndFuse makes *unmodified Captum* 3.7x faster than strict eager and 38x faster than CPU at 128 steps, with no code change. The cost is a 26 to 29 s fusion compile on the first call.

### The shipped example: Qwen3-0.6B, 42-token prompt, 64 steps, precision "highest"

This is the run behind `examples/explainability/captum_attribution.py` in the torch_tpu branch, and the numbers in the docs case study:

| mode | warm median | first call | max rel err vs CPU | peak HBM |
|---|---|---|---|---|
| strict eager | 0.80 s | 5 s | 9.2e-05 | 10.6 GiB |
| DeferAndFuse | 0.36 s | 43 s | 9.3e-05 | 11.1 GiB |
| `torch.compile(backend="tpu")` | 0.16 s | 130 s | 7.0e-05 | 15.3 GiB |

On a tiny MLP the same three modes take 0.065, 0.039 and 0.062 s, with 2e-7 relative error in all of them. The compiled mode's advantage only appears once there is enough work per call to amortize the graph launch.

A few things worth knowing about these numbers:

- **First-call time is real money.** TorchTPU's tier-2 compilation cache is disabled for single-device programs, so every new process pays the compile. The tier-3 cache (`TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=gs://...`) avoids that across processes.
- **The compile row needed two fixes to exist at all.** Out of the box, Captum's LIG wrapped around a compiled HF model fails with "expected tensor to be on tpu, got cpu". Section 6 explains the two bugs behind that, one in Captum and one in torch_tpu.
- **At tiny scale, TPU time is flat in steps.** On a 5-token prompt, TPU took about 0.55 s whether we ran 8, 32 or 64 steps. That is fixed overhead, not compute. The interesting regime starts at tens of tokens and tens of steps.

---

## 5. What XProf shows

We profiled LIG with 32 steps in each mode using `torch.profiler` with `ProfilerActivity.TPU` and `TpuProfilerConfig` (`scripts/profile_ig_modes.py`), then opened the traces in XProf.

| mode | Self CPU total | `aten::_to_copy` share | `_to_copy` calls | avg per copy | Self TPU total |
|---|---|---|---|---|---|
| strict eager | 1.93 s | 1.35 s (70%) | 522 | 2.6 ms | 5.0 s |
| DeferAndFuse | 0.78 s | 0.29 s | 522 | 0.57 ms | 1.23 s |

The story is one line: **in strict eager, 70% of host time is `aten::_to_copy`**, host-to-device and device-to-host copies that block the Python thread. Captum moves scalars and small tensors across the boundary constantly: step sizes, the scaling of inputs along the path, the `cumsum` offsets it uses to slice layer outputs. Each one is a blocking round trip in strict eager. DeferAndFuse does the same number of copies but each one costs a fifth as much, because the copies no longer wait for the preceding ops to finish individually.

That is why the docs case study is titled "Explainability as a performance lens". Attribution is small enough to profile in minutes and pathological enough to make the execution-mode differences obvious in a single table.

### Memory

Peak HBM is set by the logits tensor. Every LLM attribution materializes `[steps, T, V]` logits before selecting the last position and target token. For Qwen3 that is `[128, 53, 151936]` in fp32, 3.84 GB, held alongside the interpolation path with gradients. That is what OOMed our batched IG at 128 steps and everything at 256 steps.

Two fixes, both in user code:

- `model(input_ids, logits_to_keep=1)` (transformers v5) keeps only the last position. With it, 128 steps in one batch took 0.857 s warm on TPU (23.9 GB peak) versus 11.2 s on CPU.
- Captum's `internal_batch_size` chunks the steps. 128 steps with `internal_batch_size=64` ran in 1.42 s at 10.6 GiB peak, versus 23.9 GB unchunked.

The new docs section on `torch.accelerator` memory APIs (`memory_stats`, `max_memory_allocated`, `empty_cache` and friends, branch `docs-memory-apis`) came out of needing these numbers.

---

## 6. What explainability found in TorchTPU

This is the part that turned a tutorial into a contribution sprint. Each item below started as something Captum did that a plain training loop does not.

### 6.1 Eager scalar buffers leaked (issue #6036, PR #6102)

Captum's IG scales inputs by per-step Python floats. We wrote `scripts/scalar_leak_check.py` to check the open issue that eager device buffers for distinct Python scalars were never evicted:

| loop | live allocations after 5000 iterations |
|---|---|
| `x.add_(0.5)` (same scalar) | +2 |
| `x.add_(i / 1024)` (new scalar each time) | **+4999** |
| after `empty_cache()` | none released |

Confirmed. And Captum triggers it: 20 IG calls with the same `n_steps` leaked nothing, but sweeping `n_steps` from 10 to 109 leaked **+6361** buffers, because each `n_steps` produces a new set of step-size floats.

Fix: a bounded per-thread cache in `MakeBuffer` in `csrc/eager/op_dispatcher.cc`, plus `tests/scalar_buffer_cache_test.py`. On the patched wheel the same 5000-scalar loop leaks +903 (the 1024-entry cache clearing four times) and the Captum sweep +280. PR google-pytorch/torch_tpu#6102, open.

A side finding: on current main the strict-eager case no longer leaks thanks to a transfer refactor, but DeferAndFuse still does (+5001), so the fix still matters.

### 6.2 `aten::max_unpool2d` missing (issue #6103, PR #6105)

We ported Captum's 78 attribution test configs into a device-generic suite that runs them on CPU and on `CAPTUM_TEST_DEVICE=tpu` and compares. 266 generated tests: 262 passed, 4 failed in 5 m 25 s.

One failure was a tolerance miss at a near-zero attribution (+6.1e-5 versus -6.1e-5, precision noise). The other three were real: DeepLift's MaxPool rule calls `F.max_unpool2d`, and TorchTPU had no lowering for it. Every DeepLift variant on a CNN failed with `RuntimeError: operator 'aten::max_unpool2d' is not implemented for TPU`.

We filed the issue, then implemented `max_unpool2d` and `max_unpool3d` as a native StableHLO batched scatter. 25 unit tests and 4 error-message tests pass on v6e, and the three DeepLift conv configs now pass. PR google-pytorch/torch_tpu#6105, open.

### 6.3 Compiled partitions read CPU view arguments past their buffer (PR #6106)

The "expected tensor to be on tpu, got cpu" failure under `torch.compile` led somewhere unexpected. Reducing it, we found that `x * cpu_tensor[1].to(tpu)` under `torch.compile(backend="tpu")` returned zero. A StableHLO dump showed the compiled TPU partition being compiled against the *view's* strides and offset, while the argument is copied to the device as a contiguous tensor. The program read past the buffer it was given.

The fix in `split_compiler.py` compiles against the contiguous layout the argument is actually copied with. Test added, verified on v6e, CI green. PR google-pytorch/torch_tpu#6106, open.

### 6.4 Captum captured a CPU tensor into the compiled graph (Captum PR #1952)

The other half of the compile failure was Captum's. `LayerIntegratedGradients` computes output offsets with `torch.cumsum` on a CPU tensor and uses the result to slice layer outputs. Under `torch.compile`, that CPU tensor became a graph input on the wrong device. Replacing it with Python ints fixes it; 12 LIG tests pass on TPU. meta-pytorch/captum#1952, open.

With #1952 and #6106 together, the compile row in section 4 exists: 0.16 s warm, 7.0e-05 relative error.

### 6.5 Eight unregistered ops (branch `missing-ops-batch`)

A probe over ATen ops that TorchTPU does not register (`scripts/probe_unregistered_ops.py`) turned up a list. We implemented eight with unit tests: `smooth_l1_loss` and `huber_loss` with their backwards, `logaddexp`, `logaddexp2`, `nansum`, `std_mean`, `trace` and `addbmm`. 53 new tests pass on v6e against a wheel built from the branch. 23 files, 1733 lines.

The lesson that cost us a day: **TorchTPU forbids new composite kernels.** If a `TT_KERNEL` calls another ATen op, `OpNameStack` aborts at runtime with `CompositeOpCheck: Op trace is decomposed into as_strided ... implement the trace op by lowering it to SHLO instead`. The allow-list of known composites is capped at 150 entries with a "DO NOT ADD NEW ENTRIES" comment. `nansum`, `trace`, `std_mean` and `addbmm` all built fine as composites and all crashed; each was rewritten as a direct StableHLO lowering (a diagonal mask and reduction for `trace`, a `Select(IsNan(x), 0, x)` and sum for `nansum`, `Sqrt(Var)` and `Mean` with shared reduction plumbing for `std_mean`, batched matmul plus a batch-axis sum for `addbmm`).

Smaller gotchas, recorded so the next contributor skips them: `IgnoreInCacheKey` rejects `PromotedScalar` because it is already excluded from cache keys; `UnaryOpOptions` has no default constructor so `.op_param_cache_keys` must be set; TPU matmul tests need `set_float32_matmul_precision("highest")` to match CPU.

Still unregistered after this batch: `triu_indices`, `median`, `nanmedian`, `mode`, `kthvalue`, `frexp`, `renorm`, and the `.out` variants of about a dozen more.

### 6.6 Bounded-dynamism deny list: 11 stale entries, 1 real bug (branch `dynamism-denylist-triage`)

TorchTPU's bounded-dynamism tests (`tests/dynamism_test_utils.py`) carry a deny list of ops with "untriaged: invalid result". We removed the skips for 12 of them and ran the eager and compiled dynamic test targets on v6e (one trick: `--local_test_jobs=1`, because parallel shards fight over `/dev/vfio/0`).

Eleven pass in both modes: `eq`, `maximum`, `ne`, `neg`, `pow`, `remainder`, `round`, `sign`, `sinh`, `sub`, `trunc`. Entries removed. One is a real bug: `clamp.Tensor` (tensor-valued min and max) mismatches 80% of elements whenever any dimension is bounded, while the scalar variants pass. Moved to a diagnosed reason with the repro conditions.

### 6.7 Smaller things

- **Loader error message** (branch `loader-import-error-context`): a from-source wheel only imports against the torch version pinned in the repo lockfile. The failure was a bare `undefined symbol c10::report_overflow`. The versioned extension loader now names the glue it picked and the PyTorch it found. Test added.
- **Memory API docs** (branch `docs-memory-apis`, closes #4413): the `torch.accelerator` memory APIs documented in the API reference with numbers measured on v6e.
- **Triage comment on #4408**: the reported NaN in compiled SDPA backward does not reproduce on 0.2.0 with torch 2.15 (Qwen3, compiled versus CPU relative error 1.85e-5). Posted.
- **#4409** (compile plus backward on a view input) did not reproduce in our smoke test either; noted, not yet verified numerically.

---

## 7. What ships in TorchTPU from this

The `explainability-example` branch adds to torch_tpu:

- `examples/explainability/captum_attribution.py`: one script, flags for model (`tiny` or any HF causal LM), steps, runs, precision, `--compile`, `--internal-batch-size` and `--profile-dir`. It prints CPU reference time, first-call time, warm median, max relative error against CPU and peak HBM.
- `docs/source/case_studies_explainability.py` and its export, a seven-section case study: run Captum on a TPU, set the matmul precision, compare the execution modes, read the XProf trace, bound peak memory, check convergence, run the example. Linked from the case-study index and the examples README. `mkdocs build --strict` passes.

And a test suite for Captum (`tests/attr/test_accelerator.py`, meta-pytorch/captum#1951) that re-runs Captum's 78 attribution configs on CPU and any accelerator and compares, so "Captum works on TPU" stays a tested statement rather than a blog claim.

---

## 8. Honest caveats

- The v6e-1 is a single chip. Nothing here says anything about multi-chip or SPMD.
- Compiled mode needs two unmerged patches (Captum #1952, torch_tpu #6106). Until they land, use DeferAndFuse, which needs nothing.
- The example's CPU reference timings in the shipped run are not usable: a bazel build was loading the host CPU at the time. The TPU numbers were unaffected.
- The IG convergence problem on LLMs is real and not TPU-specific. Do not publish an attribution without its convergence delta.
- Five branches (`missing-ops-batch`, `explainability-example`, `dynamism-denylist-triage`, `docs-memory-apis`, `loader-import-error-context`) are pushed to the fork but have no upstream PR yet: GitHub refuses `createPullRequest` on the upstream repo for this account after three PRs opened the same day, while PR creation on the fork works. A retry is scheduled; the fallback is the compare page in the web UI.

---

## 9. Where everything lives

**Open upstream**

- google-pytorch/torch_tpu#6102: bounded scalar buffer cache (fixes #6036)
- google-pytorch/torch_tpu#6103: issue, `aten::max_unpool2d` missing
- google-pytorch/torch_tpu#6105: `max_unpool2d` / `max_unpool3d` lowerings
- google-pytorch/torch_tpu#6106: compile TPU partitions against the layout CPU arguments are copied with
- meta-pytorch/captum#1951: device-generic accelerator comparison tests
- meta-pytorch/captum#1952: Python ints for LayerIntegratedGradients output offsets

**Pushed, PR pending** (fork `joinalahmed/torch_tpu`)

- `missing-ops-batch`, `explainability-example`, `dynamism-denylist-triage`, `docs-memory-apis`, `loader-import-error-context`

**In this repository**

- `FINDINGS.md`: the raw numbers, in the order we found them
- `scripts/`: every benchmark, probe and repro mentioned above, plus `tpu_vm.sh` to create, set up and delete the VM
- `captum_on_torchtpu.ipynb`: the notebook version of sections 2 to 5
- `reports/Explainability tools on TPUs.md`: the prior-art research

**Budget note.** The whole thing ran on one v6e-1 that was created and deleted per work session. Delete the VM and its disk when you are done; both bill while they exist.

---

*Built with Claude Code as the pair programmer. Commits in torch_tpu carry an `Assisted-by` trailer per that project's contributing guide; AI is not a co-author there.*
