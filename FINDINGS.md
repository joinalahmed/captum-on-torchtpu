# Findings log (Captum on TorchTPU)

Setup: Cloud TPU v6e-1, us-east5-b, torch 2.15.0, torch_tpu 0.2.0, Captum 0.9.0, transformers 5.22.1.
Scripts: `scripts/tpu_smoke.py`, `scripts/tpu_correctness.py`. Tiny models only (MLP, small CNN, TinyLM).

## 1. Supported (smoke test, 12/12 PASS = no exception)
backward, autograd.grad wrt input, forward hooks, full backward hooks, double backward (create_graph),
custom autograd.Function, torch.func.grad, torch.compile(backend="tpu") + backward on a view input
(open issue #4409 did NOT reproduce on 0.2.0; not yet verified numerically), Captum IntegratedGradients,
LayerIntegratedGradients.

## 2. Numerical agreement vs CPU (fp32, same weights/inputs), max relative error
| Method | default | `torch.set_float32_matmul_precision("highest")` |
|---|---|---|
| Saliency | 2.2e-3 | 1.7e-7 |
| InputXGradient | 1.7e-3 | 7.0e-8 |
| IntegratedGradients (50 steps) | 2.1e-3 | 4.4e-8 |
| LayerIntegratedGradients | 3.1e-3 | 1.0e-7 |
| LayerGradCam | 1.0e-2 | 2.2e-7 |
| torchtpu_xai batched IG | 2.1e-3 | 1.0e-7 |
| GradientShap | 3.6e-2 | 2.5e-2 (random sampling differs across devices; expected, not a bug) |

Takeaway: on TPU, fp32 matmuls default to reduced precision, which shows up as ~1e-3 relative error in
attributions. Setting matmul precision to "highest" matches CPU to ~1e-7. Attribution users should set it.
Caveat: tiny models; confirm on a real transformer.

## 3. Real model: Qwen3-0.6B, fp32, matmul precision "highest" (`scripts/hf_captum_bench.py`)
Captum `LayerIntegratedGradients` on the embedding layer, 5-token prompt "The capital of France is",
target " Paris", 32 steps:
- captum CPU vs captum TPU: max rel err **8.5e-6**; same top-3 tokens (" France", " of", "The").
  -> Captum works unmodified on a real transformer on TorchTPU, with CPU-level numerics.
- torchtpu_xai IG vs captum CPU: max rel err 7.1e-1 -> under investigation (Captum defaults to
  Gauss-Legendre quadrature, mine is right-Riemann; or a bug). See `scripts/hf_ig_diag_bench.py diag`.
- TorchTPU warns that "highest" precision "may be significantly lower" performance; benchmark "high" too.

Timing (seconds, warm median; first run incl. compile in parens), same 5-token prompt:
| steps | captum cpu | captum tpu | xai tpu eager |
|---|---|---|---|
| 8  | 0.57 | 0.57 (1.35) | 0.50 (0.54) |
| 32 | 0.83 | 0.57 (0.50) | 0.53 (0.60) |
| 64 | 1.10 | 0.60 (1.88) | 0.58 (0.64) |
TPU time is flat in steps (~0.55 s) -> dominated by fixed overhead at this tiny size, not compute.
Only ~2x over CPU at 64 steps. Need longer prompts / more steps / compile to show real advantage.

## 4. Correction to the pitch
Captum's IntegratedGradients already runs all n_steps in ONE batched forward/backward when
`internal_batch_size` is None (it expands inputs to n_steps x batch). So "batch IG steps into one
fixed-shape call" is not a differentiator over Captum; the timing parity (captum tpu ~= xai tpu) confirms it.
Project repositioned: TorchTPU support *in Captum* (full-method compat matrix, fixes upstreamed as PRs,
TPU test path, "Captum on TPUs" tutorial incl. the matmul-precision finding). torchtpu_xai -> helpers only.

## 5. IG discrepancy resolved: not a bug, IG does not converge on Qwen3 with a pad baseline
`scripts/hf_ig_diag_bench.py diag`, Qwen3-0.6B fp32 "highest", prompt "The capital of France is", target " Paris",
baseline = pad token ids:
- Same rule, same steps: torchtpu_xai IG == Captum `method="riemann_right"` to the printed digit
  (steps=32: sum -11.3576 / delta 28.51 both; steps=128: sum 35.8381 / delta 18.69 both). Implementation is correct.
- torchtpu_xai IG cpu vs tpu (32 steps): rel err 1.4e-4 (highest precision).
- The reference itself is not converged: Captum CPU gausslegendre **256 steps has convergence delta 27.3 against
  a total attribution of 44.4**. Sums swing -26.6 -> -8.2 -> +44.4 across 32/128/256 steps.
  => For LLMs, raw-logit target + pad-token baseline is a poor IG setup; always print `return_convergence_delta`.
  Try log-softmax target, zero-embedding baseline, more steps / `internal_batch_size`. Not TPU-specific.
- Decision: drop torchtpu_xai IG as a product; keep its convergence check as a test utility.

## 6. IG scaling on Qwen3-0.6B, 53-token prompt (`hf_ig_diag_bench.py bench`, precision "highest")
| steps | captum cpu | captum tpu eager | xai tpu eager | xai tpu `torch.compile(backend="tpu")` |
|---|---|---|---|---|
| 32 | 4.28 s | 0.86 s (first 3.2) | 0.62 s | **0.13 s** (first run 112 s = compile) |
| 128 | 16.39 s | **1.17 s** (first 3.1) = 14x CPU | OOM | OOM |
OOM (xai variants only): "Attempting to allocate 3.84G ... 1.04G free" = the full logits tensor [128, 53, 151936]
fp32 held alongside the [128, 53, 1024] interpolation path with grad. Captum's LIG survived 128 steps because it
chunks internally via `internal_batch_size=steps`... but every LLM attribution still materialises [B, T, V] logits
before selecting `[:, -1, target]`; `model(..., logits_to_keep=1)` (transformers v5) removes that. After the OOM the
process was left with ~141 MB HBM free ("Error loading program tt_jit_mul"), so later rows in that process are invalid.
TPU first-compile is 112 s; tier-2 compile cache is disabled for single-device programs, so every new process pays it
(tier-3 GCS cache `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=gs://...` would avoid this).
Precision "high" (warm s): captum tpu 0.880 / 1.078 at 32 / 128 steps; xai compile 0.102. => matmul precision is
not the bottleneck for eager attribution; "highest" costs ~nothing here. Compile first-run 68 s at "high".

**DeferAndFuse (`TORCH_TPU_DEFER_AND_FUSE=1`, precision high), warm s:**
| steps | captum cpu | captum tpu strict eager | captum tpu DeferAndFuse | xai compile |
|---|---|---|---|---|
| 32 | 4.48 | 0.880 | **0.238** (first run 26 s) | 0.176 |
| 128 | 17.04 | 1.078 | **0.450** (first run 29 s) | - |
DeferAndFuse makes *unmodified Captum* 3.7x faster than strict eager and **38x faster than CPU** at 128 steps, with
no code change: one env var. Cost: ~26-29 s fusion compile on first call. This is the headline perf finding.
256 steps fail everywhere (full [256,53,151936] logits = 8 GB) -> `scripts/hf_ig_bench2.py` with logits_to_keep=1 queued.

## 7. Captum's own test configs on TPU (device-generic suite, `CAPTUM_TEST_DEVICE=tpu`)
266 generated tests (78 configs x algorithms), Captum 0.9.0, torch_tpu 0.2.0, precision "highest": **262 passed, 4 failed** in 5m25s.
- 1 tolerance miss: `basic_multi_input_dl_shap_DeepLiftShap`, values +6.1e-5 vs -6.1e-5 (|diff| 1.2e-4 > dp_delta 1e-4) at a
  near-zero attribution. Precision noise, not a method bug.
- 3 real gaps (fixed by torch_tpu#6105): `conv_layer_single_target_LayerDeepLift`, `conv_neuron_NeuronDeepLift`, `conv_neuron_slice_selector_NeuronDeepLift`
  all fail with **`RuntimeError: operator 'aten::max_unpool2d' is not implemented for TPU`**. DeepLift's MaxPool rule calls
  `F.max_unpool2d`. => missing op lowering in torch_tpu; file / implement.

## 7b. Full captum.attr method matrix on TPU (`scripts/captum_tpu_matrix.py`, MLP + small CNN, precision "highest")
**27 OK, 11 RUNS (stochastic, run-only), 0 DIFF, 0 FAIL** (first pass had Lime/KernelShap failing only for a missing `sklearn`).
Every deterministic method matches CPU to 1e-7..1e-8 relative: IG, Saliency, InputXGradient, DeepLift, GuidedBackprop,
Deconvolution, GuidedGradCam, FeatureAblation, Occlusion, ShapleyValues, LRP, and all Layer*/Neuron* variants incl.
LayerGradCam, LayerLRP, LayerFeatureAblation, NeuronConductance. Stochastic methods (GradientShap, ShapleyValueSampling,
NoiseTunnel, *Shap) run but differ as expected from RNG. The fix for #6036 passes its regression test (3 passed) on the
from-source wheel.

## 8. torch_tpu #6036 confirmed, and attribution triggers it
0.2.0 wheel, `scripts/scalar_leak_check.py`: same scalar x5000 -> +2 live allocations; new scalar each x5000 -> **+4999**;
`empty_cache()` releases none. Captum IG, same n_steps x20 -> +0; **n_steps 10..109 -> +6361** (each n_steps yields new
step-size floats, each a leaked 0-d buffer). Fix: bounded cache in `MakeBuffer` (PR google-pytorch/torch_tpu#6102). Patched wheel, same script: new scalar x5000 -> **+903**
(1024-entry cache cleared 4x), Captum n_steps sweep -> **+280** (was +6361); regression test `tests/scalar_buffer_cache_test.py` 3 passed; rebuilt after merging main (ba59861): same numbers, 3 passed.
On unpatched current main the strict-eager case no longer leaks (main's transfer refactor), DeferAndFuse still does (+5001).
From-source wheel (cp314) only imports with the torch version pinned in the repo lockfile (2.11.0+cpu / 2.14.0+cpu),
not latest 2.15: the versioned .so loader picks `glue_2_14_0` and fails with `undefined symbol c10::report_overflow`.

## 9. Other findings from the queue
- `torch.accelerator.current_accelerator()` -> `tpu`, is_available True, device_count 1: torch_tpu registers properly.
- IG with `logits_to_keep=1`, 53 tokens: 128 steps in ONE batch = 0.857 s warm on TPU vs 11.2 s CPU (13x), peak HBM 23.9 GB
  (8.0 GB at 32 steps). 256 steps OOMs on v6e-1 (32 GB) -> use `internal_batch_size` beyond ~128 steps.
- Captum + `torch.compile(backend="tpu")` through HF's `input_ids` path fails: "expected tensor to be on tpu, got cpu" for a
  shape-[2] int64 graph input (HF-created index tensor). Compiled mode needs all tensor args on device. The
  `inputs_embeds` path (our earlier 0.10 s result) avoids it.
- Other libraries (`scripts/other_xai_libs_tpu.py`): shap DeepExplainer/GradientExplainer PASS, pytorch-grad-cam GradCAM/ScoreCAM PASS,
  nnsight trace with `device_map="tpu"` PASS. Inseq FAIL `ValueError: Unknown device tpu` (device allow-list -> small upstream PR).
  TransformerLens FAIL `transformers has no attribute TRANSFORMERS_CACHE` (transformers v5 incompatibility, not TPU).
- End-to-end demo (`scripts/demo_captum_qwen_tpu.py`): `device_map="tpu"` load 2.9 s; LIG 64 steps 2.0 s incl. compile;
  `LLMAttribution(FeatureAblation)` 9.9 s; `LLMGradientAttribution(LIG)` 0.89 s. All Captum LLM APIs work on TPU.
- #4408 guard (`scripts/compile_nan_check.py`, Qwen3 sdpa, d logit/d embeds): no NaN in eager or compiled; compiled vs CPU rel err
  1.85e-5, eager vs CPU 2.76e-5. Does not reproduce on 0.2.0 / torch 2.15.
- Compile + Captum: a plain `torch.compile(model, backend="tpu")(input_ids=..., logits_to_keep=1)` call works; the failure above
  only appears when Captum's LIG wraps the compiled model (its forward hooks / index tensors), so it is a Captum+compile interaction.
- XProf (`scripts/profile_ig_modes.py`, LIG 32 steps, torch.profiler with ProfilerActivity.TPU): strict eager Self CPU 1.93 s, of
  which `aten::_to_copy` is 70% (1.35 s, 522 calls, 2.6 ms avg = host<->device copies that block); DeferAndFuse Self CPU 0.78 s,
  `_to_copy` 0.29 s (0.57 ms avg). Self TPU 5.0 s vs 1.23 s. Traces in `~/prof/{strict,daf}/` on the VM. Compiled mode fails as above.
- Missing op filed: google-pytorch/torch_tpu#6103 (`aten::max_unpool2d`, repro `scripts/max_unpool2d_repro.py`).

## Upstream status
- meta-pytorch/captum#1951 (device-generic accelerator tests) - open, needs Meta CLA + maintainer CI approval.
- google-pytorch/torch_tpu#6102 (bounded scalar buffer cache, fixes #6036) - open, needs Google CLA.
- google-pytorch/torch_tpu#6103 (max_unpool2d missing) - filed, then implemented in google-pytorch/torch_tpu#6105
  (native StableHLO batched scatter; 25 unit + 4 error tests pass on v6e; Captum's 3 DeepLift conv configs now pass).

## 10. Explainability example across execution modes (`upstream/torch_tpu` branch `explainability-example`)
`examples/explainability/captum_attribution.py`, Qwen3-0.6B LIG, 42-token prompt, 64 steps, fp32 "highest", v6e-1:
| mode | warm median | first call | max rel err vs CPU | peak HBM |
|---|---|---|---|---|
| strict eager | 0.80 s | 5 s | 9.2e-05 | 10.6 GiB |
| DeferAndFuse | 0.36 s | 43 s | 9.3e-05 | 11.1 GiB |
| torch.compile(backend="tpu") (patched Captum #1952 + torch_tpu #6106, env_fix) | 0.16 s | 130 s | 7.0e-05 | 15.3 GiB |
Tiny MLP IG: 0.065 s strict / 0.039 s DaF / 0.062 s compile, rel err 2e-7 in all modes. 128 steps with
`internal_batch_size=64`: 1.42 s, rel err 5.5e-4, peak 10.6 GiB. Numbers taken while a bazel build loaded the host CPU;
CPU reference timings from that run (67 s) are not usable.

## 11. Missing-ops batch (`upstream/torch_tpu` branch `missing-ops-batch`)
smooth_l1_loss(+backward), huber_loss(+backward), logaddexp, logaddexp2, nansum, std_mean, trace, addbmm implemented;
53 new unit tests pass on v6e against a wheel built from the branch; the probe over `UNREGISTERED_OPS` shows all eight dispatch.
Lesson: TorchTPU forbids new composite kernels. `OpNameStack` aborts with "CompositeOpCheck: Op X is decomposed into Y"
when a TT_KERNEL calls another ATen op, and `known_composite_ops` is capped at 150 entries. nansum/trace/std_mean/addbmm were
first written as at::-composites (built fine, crashed at runtime) and rewritten as StableHLO lowerings.
Also: `IgnoreInCacheKey` rejects `PromotedScalar` (already excluded from keys); `UnaryOpOptions` needs
`.op_param_cache_keys` set (no default ctor); TPU matmul tests need `set_float32_matmul_precision("highest")`.
Still unregistered after this batch (probe): triu_indices, median/nanmedian/mode/kthvalue, frexp, renorm, and `.out` variants of
hypot, copysign, erfc, sinc, mish, index_select, heaviside, nextafter, range, logspace, hardshrink.

## 12. Bounded-dynamism deny-list triage (`upstream/torch_tpu` branch `dynamism-denylist-triage`)
Ran 12 "untriaged: invalid result" entries of `tests/dynamism_test_utils.py` with their skips removed, via
`bazel test //tests:ops_test_dynamic_tpu-v6e` and `..._compiled_tpu-v6e` (all dtypes, 1 sample/op, `--local_test_jobs=1`;
parallel shards fail with "Failed to acquire a TPU device node /dev/vfio/0"). eq, maximum, ne, neg, pow, remainder, round, sign,
sinh, sub, trunc pass in both modes -> entries removed. clamp fails: `clamp.Tensor` (tensor min/max) mismatches 80% of elements
with any dim bounded (eager), scalar variants pass -> moved to a diagnosed reason. clamp_min left untriaged.

## Upstream status (continued)
- google-pytorch/torch_tpu#6106 (CPU view args in split compile) - open, CI green.
- meta-pytorch/captum#1952 (LIG cumsum as Python ints) - open.
- #4408 triage comment posted (no NaN on 0.2.0; compiled/CPU rel err 1.85e-5).
- Branches pushed, PR creation blocked by GitHub: `docs-memory-apis`, `loader-import-error-context`, `missing-ops-batch`,
  `explainability-example`, `dynamism-denylist-triage`. Retried 2026-11-01 09:35 (+03): GraphQL `createPullRequest` still refused,
  REST `POST /pulls` returns 404, while a no-op PATCH of our own PR #6106 succeeds and rate limits are untouched. So the account can
  write to the repo but not open PRs there; fork PR creation works. Root cause (web compare page, 2026-11-01): "Users without write access can have up to 3 pull requests open at a time" on this
  repository; #6102, #6105 and #6106 fill the quota. The remaining five wait until one of those is merged or closed.
- google-cla[bot] on #6102: CLA not yet signed; the check fails until it is.
- Issues filed 2026-11-01 to claim the blocked branches: #6111 missing-ops-batch, #6112 dynamism-denylist-triage,
  #6113 loader-import-error-context, #6114 explainability-example; docs-memory-apis is a comment on #4413.
- PR #6106 retitled "Fix compiled TPU partitions reading CPU view arguments past their buffer".
- Approach comments posted on #6111-#6114 (`comment_61xx_approach.md`). Self-assignment is impossible: GitHub only allows
  collaborators/org members as assignees (`/repos/.../assignees/joinalahmed` -> 404) and the repo has no `/assign` bot; each
  comment asks a maintainer to assign.

## 13. Multi-chip validation on a v6e-4 (us-east5-b, 2026-11-01)
Capacity: on-demand and spot v5litepod-4/8 and v6e-4/8 were unavailable in us-east5-b and us-central1-a all morning
("Insufficient capacity"); a v5litepod-8 is also over the 4-chip per-zone serving quota. A queued v6e-4 landed after ~3 h.
Setup/test scripts: `scripts/vm_multichip_setup.sh`, `scripts/vm_multichip_tests.sh`, `scripts/captum_multichip_ig.py`.
- Process model: libtpu is exclusive per process; a second process touching the TPU fails with "The TPU is already in use by
  process with pid N". Multi-chip means one torchrun worker per chip (`eval $(python -m torch.tpu.distributed | sed 's/^/export /')`,
  `torchrun --nproc_per_node=$WORLD_SIZE`); each worker sees `device_count() == 1`. `TPU_VISIBLE_CHIPS`-style variables do nothing.
- Attribution matrix on chip 0 (`results/v6e-4/matrix_chip0.txt`): 27 OK, 11 RUNS, 0 DIFF/FAIL, identical to the v6e-1.
- IG sharded across 4 chips with `tpu_dist` all_reduce (`results/v6e-4/multichip_ig.md`): 64 steps 0.355 s vs 0.219 s on one chip
  (0.62x, the problem is too small to split); 256 steps 0.327 s vs 0.547 s with `internal_batch_size=64` (1.67x) and OOM for a
  single 256 batch on one chip; agreement with Captum 5.6e-8 at equal batch size, 1.2e-4 otherwise.
- From-source wheel on the 4-chip host: `torch_tpu-0.2.1-cp314` (hermetic default Python is 3.14), needs torch 2.11.0+cpu,
  `libtpu`, `pyzmq`, `expecttest`/`hypothesis` for the in-repo tests, and `_internal/utils/test_utils.py` copied from `src/` (the
  wheel omits test helpers). Each branch rebuild from the main cache takes ~9 min on 180 vCPUs.
- Branch results (`results/v6e-4/branches/`, variant `_tpu-v6e`, `--local_test_jobs=1`, each branch rebuilt from source on the host):

  | branch | bazel (as CI runs it) | pytest against the installed wheel |
  |---|---|---|
  | fix-scalar-buffer-cache-bound | - | `scalar_buffer_cache_test.py` 3 passed; leak counts identical to v6e-1 (+2 / +903 / +0 / +280) |
  | max-unpool2d | `ops_test_tpu-v6e` (test_max_unpool*) PASSED | `ops_unit_test.py -k unpool` 25 passed |
  | cpu-view-args-split-compile | `//tests/compile:split_compile_test` (cpu_view_argument) PASSED | `split_compile_test.py -k cpu_view_argument` 4 passed |
  | missing-ops-batch | `ops_test_tpu-v6e` (7 ops) PASSED | `ops_unit_test.py` (8 ops) 53 passed |
  | dynamism-denylist-triage | `ops_test_dynamic_tpu-v6e` and `ops_test_dynamic_compiled_tpu-v6e` (11 ops) PASSED | - |
  | loader-import-error-context | `_versioned_so_loader_test` PASSED | 16 failed: the test inspects the repo package layout, invalid against a wheel |
  | docs-memory-apis | - | memory_allocated / max_memory_allocated / memory_reserved / get_memory_info / hbm_bytes_per_device all return sane values |
  | explainability-example | - | strict, DeferAndFuse, compile: all run, max rel err vs CPU 2.0e-7 |

  Two sessions ran the loop concurrently for ~3 minutes (racing the checkout, bazel lock and exclusive TPU); that pass was
  discarded (`multichip_summary_collided.log`).
- At ~11:45 the host-side TPU session manager started refusing every process: libtpu logs `tpunetd_client ... MaybeInitializeSliceBuilder
  failed: GRPC_ERROR START_SESSION failed. Contact GCE with UUID ...`. A guest reboot and a `tpu-runtime.service` restart did not clear it,
  `gcloud compute tpus tpu-vm stop` is not supported for queued-resource nodes, and the node still reports HEALTHY. The last thing that
  touched the TPU before the failure was a 4-worker torchrun from an env whose `libtpu` had just been downgraded to 0.0.50 by the repo
  requirements lock (the released env uses 0.0.51); whether that is cause or coincidence is unknown. Retried every ~3 min for 35 min
  with no recovery; the node was deleted at 12:20 UTC on 2026-11-01. Build disks kept.
- A second v6e-4 was queued with the cache disk and landed at 13:45 UTC (fresh boot disk; setup from the disk cache took 7 min, each
  branch rebuild ~9 min). The items the wedged host had blocked all completed (`results/v6e-4/run2/`): the pytest halves of
  max-unpool2d (25 passed) and missing-ops-batch (53 passed) once `attrs` and `cattrs` were added to the from-source env,
  `//tests/compile:split_compile_test` for the CPU-view branch (PASSED), and the per-chip torchrun leak check, where all four ranks
  report the same counts as the single chip (+2 / +903 / +0 / +280). The table above reflects the final state. The same libtpu
  0.0.51 was used in both envs this time and the host stayed healthy, which is weak evidence for the 0.0.50-downgrade theory above.
  Still uncovered: a v5e-8 (quota) and `tests/distributed/`. The second node and both build disks were deleted at ~18:15 UTC on
  2026-11-01; nothing from this work is billing.

## TODO
- Captum LIG + torch.compile(backend="tpu"): find which CPU index tensor reaches the compiled graph; fix in Captum or document.
- Inseq device allow-list PR; torch_tpu examples/docs entry; precision-guide numbers (#5727); #4409 triage comment.
