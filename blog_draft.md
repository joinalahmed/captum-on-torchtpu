# Captum on TorchTPU: explainability for PyTorch models on Cloud TPUs

*#TorchTPUSprint — DRAFT. Slots marked TODO are filled from FINDINGS.md once runs complete.*

## Why
Captum is PyTorch's explainability library (~355k downloads/month). It has CPU and GPU CI and no mention of
TPUs anywhere: code, tests, docs, issues. With TorchTPU, a TPU is a normal PyTorch device, so the question is
simple: does Captum just work on `torch.device("tpu")`, is it numerically right, and is it fast? This post
answers all three and adds a device-generic test suite to Captum so the answer stays true.

## 1. Does it run? (yes, unmodified)
Smoke test of what attribution depends on, on a v6e-1 with torch_tpu 0.2.0: backward, `autograd.grad`,
forward/backward hooks, double backward, custom `autograd.Function`, `torch.func.grad`,
`torch.compile(backend="tpu")` + backward — 12/12 pass. Captum's `IntegratedGradients` and
`LayerIntegratedGradients` run with no code changes.

Full method matrix (every `captum.attr` class, MLP + CNN): TODO table from `results_matrix.txt`.

## 2. Is it right? (yes, with one setting)
Default TPU fp32 matmuls are reduced-precision. Attributions differ from CPU by ~1e-3 (LayerGradCam 1e-2).
```python
torch.set_float32_matmul_precision("highest")
```
brings every deterministic method to ~1e-7 of CPU. On Qwen3-0.6B, Captum `LayerIntegratedGradients`
CPU vs TPU: max relative error 8.5e-6, identical top tokens. TODO: precision "high" trade-off numbers.

## 3. Is it fast? (TODO)
Qwen3-0.6B, ~60-token prompt, IG steps 32/128/256: CPU vs TPU eager vs `torch.compile` vs DeferAndFuse,
at matmul precision "highest" and "high". TODO table from `results_ig.txt`. Note Captum already batches all
IG steps into one forward when `internal_batch_size` is None; that is what makes TPU shine here.

## 4. End to end on TPU
`AutoModelForCausalLM.from_pretrained(..., device_map="tpu")`, Captum `LayerIntegratedGradients`,
`LLMAttribution`, `LLMGradientAttribution`, HTML heatmap. TODO: what worked / what didn't from `results_demo.txt`.

## 5. Making it stick: a device-generic test suite for Captum
`tests/attr/test_accelerator.py` reuses Captum's 78 attribution configs and compares CPU vs any accelerator
(`CAPTUM_TEST_DEVICE=tpu`). TODO: pass/fail counts on TPU, issues filed, PR link.

## Gotchas
- Set matmul precision to "highest" for attribution (or accept ~1e-3).
- First call compiles; benchmark warm runs.
- GradientShap / NoiseTunnel / Lime etc. are stochastic: CPU and TPU won't match bit-for-bit.
- TODO: anything from the matrix / LLMAttribution runs.

Code and notebook: TODO links.
