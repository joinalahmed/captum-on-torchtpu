I spent the last several days running Captum's attribution methods on Cloud TPUs through TorchTPU, and wrote it up as a work log.

Why attribution? Integrated Gradients is a loop of forward and backward passes with Python scalars flowing into device ops, hooks on intermediate layers, and outputs that are differences of gradients. It stresses exactly the parts of a new PyTorch backend that a training loop does not.

What I found:

- Every public captum.attr method runs on torch.device("tpu") unmodified. 27 deterministic methods match CPU to ~1e-7; the 11 stochastic ones run.
- One setting decides correctness: torch.set_float32_matmul_precision("highest"). Default precision gives ~1e-3 error on attributions, "highest" gives ~1e-7, and for this workload it is free.
- On Qwen3-0.6B, Layer Integrated Gradients at 64 steps: 0.80 s strict eager, 0.36 s DeferAndFuse, 0.16 s torch.compile(backend="tpu"). CPU: 4 to 17 s.
- Eager attribution is launch-bound, not compute-bound. A device-side trace of one call showed 363 XLA programs, a 2% TensorCore duty cycle, and half the device op time spent splitting 64-bit Python scalars.
- Along the way: a device-buffer leak for Python scalars, a missing max_unpool2d lowering, a compiled-mode bug reading CPU view arguments past their buffer, a Captum bug placing a CPU tensor in the compiled graph, eight unregistered ops, 11 stale deny-list entries. Three PRs open on torch_tpu, two on Captum, five more branches waiting on a PR-slot limit.
- Everything re-validated on a four-chip v6e-4, including a sharded IG that matches the single-chip result to 5.6e-8 and pays off only when one chip runs out of memory.

The biggest lesson was not about TPUs. Before comparing two attribution implementations, check that the reference has converged. Mine had not.

Full log, with every number and the raw logs: https://github.com/joinalahmed/captum-on-torchtpu

Thanks to the TorchTPU and Captum maintainers for quick reviews.

#PyTorch #TPU #Explainability #Captum #TorchTPU #GoogleCloud #MachineLearning
