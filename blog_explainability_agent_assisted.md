# Explainability on TPU, agent-assisted: a first project with the TPU agent suite

*A follow-up to [Running Captum on Cloud TPUs with TorchTPU](blog_captum_torchtpu.md). Same workload, Captum Integrated Gradients on `torch.device("tpu")`, but this time driven through Google's experimental TPU agent suite from inside Claude Code. Measurements on a Cloud TPU v6e-4 (`xai-v6e4`, us-east5-b); versions are in the result table. Code: [`tpu_agent_project/`](tpu_agent_project/) in [github.com/joinalahmed/captum-on-torchtpu](https://github.com/joinalahmed/captum-on-torchtpu).*

---

## Summary

- The TPU agent suite ([google-pytorch/tpu-agent-suite](https://github.com/google-pytorch/tpu-agent-suite), v0.7.0) is a package of specialist sub-agents and a unified compute layer, compiled into a native plugin for Claude Code (also Antigravity, with Codex and Cursor in progress). You install it into a project and talk to one skill, `tpu-agent`, which routes to a specialist.
- Its compute layer dispatched our Captum Integrated Gradients script to the TPU VM with one command and brought back a results file and an XProf trace. No hand-written `gcloud ssh` or `scp`.
- Its knowledge-base agent answered an IG memory-sizing question with every claim footnoted to a public URL and the gaps stated: 32 GB HBM per v6e chip at 1638 GBps, four separate HBMs in a v6e-4, and the stable `torch.accelerator.memory_*` API for the `tpu` device. It took 7.7 minutes and $2.85 of API spend for three facts.
- Its diagnostics agent read the XProf trace of one warm IG call and found the TensorCore idle 98% of the window: 363 XLA programs for one Captum call, half of the device op time spent splitting 64-bit Python scalars, the MLP's matmuls 0.02%. Fourteen minutes and $7.74, every number traceable to an xprof tool call.
- What it does not cover yet: there is no specialist for "run my own script", training, or explainability, so the Captum workload went through the compute layer by hand. The hooks call `python3`, which must be 3.12, so a virtualenv has to be active when Claude Code starts.

---

## 1. What the TPU agent suite is

The TorchTPU FAQ answers "Is there an agent to make my onboarding easier?" with a link to the suite. It is not a model and not a hosted service. The repository holds three things:

| Part | What it is |
|---|---|
| `package/` | Markdown agent prompts, skills, JSON request/result schemas, Python tools, a curated knowledge base of public TPU pages |
| `framework/` | a compiler that translates the package into a native plugin per harness, an installer, a runtime with hooks, and the compute drivers |
| `eval/` | Harbor suites that score the agents on golden prompts |

Installed into a workspace for Claude Code it writes `.claude/skills/tpu-agent/`, one compiled prompt per specialist under `.claude/agents/`, hooks into `.claude/settings.json`, and two managed blocks into `CLAUDE.md`. The runtime, scripts, schemas and knowledge-base pages go to `~/.tpu-agent-suite/packages/<name@version>/`, and every session gets a run directory with artifacts and a journal.

Three design choices shape how it feels to use:

1. **The router has no shell.** `tpu-agent` can read, write and ask. A hook blocks every other tool once the skill is active, so the only way TPU work happens is by delegating to a specialist with a schema-checked JSON request. The authors explain why in the skill's own frontmatter: a router with a shell ran 48 XProf invocations itself instead of delegating once.
2. **All TPU work goes through `compute`.** One CLI covers a local TPU VM, a remote TPU VM over gcloud ssh, a GKE pod, a Kueue-aware GKE job, and XPK. `compute run --code X --input A --output B` copies the script in, runs it under the profile's interpreter, copies declared outputs back. A SessionEnd hook cancels jobs a session left running.
3. **Public facts are cited, never recalled.** The `CLAUDE.md` block forbids answering TPU hardware or runtime facts from memory. The `tpu-knowledge-base` agent reads indexed public pages and returns the URL for each claim.

The specialists in the release package: `tpu-model-onboarding` (Hugging Face model to TorchTPU inference, parity, benchmark, optional vLLM serving), `tpu-diagnostics` (XProf capture and reading), `tpu-knowledge-base`, `tpu-kernel-development` with `kernel-worker` (Pallas kernels), `tpu-unwiring` (removing CUDA assumptions) and `analyze-cuda-reference`. An experimental manifest adds roofline analysis, sharding search and vLLM optimization skills.

## 2. Setting it up

The suite is built from source today (release archives are planned). The whole setup is in [`tpu_agent_project/setup.sh`](tpu_agent_project/setup.sh):

```bash
git clone --depth 1 https://github.com/google-pytorch/tpu-agent-suite upstream/tpu-agent-suite
uv venv --python 3.12 .venv && uv pip install -r upstream/tpu-agent-suite/package/requirements-client.txt
python framework/compiler.py translate package --harness claude-code --output dist/claude-code/tpu-agent-suite
python framework/installer.py install dist/claude-code/tpu-agent-suite --destination tpu_agent_project --non-interactive
```

Compute is configured after the install, either by the agent in the first session or by hand. `setup plan` runs probes: it read our gcloud project and listed TPU VMs, and suggested `xai-v6e4` in `us-east5-b` by itself. `setup apply` saves the profile without contacting the VM; `setup verify` is the first thing that does.

```bash
runtime.py compute --package tpu-agent-suite@0.7.0 setup apply --profile default --answers \
  '{"backend":"gce_tpu_vm","project":"...","zone":"us-east5-b","tpu_name":"xai-v6e4","python_interpreter":"~/env/bin/python3"}'
```

Two things to know. The host needs Python 3.12 and the hooks call plain `python3`, so start Claude Code with the venv activated or the SessionStart hook fails quietly (`on_failure: allow`). And the installed `requirements-client.txt` pulls in XProf, Vertex AI and BigQuery clients, DuckDB and pandas: a few hundred megabytes, needed only by the diagnostics and onboarding specialists.

## 3. The project: Captum IG through the compute layer

[`captum_ig_tpu.py`](tpu_agent_project/captum_ig_tpu.py) is a 200-line workload in the style of the suite's own `hello_tpu.py`: it acquires the `tpu` device and labels the result (`native-tpu`, `cpu-fallback`, `cpu-simulated`), runs Saliency, IntegratedGradients and LayerIntegratedGradients on a small MLP and a small CNN, compares each against the CPU at the default fp32 matmul precision and at `"highest"`, reports the IG convergence delta and a warm timing, and under `--profile` wraps one warm IG call in `torch_tpu._internal.profiler` and copies the `.xplane.pb` beside itself. It prints a markdown table meant to be copied verbatim into a report, and fails if the `"highest"` results disagree with the CPU.

The dispatch is one command, in [`run_captum_ig.sh`](tpu_agent_project/run_captum_ig.sh):

```bash
runtime.py compute --package tpu-agent-suite@0.7.0 run --code captum_ig_tpu.py --timeout 900 \
  --output results.json:results/results.json --output xprof.xplane.pb:results/ig.xplane.pb -- --profile
```

The whole round trip, ssh in, run, scp two outputs back, took 51 seconds. The table the script printed, verbatim:

| field                                            | value                        |
| ------------------------------------------------ | ---------------------------- |
| device_type                                      | native-tpu                   |
| accelerator                                      | unknown                      |
| torch / torch_tpu / captum                       | 2.15.0+cu132 / 0.2.0 / 0.9.0 |
| Saliency/MLP rel err @default                    | 2.22e-03                     |
| IntegratedGradients/MLP rel err @default         | 2.07e-03                     |
| LayerIntegratedGradients/MLP[0] rel err @default | 3.13e-03                     |
| IntegratedGradients/CNN rel err @default         | 1.02e-02                     |
| Saliency/MLP rel err @highest                    | 1.65e-07                     |
| IntegratedGradients/MLP rel err @highest         | 4.36e-08                     |
| LayerIntegratedGradients/MLP[0] rel err @highest | 1.01e-07                     |
| IntegratedGradients/CNN rel err @highest         | 3.07e-07                     |
| IG convergence |delta| (highest)                 | 7.45e-08                     |
| IG 50 steps warm median s: tpu / cpu             | 0.0253 / 0.0037              |
| xprof                                            | captured (5634477 bytes)     |

Three things in it. First, the precision finding from the first post reproduces on a v6e-4 through a different path: 2e-3 relative error at the default (1e-2 for the CNN), 1e-7 with `"highest"`. Second, IG's convergence delta is at float noise, so the 50-step integral is converged on this model. Third, the TPU is slower than the CPU here, 25 ms against 4 ms per call: a 16-wide MLP cannot feed a systolic array, and the time is dispatch overhead, not arithmetic. The trace answers where exactly.

One failure on the way is worth keeping: the first attempt died because torch_tpu accepts only `highest`, `high` or `medium` as matmul precision values and I had passed `"default"`. The script exited in ten seconds; the compute layer then retried the `scp` of each of the two declared outputs ten times before reporting them missing, 3.5 minutes in all. A script that exits non-zero should short-circuit that.

## 4. Asking the agents

### Sizing IG memory with cited facts

Integrated Gradients expands the input to `n_steps` copies in one batched forward/backward, so peak memory scales with `n_steps`. We asked `tpu-agent`, headless (`claude -p`, [`run_agent.sh`](tpu_agent_project/run_agent.sh)), three sizing questions and told it to save a footnoted Markdown answer. The full answer is [`results/knowledge_base_answer.md`](tpu_agent_project/results/knowledge_base_answer.md). The parts that matter:

- 32 GB HBM per v6e chip, 1638 GBps, from the Cloud TPU v6e page; the scaling book gives the same figures rounded.
- A v6e-4 (`ct6e-standard-4t`, 2x2) is a half host with 4 chips. 128 GB total is four separate 32 GB HBMs, so an unsharded IG batch must fit in one chip's 32 GB.
- `torch.accelerator.memory_allocated`, `memory_reserved`, `max_memory_allocated`, `memory_stats` and friends are marked stable for TPU in TorchTPU's public API list; there is no `torch.tpu.memory_*`, and `get_device_properties.total_memory` is deprecated as of 0.2.1.
- Stated as not stated: how much of the 32 GB is free for user tensors, and what "reserved" means for the PJRT allocator.

That last bullet is the point of the design. The agent did not fill the gaps.

The cost: 3 turns, 7.7 minutes, $2.85. Most of the time is the knowledge-base agent fetching and summarising pages on first use; it caches them under `~/.tpu-agent-suite/tpu-knowledge-base/cache/` for later sessions.

### Reading the XProf trace

The second prompt ([`prompts/profile.md`](tpu_agent_project/prompts/profile.md)) pointed `tpu-agent` at `results/ig.xplane.pb`, described the workload, asked for device time, top ops, idle versus busy and a bound verdict, and forbade capturing anything new. The router delegated to `tpu-diagnostics`, which located the XProf skill shipped with the installed `xprof` 2.24.2, copied the trace into its run's artifact directory so the profiler's cache files would not land in the workspace, ran the profiler tools, and wrote a 757-line report ([`results/xprof_report.md`](tpu_agent_project/results/xprof_report.md)) in which every number is marked measured, derived with its formula, or inferred. The numbers:

| Metric | Value | XProf tool |
|---|---|---|
| Profiled window (TPU:0, 1 of 4 v6e chips) | 23.254 ms | get_roofline_model / query_events_db |
| Total device op time | 476.277 us | get_kernel_stats |
| Time inside XLA programs (363 launches) | 980.735 us | query_events_db |
| Gaps between programs (362 gaps, median 56.961 us) | 22,273 us = 95.8% | list_xplane_events |
| Duty cycle / idle | 2.0% / 98.0% | get_kpi_metrics, get_overview |
| MXU utilization | 0.0% | get_kpi_metrics |
| `X64SplitLow`/`X64SplitHigh` custom calls (302 occurrences) | 231.367 us = 48.58% of op time | get_kernel_stats |
| The MLP's four matmul fusions | 5.23 us = 0.02% of the window | get_kernel_stats |

The verdict is overhead-bound, and the report earns it: eager TorchTPU dispatches one XLA program per PyTorch op, so an IG call over 50 interpolation points is 363 launches with a median 57 us gap between them. The matmuls that are the "model" are 5 microseconds. Half of the op time is `X64SplitLow`/`X64SplitHigh`, the custom calls that split Python scalars arriving as f64/s64, which is the same scalar path whose buffer leak we fixed upstream in google-pytorch/torch_tpu#6102. The three `[4,16]` broadcast fusions are the interpolation `baseline + alpha * (input - baseline)`, once per step.

Two things the agent did that a human reading the trace might not: it noted that the roofline tool's `bound_by: HBM` label is an argmax over utilizations that are all 0.00% and declined to treat it as a bandwidth limit, and it recorded a sha256 of the trace it read. Cost: 11 turns, 14.3 minutes, $7.74. That is the price of a report whose every number you can check.

## 5. What we learned

- The compute layer is the most immediately useful part for a researcher: a reproducible "run this file over there and bring these outputs back", with job tracking and cleanup, that works the same on a VM and on GKE.
- The suite is built for inference onboarding of Hub models. Anything else, explainability included, is a compute-layer job plus the agents for what they own (facts, profiles). A "run my workload" specialist would widen it a lot.
- Citation-only answers are slower and pricier than a chat answer, and that is a feature when the number ends up in a capacity plan.
- The trace confirms from the device side what the first post measured with a stopwatch: eager attribution on TorchTPU is launch-bound, and the fix is `torch.compile(backend="tpu")` or DeferAndFuse, not a faster chip.
- Rough edges a first user hits: `python3` in hooks resolves to the system interpreter; `setup verify` fails on the suite's own `torch<2.14` pin against a released torch_tpu that ships 2.15; `inspect` reports a READY VM as unreachable when `tpu_info` is missing on it; a failed script still triggers ten scp retries per declared output. All of these, with evidence, are in [`tpu_agent_project/FEEDBACK.md`](tpu_agent_project/FEEDBACK.md).

## Where everything lives

| Path | What |
|---|---|
| `tpu_agent_project/README.md` | the suite explained, how to run the project, what it produced |
| `tpu_agent_project/setup.sh` | build, install, venv, compute profile |
| `tpu_agent_project/captum_ig_tpu.py`, `run_captum_ig.sh` | the workload and its dispatch |
| `tpu_agent_project/prompts/`, `run_agent.sh` | the two agent requests and the headless runner |
| `tpu_agent_project/results/` | run log, results.json, agent answers and reports |
| `upstream/tpu-agent-suite/` | the suite checkout (git-ignored) |
