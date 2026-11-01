# Explainability on TPU, agent-assisted: a first project with the TPU agent suite

*A follow-up to [Running Captum on Cloud TPUs with TorchTPU](blog_captum_torchtpu.md). Same workload, Captum Integrated Gradients on `torch.device("tpu")`, but this time driven through Google's experimental TPU agent suite from inside Claude Code. Measurements on a Cloud TPU v6e-4 (`xai-v6e4`, us-east5-b); versions are in the result table. Code: [`tpu_agent_project/`](tpu_agent_project/) in [github.com/joinalahmed/captum-on-torchtpu](https://github.com/joinalahmed/captum-on-torchtpu).*

---

## Summary

- The TPU agent suite ([google-pytorch/tpu-agent-suite](https://github.com/google-pytorch/tpu-agent-suite), v0.7.0) is a package of specialist sub-agents and a unified compute layer, compiled into a native plugin for Claude Code (also Antigravity, with Codex and Cursor in progress). You install it into a project and talk to one skill, `tpu-agent`, which routes to a specialist.
- Its compute layer dispatched our Captum Integrated Gradients script to the TPU VM with one command and brought back a results file and an XProf trace. No hand-written `gcloud ssh` or `scp`.
- Its knowledge-base agent answered an IG memory-sizing question with every claim footnoted to a public URL and the gaps stated: 32 GB HBM per v6e chip at 1638 GBps, four separate HBMs in a v6e-4, and the stable `torch.accelerator.memory_*` API for the `tpu` device. It took 7.7 minutes and $2.85 of API spend for three facts.
- Its diagnostics agent read the XProf trace of one warm IG call. <!-- TODO: one-line finding from results/xprof_report.md -->
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

<!-- TODO: paste the verbatim result table from results/run.log, then 2-3 sentences: precision finding reproduced (default ~1e-3 vs highest ~1e-7), timing, trace size. -->

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

<!-- TODO: what prompts/profile.md asked, what tpu-diagnostics reported (results/xprof_report.md), with its numbers verbatim; cost and time. -->

## 5. What we learned

<!-- TODO: finalize after runs. Draft points: -->
- The compute layer is the most immediately useful part for a researcher: a reproducible "run this file over there and bring these outputs back", with job tracking and cleanup, that works the same on a VM and on GKE.
- The suite is built for inference onboarding of Hub models. Anything else, explainability included, is a compute-layer job plus the agents for what they own (facts, profiles). A "run my workload" specialist would widen it a lot.
- Citation-only answers are slower and pricier than a chat answer, and that is a feature when the number ends up in a capacity plan.
- Rough edges a first user hits: `python3` in hooks; the installer printed its CLAUDE.md message twice; the router's query-understanding is onboarding-shaped, so a non-onboarding request leans on the model to improvise.

## Where everything lives

| Path | What |
|---|---|
| `tpu_agent_project/README.md` | the suite explained, how to run the project, what it produced |
| `tpu_agent_project/setup.sh` | build, install, venv, compute profile |
| `tpu_agent_project/captum_ig_tpu.py`, `run_captum_ig.sh` | the workload and its dispatch |
| `tpu_agent_project/prompts/`, `run_agent.sh` | the two agent requests and the headless runner |
| `tpu_agent_project/results/` | run log, results.json, agent answers and reports |
| `upstream/tpu-agent-suite/` | the suite checkout (git-ignored) |
