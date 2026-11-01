# Captum on TPU through the TPU agent suite

A first project with the **TPU agent suite** (`google-pytorch/tpu-agent-suite`,
v0.7.0, experimental). It drives the Captum explainability-on-TorchTPU work of
this repository through the suite's two layers: the **unified compute layer**
dispatches a Captum Integrated Gradients script onto the Cloud TPU VM, and the
**`tpu-agent` skill** (Claude Code) answers a sizing question with cited TPU
facts and reads the XProf trace the run brought back.

## What the TPU agent suite is

The torch_tpu FAQ ("Is there an agent to make my onboarding easier?") points
to it. It is an open-source *package of agents*, not a model: Markdown agent
prompts, skills, JSON request/result schemas, Python scripts and a knowledge
base, compiled by its own framework into native plugins for several agent
harnesses (Claude Code, Antigravity, Codex and Cursor). Installed into a
workspace it adds:

| Piece | Where it lands (Claude Code, project scope) |
|---|---|
| `tpu-agent` router skill + `tpu-agent-export-trajectory` | `.claude/skills/` |
| Specialist sub-agents: `tpu-model-onboarding`, `tpu-diagnostics` (XProf), `tpu-knowledge-base` (cited facts), `tpu-kernel-development` + `kernel-worker` (Pallas), `tpu-unwiring`, `analyze-cuda-reference` | `.claude/agents/` |
| Lifecycle hooks (run id per session, schema validation of every delegation, tool allow-list enforcement on the router, job reaping at session end) | `.claude/settings.json` |
| Managed memory blocks (never answer public TPU facts from memory; all TPU work goes through `compute`) | `CLAUDE.md` |
| Compute profile(s) + ownership ledger | `.tpu-agent-suite/` |
| Runtime, scripts, schemas, knowledge-base pages | `~/.tpu-agent-suite/packages/tpu-agent-suite@0.7.0/` |
| Per-session artifacts and an experiment journal | `~/.tpu-agent-suite/runs/<run_id>/` |

Design choices worth knowing before using it:

- **You talk to one skill.** `tpu-agent` has `read`/`write`/`ask` tools and
  deliberately **no shell**; a PreToolUse hook blocks anything else once the
  skill is active, so the only path to TPU work is delegating to a specialist.
- **Every TPU job goes through `runtime.py compute`.** Backends: `local`,
  `gce_tpu_vm` (gcloud ssh/scp with a persistent control socket), `gke_pod`,
  `gke_job` (Kueue-aware), `gke_xpk`. `compute run --code X --input A --output B`
  copies the script and inputs to the target, runs it under the profile's
  `python_interpreter`, and copies declared outputs back. Jobs a session leaves
  running are cancelled by a SessionEnd hook.
- **Facts are cited.** `tpu-knowledge-base` resolves public TPU facts against
  an index of public pages and returns URLs; the memory block forbids
  answering such facts from memory.
- **Scope today is inference onboarding, profiling and kernels** for
  Hugging Face models on v6e/v7x. There is no "run my arbitrary script"
  specialist, which is why this project uses the compute layer directly for
  the Captum workload and the agents for what they own (facts, profiles).

## Layout

| File | Purpose |
|---|---|
| `setup.sh` | clone + build the suite, install it here for Claude Code, create the Python 3.12 host venv, save the `gce_tpu_vm` profile (`xai-v6e4`, `us-east5-b`) |
| `captum_ig_tpu.py` | the workload: Saliency / IntegratedGradients / LayerIntegratedGradients on an MLP and a CNN, CPU-vs-TPU agreement at default and `"highest"` fp32 matmul precision, IG convergence delta, warm timing, optional XProf capture |
| `run_captum_ig.sh` | `compute preflight` -> `compute inspect` -> `compute run ... --output results.json --output xprof.xplane.pb` |
| `prompts/knowledge_base.md`, `prompts/profile.md` | the two `tpu-agent` requests |
| `run_agent.sh` | headless Claude Code (`claude -p`) in this workspace with a tool allow-list |
| `results/` | what the runs produced (trace files are git-ignored) |
| `FEEDBACK.md` | detailed first-user feedback for the suite's authors |
| `../blog_explainability_agent_assisted.md` | the sprint write-up built on this project |
| `.claude/`, `CLAUDE.md` | written by the suite's installer; committed so the workspace is inspectable. `.tpu-agent-suite/` (compute profile, ownership ledger, current run id) is also written by it, and the installer adds it to `.git/info/exclude`, so it stays local |

## How to run

Prerequisites: `uv`, `gcloud` authenticated (`source ../scripts/env.sh`),
Claude Code >= 2.1.219, a TPU VM whose `~/env/bin/python3` has `torch_tpu`
and `captum` (this repo's `scripts/tpu_vm.sh setup` does that).

```bash
./setup.sh                              # one-time; no VM contact
./run_captum_ig.sh                      # Captum IG on the TPU VM via the compute layer
./run_agent.sh prompts/knowledge_base.md   # cited TPU facts (host only)
./run_agent.sh prompts/profile.md          # tpu-diagnostics reads results/ig.xplane.pb
```

Interactive use works too: `cd tpu_agent_project && source .venv/bin/activate && claude`,
then ask in plain language; the venv matters because the hooks call `python3`
and the runtime needs 3.12.

## What it produced so far (status 2026-11-01)

| Step | Status | Output |
|---|---|---|
| Suite built from source (all four harnesses) and installed here for Claude Code | done | `.claude/`, `CLAUDE.md`, `.tpu-agent-suite/` |
| `gce_tpu_vm` compute profile for `xai-v6e4` saved via `setup plan` / `setup apply` | done, unverified | `.tpu-agent-suite/environment.json` |
| `tpu-agent` -> `tpu-knowledge-base`, headless, IG memory-sizing question | done: 3 turns, 7.7 min, $2.85 | [`results/knowledge_base_answer.md`](results/knowledge_base_answer.md) |
| `captum_ig_tpu.py` logic check with `MOCK_TPU=1` (CPU) | done, exit 0 | not kept |
| `compute preflight` / `setup verify` / `inspect` against the VM | **not run** | |
| `compute run captum_ig_tpu.py --profile` on the TPU | **not run** | |
| `tpu-agent` -> `tpu-diagnostics` on the trace | **not run** | |

The TPU steps did not run: `xai-v6e4` was held by another test loop all
afternoon and then failed at TPU initialisation (host-side `tpunetd
START_SESSION failed`, not cleared by a reboot) and was deleted. When a
replacement v6e-4 lands, the only changes needed are the profile's `tpu_name`
/ `zone` (`setup apply`) and `torch_tpu` + `captum` in the VM's `~/env`; then
`./run_captum_ig.sh` and `./run_agent.sh prompts/profile.md` fill the rest of
this table.

What the knowledge-base run showed, in short: 32 GB HBM per v6e chip at 1638
GBps, a v6e-4 is four separate 32 GB HBMs (so an unsharded IG batch must fit in
one chip), and `torch.accelerator.memory_*` is the stable memory API for the
`tpu` device. Each claim carries its public URL, and what no source states is
said so. See [FEEDBACK.md](FEEDBACK.md) for what worked and what did not.
