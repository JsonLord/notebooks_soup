# Training Agent — Existing Colab Session Takeover & Capability Assessment

## Mission

Take control of the most recently active, already-running Google Colab session for the OpenTraining workflow, using the local workdir:

```text
/a0/usr/work/notebooks_soup
```

This is a discovery and planning pass first. Do not start a real training job until the environment, repository, accelerator, dependencies, artifact paths, and viable experiment classes are understood.

## Non-negotiable execution policy

The controller terminal is NOT a Python/training runtime.

You MUST obey AGENTS.md and COLAB_ONLY_EXECUTION_POLICY.md.

Local execution of Python, Jupyter, Soup, Unsloth, Transformers/TRL training, CUDA probes, .py/.ipynb/training .sh files, torchrun, accelerate, pytest that executes repo code, or equivalent interpreter bypasses is forbidden.

If such execution happens locally:

```text
LOCAL_EXECUTION_POLICY_VIOLATION
HARD STOP
```

Report the exact command and stop.

All notebook/training Python must execute in the selected remote Colab runtime through:

```bash
bash scripts/colab_remote_exec.sh <file.py|file.ipynb> "$COLAB_SESSION"
```

or direct `colab exec`.

There is NO local fallback.

Do not create a new Colab session during this pass.

## 1. Enter workdir and activate guard

Use local orchestration commands only:

```bash
cd /a0/usr/work/notebooks_soup
git status --short --branch
git rev-parse HEAD
source scripts/activate_colab_only_guard.sh
```

Confirm:

```bash
echo "$TRAINING_EXECUTION_MODE"
type python
type python3
type jupyter
type torchrun
```

Expected:

```text
TRAINING_EXECUTION_MODE=COLAB_ONLY
```

## 2. Resolve latest existing Colab session

First inspect:

```bash
colab sessions
```

If `COLAB_SESSION` is already set and active, prefer it.

Otherwise determine the most recently active session from the available session/status metadata using only orchestration tools such as `colab status`, `jq`, `grep`, `sed`, `awk`, `sort`.

Do not execute Python locally to parse session state.

Set:

```bash
export COLAB_SESSION="<resolved-name>"
```

Verify:

```bash
colab status -s "$COLAB_SESSION"
```

If no active session exists:

```text
COLAB_REMOTE_REQUIRED
HARD STOP
```

If multiple sessions exist and "latest" cannot be resolved safely:

```text
COLAB_SESSION_AMBIGUOUS
HARD STOP
```

Return the candidates and evidence. Do not create another runtime.

## 3. Capture session/account state

Record without exposing secrets:

```bash
colab status -s "$COLAB_SESSION"
colab usage
colab url -s "$COLAB_SESSION"
```

Do not print access tokens, session tokens, OAuth secrets, or credential files.

## 4. Remote capability probe

Create the probe source locally as text only. Do NOT run it locally.

The probe executes remotely and should collect:

- execution target marker
- hostname
- Python version
- platform/architecture
- CPU count
- total and available RAM
- disk capacity/free space
- CUDA availability
- GPU name
- GPU count
- torch version
- CUDA runtime reported by torch
- `nvidia-smi` summary when available
- package versions (without installing anything) for:
  - torch
  - transformers
  - datasets
  - accelerate
  - peft
  - trl
  - bitsandbytes
  - unsloth
  - unsloth_zoo
  - scipy/numpy
  - jupyter/ipykernel
- current working directory
- contents of /content at shallow depth
- Google Drive mount state
- network/download viability via a lightweight HEAD/metadata request if already permitted
- HF token presence as BOOLEAN ONLY; never print token
- WANDB token presence as BOOLEAN ONLY
- Kaggle credential presence as BOOLEAN ONLY

Execute only through:

```bash
bash scripts/colab_remote_exec.sh <probe-file.py> "$COLAB_SESSION"
```

Store the returned output under a local evidence directory such as:

```text
training_evidence/colab/<timestamp-or-session>/
```

using shell redirection from the `colab exec` command.

## 5. Repository assessment

Inspect the local repository without executing its Python:

```bash
git status --short --branch
git log -5 --oneline
find nb kaggle scripts python_scripts tests -maxdepth 2 -type f | sort
find . -maxdepth 2 -name '*.ipynb' -o -name '*.py' | wc -l
```

Use `rg`, `grep`, `jq`, `sed`, `awk` to statically inspect candidate notebooks.

Identify:

1. notebooks suitable for very small SFT smoke tests
2. notebooks suitable for LoRA/QLoRA
3. notebooks suitable for GRPO/RL
4. notebooks requiring A100/H100-class hardware
5. notebooks explicitly tailored for DGX Spark
6. notebooks likely compatible with the currently assigned Colab GPU
7. notebooks unsuitable for automated coordinator use without adaptation

Do not execute notebooks yet.

## 6. Evaluate current workdir as an OpenTraining backend

Assess whether the current repo/session can satisfy:

```text
probe()
capabilities()
submit()
status()
logs()
cancel()
artifacts()
cleanup()
```

For each operation return:

```text
PASS
PARTIAL
MISSING
```

with the concrete implementation already available and the smallest required adaptation.

## 7. Training Coordinator readiness

Evaluate this session against the progressive training strategy:

```text
many short exploratory runs
→ candidate elimination
→ fewer longer runs
→ hardware promotion
→ final validation
```

Determine what is possible RIGHT NOW with one existing Colab session.

Specifically assess:

- sequential short-run sweeps inside one runtime
- checkpoint/artifact isolation per candidate
- early stopping
- per-candidate deterministic manifests
- reusable model/dataset cache between short runs
- resetting GPU state between candidates
- preventing cross-run contamination
- remote cancellation
- session persistence/reclamation risk
- artifact download/recovery
- coordinator restart/reconnection
- reproducibility
- compute-unit awareness

Do not assume parallelism where only one GPU runtime exists.

## 8. Identify adaptations

Prioritize adaptations by value.

Consider at minimum:

- standard `training_jobs/` directory
- standard `manifests/`
- standard remote workspace under `/content/opentraining/`
- `runs/<job-id>/` isolation
- artifact schema shared with Kaggle/Spark
- remote environment bootstrap
- dependency lock/fingerprint
- model/dataset cache strategy
- resumable checkpoint protocol
- structured metrics JSONL
- JobSpec → notebook parameter injection
- notebook → headless training-script extraction where useful
- static notebook compatibility metadata
- accelerator requirement metadata
- run cancellation marker
- remote health/progress heartbeat
- final artifact retrieval
- remote cleanup
- evidence ledger integration

Do not install or modify anything in this discovery pass unless required only to create harmless local text/evidence files.

## 9. Use-case map

Return concrete use cases under these categories:

### Immediately achievable
What can be done using the current repo + current live Colab session with no architectural changes?

### Small adaptations
What becomes possible with a small amount of adapter/manifest work?

### Coordinator-ready
What is required before autonomous successive-halving / multi-candidate optimization?

### Cross-provider
What can be standardized so the SAME JobSpec works on Colab and Kaggle?

### Spark promotion
Which outputs from Colab should become the evidence used to decide whether a candidate is promoted to DGX Spark?

## 10. No real training yet

This pass must NOT start a model training run.

It MAY execute only the harmless remote capability probe.

Do not install packages.
Do not download large models.
Do not mutate the Colab environment.
Do not stop the session.
Do not provision another session.

## 11. Final report

Return exactly:

```text
A. SELECTED SESSION
name:
status:
last activity:
runtime URL:
compute-unit state:

B. REMOTE HARDWARE
GPU:
GPU memory:
CPU:
RAM:
disk:
CUDA:
torch:

C. REMOTE SOFTWARE
package/version matrix:

D. REPO STATE
branch:
commit:
dirty?:
notebook inventory summary:

E. OPENTRAINING BACKEND CONTRACT
probe:
capabilities:
submit:
status:
logs:
cancel:
artifacts:
cleanup:

F. CURRENTLY ACHIEVABLE
concrete workflows possible now:

G. CURRENT LIMITATIONS
hard limits:
soft limits:

H. NOTEBOOK COMPATIBILITY
best immediate candidates:
requires adaptation:
incompatible with current accelerator:

I. TRAINING-COORDINATOR READINESS
sequential short sweeps:
candidate isolation:
early stopping:
metrics:
artifact handling:
reconnection:
resume:
overall:

J. ADAPTATION PLAN
P0:
P1:
P2:

K. USE CASES
immediate:
near-term:
advanced:

L. COLAB/KAGGLE/SPARK DIVISION
what should run on Colab:
what should run on Kaggle:
what should be promoted to Spark:

M. NEXT EXPERIMENT
Recommend exactly ONE smallest, high-information remote training smoke test.
Do not execute it.
```

The report should distinguish observed facts from recommendations.
