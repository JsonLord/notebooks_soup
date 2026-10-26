# AGENTS.md — Colab-only training workdir

## Hard rule
This repository is a source/work directory. It is NOT a local training runtime.
Notebook code, training Python, training shell commands, CUDA probes, Soup, Unsloth, Transformers/TRL training, Jupyter execution, pytest that imports repo code, torchrun, accelerate, uv run, and direct execution of .py/.ipynb/training .sh files are forbidden in the controller terminal.

Do not bypass this with absolute interpreter paths, heredocs, subprocesses, containers, Makefiles, eval, source, bash/sh wrappers, or alternate Python executables.

## Violation behavior
If notebook/training code is executed locally: STOP immediately; report LOCAL_EXECUTION_POLICY_VIOLATION and the exact command; do not continue until the operator explicitly re-authorizes.

## Required execution path
A named already-running Colab session is mandatory. COLAB_SESSION must identify it.
Before execution, run only the orchestration check: colab status -s "$COLAB_SESSION".
If the session is absent/expired/unreachable: report COLAB_REMOTE_REQUIRED and HARD STOP. There is no local fallback.

Execute notebook/training source only with:
  bash scripts/colab_remote_exec.sh <file.py|file.ipynb> [session]
or the direct equivalent:
  colab exec -s "$COLAB_SESSION" -f <file.py|file.ipynb>

Do not call colab new or colab run unless the operator explicitly authorizes provisioning. This workdir is intended to attach to the selected running Colab runtime.

## Allowed local orchestration
Local commands may inspect/version/transport files: git, gh, ls, find, cat, head, tail, grep, rg, sed, awk, jq, sha256sum, wc, stat, mkdir, cp, mv, tar, and Colab session/status/log/download/stop/exec commands.
These commands must not be used to execute notebook/training code locally.

## LOCAL CONTROLLER PYTHON (narrow exception)
Direct execution of stdlib-only local controller scripts is allowed ONLY when all hold:
stdlib/controller-local dependencies only; no training-framework imports (torch, Transformers, TRL, PEFT, datasets, accelerate, unsloth); no model or dataset loading; no local notebook/training execution; only git/filesystem/Colab orchestration; all remote execution still via colab exec.
Explicitly allowed: training_jobs/artifact_transaction.py, training_jobs/run_remote_grpo.py, and their deterministic controller tests.
This exception does NOT weaken COLAB_ONLY. Training job .py files, notebooks, pytest that imports training code, CUDA probes, and local model/dataset execution remain forbidden.

## Remote proof
Every training job must capture evidence from the remote kernel identifying EXECUTION_TARGET=COLAB_REMOTE, hostname, CUDA availability, and GPU name when present.

## Data policy
D0 remote allowed; D1 only if policy permits; D2/D3 stay local unless explicitly approved. Colab-only execution does not override data-sensitivity policy.