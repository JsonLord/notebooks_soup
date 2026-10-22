# Colab-only Execution Policy

This repository is a local source/work tree. Training execution happens only on a selected existing Google Colab runtime.

## Invariant
Notebook and training code must never execute in the controller terminal.

Set COLAB_SESSION to the existing session and verify it with colab status. If status fails, stop. There is no local fallback.

Use scripts/colab_remote_exec.sh to forward .py or .ipynb source to colab exec.

Any local notebook/training execution is LOCAL_EXECUTION_POLICY_VIOLATION and requires an immediate hard stop.

Do not automatically provision a new Colab runtime with colab new or colab run unless the operator explicitly requests it.