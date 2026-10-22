#!/usr/bin/env bash
# Source this file in the controller shell before agent work:
#   source scripts/activate_colab_only_guard.sh

_colab_only_block() {
  echo "LOCAL_EXECUTION_POLICY_VIOLATION: local notebook/training execution is forbidden." >&2
  echo "Use: bash scripts/colab_remote_exec.sh <file.py|file.ipynb> [session]" >&2
  return 126
}

python() { _colab_only_block; }
python3() { _colab_only_block; }
ipython() { _colab_only_block; }
jupyter() { _colab_only_block; }
jupyter-lab() { _colab_only_block; }
jupyter-nbconvert() { _colab_only_block; }
papermill() { _colab_only_block; }
pytest() { _colab_only_block; }
torchrun() { _colab_only_block; }
accelerate() { _colab_only_block; }
soup() { _colab_only_block; }
unsloth() { _colab_only_block; }

export -f _colab_only_block python python3 ipython jupyter jupyter-lab jupyter-nbconvert papermill pytest torchrun accelerate soup unsloth
export TRAINING_EXECUTION_MODE=COLAB_ONLY

echo "[colab-only] local Python/Jupyter/training entry points blocked in this shell." >&2
echo "[colab-only] COLAB_SESSION=${COLAB_SESSION:-<unset>}" >&2