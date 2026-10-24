#!/usr/bin/env python3
"""Remote-only READ-ONLY: bundle a GRPO attempt dir for retrieval + read-only
trl.GRPOConfig signature introspection (NO re-run). Attempt from GRPO_ATTEMPT env."""
import hashlib
import inspect
import os
import tarfile

JOB_ID = "grpo-smoldataenvs-smoke-001"
# §2/§12 identity-leak guard: no stale/default attempt. An explicit GRPO_ATTEMPT
# is required; otherwise refuse (the silent default previously caused archive loss).
import os as _os
_ATTEMPT_ENV = (_os.environ.get("GRPO_ATTEMPT", "").strip())
if not _ATTEMPT_ENV:
    print("EXECUTION_TARGET=COLAB_REMOTE", flush=True)
    print("ATTEMPT_MISSING=1 set GRPO_ATTEMPT explicitly", flush=True)
    print("NOTE: use training_jobs/artifact_produce.py (guarded contract)", flush=True)
    raise SystemExit(2)
ATTEMPT = _ATTEMPT_ENV
RUN = "/content/opentraining/runs/%s/%s" % (JOB_ID, ATTEMPT)
BUNDLE = "/content/opentraining/grpo_bundle.tar.gz"

print("EXECUTION_TARGET=COLAB_REMOTE", flush=True)
print("ATTEMPT=%s" % ATTEMPT, flush=True)
print("RUN_EXISTS=%s" % os.path.isdir(RUN), flush=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

if os.path.isdir(RUN):
    names = []
    for root, _d, files in os.walk(RUN):
        for f in files:
            p = os.path.join(root, f)
            names.append((os.path.relpath(p, "/content/opentraining"), os.path.getsize(p)))
    print("BUNDLE_FILES=%d" % len(names), flush=True)
    for rel, sz in sorted(names):
        print("  FILE %8d %s" % (sz, rel), flush=True)
    with tarfile.open(BUNDLE, "w:gz") as tf:
        tf.add(RUN, arcname=os.path.join(JOB_ID, ATTEMPT))
    print("BUNDLE_PATH=%s" % BUNDLE, flush=True)
    print("BUNDLE_BYTES=%d" % os.path.getsize(BUNDLE), flush=True)
    print("BUNDLE_SHA256=%s" % sha256(BUNDLE), flush=True)
else:
    print("BUNDLE_MISSING=1", flush=True)

try:
    import trl
    from trl import GRPOConfig, GRPOTrainer
    sig = set(inspect.signature(GRPOConfig.__init__).parameters)
    try:
        fields = set(GRPOConfig.__dataclass_fields__.keys())
    except Exception:
        fields = set()
    cfg_fields = sig | fields
    req = ["fp16", "bf16", "use_vllm", "num_generations", "max_steps", "max_completion_length",
           "gradient_checkpointing", "logging_steps", "learning_rate", "temperature", "repetition_penalty"]
    print("TRL_VERSION=%s" % trl.__version__, flush=True)
    print("PREFLIGHT_ACCEPTED=%s" % [f for f in req if f in cfg_fields], flush=True)
    print("PREFLIGHT_UNSUPPORTED=%s" % [f for f in req if f not in cfg_fields], flush=True)
    print("HAS_max_prompt_length=%s" % ("max_prompt_length" in cfg_fields), flush=True)
    tp = set(inspect.signature(GRPOTrainer.__init__).parameters)
    print("TRAINER_HAS=%s" % [k for k in ("model", "reward_funcs", "args", "train_dataset") if k in tp], flush=True)
except Exception as e:
    print("SIG_INTROSPECT_ERR=%s:%s" % (type(e).__name__, str(e)[:120]), flush=True)

print("BUNDLE_DONE=done", flush=True)
