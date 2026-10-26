#!/usr/bin/env python3
# LEGACY_DEBUG_HELPER — NOT_PRODUCTION_ARTIFACT_PATH
# Production artifact path: artifact_produce.py + artifact_transaction.py
# (durable archive + sibling .sha256, download-authoritative, guarded stop).
"""Remote READ-ONLY: bundle §13 artifacts excluding multi-GB weight/optimizer
blobs so the archive is small enough to hash + download reliably. NO training."""
import hashlib
import os
import tarfile

JOB_ID = "grpo-smoldataenvs-smoke-001"
# §2/§12 identity-leak guard: explicit GRPO_ATTEMPT required, no stale default.
_ATTEMPT_ENV = (os.environ.get("GRPO_ATTEMPT", "").strip())
if not _ATTEMPT_ENV:
    print("EXECUTION_TARGET=COLAB_REMOTE", flush=True)
    print("ATTEMPT_MISSING=1 set GRPO_ATTEMPT explicitly", flush=True)
    print("NOTE: use training_jobs/artifact_produce.py (guarded contract)", flush=True)
    raise SystemExit(2)
ATTEMPT = _ATTEMPT_ENV
RUN = "/content/opentraining/runs/%s/%s" % (JOB_ID, ATTEMPT)
BUNDLE = "/content/opentraining/grpo_artifacts.tar.gz"
MAXB = 50 * 1024 * 1024

print("EXECUTION_TARGET=COLAB_REMOTE", flush=True)
print("ATTEMPT=%s" % ATTEMPT, flush=True)
print("RUN_EXISTS=%s" % os.path.isdir(RUN), flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


if not os.path.isdir(RUN):
    print("ART_BUNDLE_MISSING=1", flush=True)
    print("ART_BUNDLE_DONE=done", flush=True)
    raise SystemExit

inc, skipped = [], []
for root, _d, files in os.walk(RUN):
    for f in files:
        p = os.path.join(root, f)
        sz = os.path.getsize(p)
        rel = os.path.relpath(p, "/content/opentraining")
        (inc if sz < MAXB else skipped).append((rel, sz))
print("ART_INCLUDED=%d" % len(inc), flush=True)
for rel, sz in sorted(inc):
    print("  FILE %8d %s" % (sz, rel), flush=True)
print("ART_SKIPPED_BIG=%s" % [r for r, s in sorted(skipped)], flush=True)
print("ART_TOTAL_BYTES=%d" % sum(s for _, s in inc), flush=True)
with tarfile.open(BUNDLE, "w:gz") as tf:
    for rel, sz in sorted(inc):
        tf.add(os.path.join("/content/opentraining", rel),
               arcname=os.path.join(JOB_ID, ATTEMPT, *rel.split("/")[2:]))
print("ART_BUNDLE_PATH=%s" % BUNDLE, flush=True)
print("ART_BUNDLE_BYTES=%d" % os.path.getsize(BUNDLE), flush=True)
print("ART_BUNDLE_SHA256=%s" % sha256(BUNDLE), flush=True)
print("ART_BUNDLE_DONE=done", flush=True)
