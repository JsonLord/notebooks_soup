#!/usr/bin/env python3
"""Canonical remote ARTIFACT PRODUCER for the guarded artifact-transaction contract.

Runs REMOTE-ONLY via `colab exec`. It is the durable authority side of the
transport: it produces an archive at an EXPLICIT absolute path and writes a
SIBLING <archive>.sha256 file on the remote filesystem so the controller can
verify integrity WITHOUT trusting stdout. stdout here is DIAGNOSTIC ONLY and
confers ZERO authority (see training_jobs/artifact_transaction.py).

Explicit identifiers via environment; there are NO defaults, so a stale attempt
name cannot leak from prior shell state (contract §2). No model/dataset/GPU/HF
token is needed for transport production.

Env (all required except EXCLUDE_MAX_BYTES):
  JOB_ID, ATTEMPT_ID, RUN_INSTANCE, REMOTE_RUN_DIR, REMOTE_ARCHIVE,
  EXCLUDE_MAX_BYTES (optional, default 50 MiB; files >= this are skipped so the
  archive stays small enough to hash + download reliably).
"""
import hashlib
import os
import sys
import tarfile


def die(code, detail=""):
    print("PRODUCE_ERROR=%s %s" % (code, detail), flush=True)
    raise SystemExit(2)


REQUIRED = ("JOB_ID", "ATTEMPT_ID", "RUN_INSTANCE", "REMOTE_RUN_DIR", "REMOTE_ARCHIVE")
vals = {}
for k in REQUIRED:
    v = os.environ.get(k, "").strip()
    if not v:
        die("MISSING_IDENTIFIER", k)
    if any(c.isspace() for c in v):
        die("INVALID_IDENTIFIER", k)
    vals[k] = v
JOB_ID = vals["JOB_ID"]
ATTEMPT_ID = vals["ATTEMPT_ID"]
RUN_INSTANCE = vals["RUN_INSTANCE"]
REMOTE_RUN_DIR = vals["REMOTE_RUN_DIR"]
REMOTE_ARCHIVE = vals["REMOTE_ARCHIVE"]

if not REMOTE_ARCHIVE.startswith("/"):
    die("REMOTE_ARCHIVE_NOT_ABSOLUTE", REMOTE_ARCHIVE)
if REMOTE_ARCHIVE.rstrip("/") == REMOTE_RUN_DIR.rstrip("/"):
    die("REMOTE_ARCHIVE_IS_RUN_DIR", REMOTE_ARCHIVE)

try:
    EXCLUDE_MAX_BYTES = int(os.environ.get("EXCLUDE_MAX_BYTES", "") or (50 * 1024 * 1024))
except ValueError:
    EXCLUDE_MAX_BYTES = 50 * 1024 * 1024

REMOTE_SHA_FILE = REMOTE_ARCHIVE + ".sha256"

print("EXECUTION_TARGET=COLAB_REMOTE", flush=True)
print("JOB_ID=%s" % JOB_ID, flush=True)
print("ATTEMPT_ID=%s" % ATTEMPT_ID, flush=True)
print("RUN_INSTANCE=%s" % RUN_INSTANCE, flush=True)
print("REMOTE_RUN_DIR=%s" % REMOTE_RUN_DIR, flush=True)
print("REMOTE_ARCHIVE=%s" % REMOTE_ARCHIVE, flush=True)
print("REMOTE_SHA_FILE=%s" % REMOTE_SHA_FILE, flush=True)

if not os.path.isdir(REMOTE_RUN_DIR):
    die("REMOTE_RUN_DIR_MISSING", REMOTE_RUN_DIR)

inc, skipped = [], []
for root, _d, files in os.walk(REMOTE_RUN_DIR):
    for f in files:
        p = os.path.join(root, f)
        sz = os.path.getsize(p)
        rel = os.path.relpath(p, REMOTE_RUN_DIR)
        (inc if sz < EXCLUDE_MAX_BYTES else skipped).append((rel, sz))

print("PRODUCE_INCLUDED=%d" % len(inc), flush=True)
for rel, sz in sorted(inc):
    print("  FILE %8d %s" % (sz, rel), flush=True)
print("PRODUCE_SKIPPED_BIG=%s" % [r for r, _ in sorted(skipped)], flush=True)

os.makedirs(os.path.dirname(REMOTE_ARCHIVE) or "/", exist_ok=True)
with tarfile.open(REMOTE_ARCHIVE, "w:gz") as tf:
    for rel, sz in sorted(inc):
        tf.add(os.path.join(REMOTE_RUN_DIR, rel),
               arcname=os.path.join(JOB_ID, ATTEMPT_ID, *rel.split("/")))

h = hashlib.sha256()
with open(REMOTE_ARCHIVE, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
sha = h.hexdigest()

# DURABLE sibling checksum file. Conventional `sha256sum` form:
#   <sha>  <basename>
sha_body = "%s  %s\n" % (sha, os.path.basename(REMOTE_ARCHIVE))
tmp = REMOTE_SHA_FILE + ".tmp"
with open(tmp, "w") as fh:
    fh.write(sha_body)
os.replace(tmp, REMOTE_SHA_FILE)

# Post-produce durable postconditions (§5): archive + sha both exist, non-empty.
archive_ok = os.path.isfile(REMOTE_ARCHIVE) and os.path.getsize(REMOTE_ARCHIVE) > 0
sha_ok = os.path.isfile(REMOTE_SHA_FILE) and os.path.getsize(REMOTE_SHA_FILE) > 0

print("PRODUCE_ARCHIVE_BYTES=%d" % os.path.getsize(REMOTE_ARCHIVE), flush=True)
print("PRODUCE_SHA_BYTES=%d" % os.path.getsize(REMOTE_SHA_FILE), flush=True)
# Diagnostic only (never authority):
print("PRODUCE_ARCHIVE_OK=%s" % archive_ok, flush=True)
print("PRODUCE_SHA_OK=%s" % sha_ok, flush=True)
print("PRODUCE_DONE=done", flush=True)
