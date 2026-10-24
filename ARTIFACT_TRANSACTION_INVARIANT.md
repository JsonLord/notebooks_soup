# Artifact Transaction Invariant (production)

Qualified on CPU: `ARTIFACT_TRANSACTION_PASS`. Implemented for the production
GRPO remote-run controller in `training_jobs/artifact_transaction.py`. The
remote producer that writes the durable sibling checksum is
`training_jobs/artifact_produce.py`.

## Pipeline

```
remote producer
    -> archive file (absolute REMOTE_ARCHIVE)
    -> sibling <REMOTE_ARCHIVE>.sha256 file (durable, remote filesystem)
    -> foreground `colab download` of BOTH
    -> local non-empty checks
    -> local SHA computation
    -> remote-FILE SHA comparison
    -> archive validation / inventory
    -> ONLY THEN session shutdown
```

## Final invariant

```
TRANSFER_VERIFIED =
      DOWNLOAD_COMPLETE
  AND REMOTE_SHA_AVAILABLE
  AND LOCAL_SHA_AVAILABLE
  AND SHA_MATCH

ARTIFACT_TRANSACTION_COMPLETE =
      TRANSFER_VERIFIED
  AND ARCHIVE_INVENTORY_OK

SESSION_STOP_ALLOWED =
      ARTIFACT_TRANSACTION_COMPLETE
```

## Rules

- **stdout is diagnostic only.** `BUNDLE_DONE` / `PRODUCE_DONE` / any printed
  SHA confer ZERO authority. They may be logged; they are never consulted.
- **Durable remote files are authoritative.** The expected checksum is read ONLY
  from the downloaded sibling `<archive>.sha256` file.
- **Empty operands never match.** `SHA_MATCH` is fail-closed and requires both
  SHAs to be present and valid 64-hex. `"" == ""` is NOT a match (this closes
  the prior controller defect that prematurely stopped a session).
- **Session termination is gated.** Only `guarded_stop()` may stop a session,
  and only after re-reading a persisted journal whose status is
  `ARTIFACT_TRANSACTION_PASS` with every required field true. `operator_force_stop`
  is a separately-named emergency path that never records success.
- **Identifiers are explicit.** `job_id, attempt_id, run_instance, remote_run_dir,
  remote_archive, remote_sha_file, local_evidence_dir` are always passed in; no
  default/stale attempt name and no path derived one directory deeper is allowed.
- **Transaction journal** `artifact_transaction.json` is written atomically and
  stores no secrets — hashes and inventory only.

## Preserved qualifications (not collapsed)

- T4 GRPO kernel: PASS (reproduced twice)
- T4 GRPO infrastructure: PASS
- attempt-4 structured archive: LOST with reclaimed VM (not recovered)
- CPU artifact transport transaction: PASS

## Crash/restart boundaries (§16)

A archive made, sha not yet -> do not stop.
B archive+sha made, not downloaded -> resume download.
C archive downloaded, sha not -> resume sha download.
D both downloaded, unverified -> verify locally.
E SHA matches, inventory not checked -> check archive/inventory.
F transaction complete, dies before stop -> re-read persisted PASS, then stop.
G stop sent, dies before record -> reconcile session; never start a new attempt.
