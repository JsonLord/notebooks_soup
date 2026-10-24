#!/usr/bin/env python3
"""Guarded artifact-transaction controller (local side).

Qualified contract (ARTIFACT_TRANSACTION_PASS, proven on CPU) implemented here
for the production GRPO remote-run path:

    remote producer -> archive file -> sibling <archive>.sha256 file
    -> foreground download of BOTH -> local non-empty checks -> local SHA
    -> remote-FILE SHA comparison -> archive validation/inventory
    -> ONLY THEN session shutdown.

Design invariants (see ARTIFACT_TRANSACTION_INVARIANT.md):
  * stdout is DIAGNOSTIC ONLY; durable remote files are the authority.
  * empty operands NEVER match (SHA_MATCH is fail-closed).
  * session termination is gated by ARTIFACT_TRANSACTION_COMPLETE.
  * identifiers are explicit; no default/stale attempt may leak.

Stdlib-only and dependency-free. The transport is injectable so the contract is
testable deterministically WITHOUT any GPU / GRPO / dataset / model / HF Sandbox
/ Colab session. No secrets are ever written to the transaction journal.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Optional

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Sentinel attempt names that must never be accepted as a resolved identity.
FORBIDDEN_ATTEMPT = {"", "attempt", "attempt-1", "latest", "default", "current", "none", "null"}


class SpecError(Exception):
    """Invalid or non-explicit transaction identity/paths."""


class TransportError(Exception):
    """Remote transport failure."""


# ---------------------------------------------------------------------------
# Failure codes (§13)
# ---------------------------------------------------------------------------
REMOTE_ARCHIVE_MISSING = "REMOTE_ARCHIVE_MISSING"
REMOTE_SHA_MISSING = "REMOTE_SHA_MISSING"
REMOTE_SHA_INVALID = "REMOTE_SHA_INVALID"
DOWNLOAD_ARCHIVE_FAILED = "DOWNLOAD_ARCHIVE_FAILED"
DOWNLOAD_SHA_FAILED = "DOWNLOAD_SHA_FAILED"
LOCAL_ARCHIVE_EMPTY = "LOCAL_ARCHIVE_EMPTY"
LOCAL_SHA_FILE_EMPTY = "LOCAL_SHA_FILE_EMPTY"
LOCAL_SHA_UNAVAILABLE = "LOCAL_SHA_UNAVAILABLE"
SHA_MISMATCH = "SHA_MISMATCH"
ARCHIVE_INVALID = "ARCHIVE_INVALID"
ARCHIVE_TRAVERSAL = "ARCHIVE_TRAVERSAL"
INVENTORY_MISMATCH = "INVENTORY_MISMATCH"
ARTIFACT_TRANSACTION_PASS = "ARTIFACT_TRANSACTION_PASS"


# ---------------------------------------------------------------------------
# Explicit transaction identity (§2, §3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TransactionSpec:
    job_id: str
    attempt_id: str
    run_instance: str
    remote_run_dir: str
    remote_archive: str
    local_evidence_dir: str
    expected_inventory: tuple = ()
    remote_sha_file: str = ""

    def __post_init__(self):
        self._validate()

    def _require_explicit(self, name: str, value: str):
        if value is None or not str(value).strip():
            raise SpecError(f"identifier {name!r} must be non-empty and explicit")
        if any(c.isspace() for c in str(value)):
            raise SpecError(f"identifier {name!r} must not contain whitespace")

    def _validate(self):
        self._require_explicit("job_id", self.job_id)
        self._require_explicit("run_instance", self.run_instance)
        self._require_explicit("remote_run_dir", self.remote_run_dir)
        self._require_explicit("remote_archive", self.remote_archive)
        self._require_explicit("local_evidence_dir", self.local_evidence_dir)
        # §2: reject stale/default attempt names.
        if self.attempt_id.strip().lower() in FORBIDDEN_ATTEMPT:
            raise SpecError(f"attempt_id {self.attempt_id!r} is a forbidden default/stale value")
        self._require_explicit("attempt_id", self.attempt_id)
        # §3: remote archive must be an absolute path, not the run dir, and not
        # derived to sit one directory deeper.
        if not os.path.isabs(self.remote_archive):
            raise SpecError("remote_archive must be an absolute path")
        if os.path.normpath(self.remote_archive) == os.path.normpath(self.remote_run_dir):
            raise SpecError("remote_archive must not equal remote_run_dir")
        # §4: sha file is exactly <archive>.sha256 unless caller overrode it.
        if not self.remote_sha_file:
            object.__setattr__(self, "remote_sha_file", self.remote_archive + ".sha256")
        if self.remote_sha_file != self.remote_archive + ".sha256":
            raise SpecError("remote_sha_file must be '<remote_archive>.sha256'")

    # Local landing paths (basename preserved so both downloads are distinct).
    @property
    def local_archive(self) -> str:
        return os.path.join(self.local_evidence_dir, os.path.basename(self.remote_archive))

    @property
    def local_sha_file(self) -> str:
        return os.path.join(self.local_evidence_dir, os.path.basename(self.remote_sha_file))


# Production GRPO inventory. Kept distinct from the CPU qualification fixture so
# the GRPO transport contract never hardcodes the demo inventory (§10, §19).
GRPO_EXPECTED_INVENTORY = (
    "manifest.json",
    "environment.json",
    "hardware.json",
    "sandbox.json",
    "auth.json",
    "grpo_api_preflight.json",
    "metrics.jsonl",
    "results.json",
    "heartbeat.json",
)


def grpo_spec(*, job_id: str, attempt_id: str, run_instance: str,
              remote_run_dir: str, remote_archive: str, local_evidence_dir: str,
              expected_inventory=GRPO_EXPECTED_INVENTORY) -> TransactionSpec:
    """Build + validate an explicit GRPO transaction spec (§2/§8)."""
    return TransactionSpec(
        job_id=job_id,
        attempt_id=attempt_id,
        run_instance=run_instance,
        remote_run_dir=remote_run_dir,
        remote_archive=remote_archive,
        local_evidence_dir=local_evidence_dir,
        expected_inventory=tuple(expected_inventory),
    )


# ---------------------------------------------------------------------------
# Pure helpers (directly unit-testable)
# ---------------------------------------------------------------------------
def parse_remote_sha(text: str) -> Optional[str]:
    """Accept `<sha>  <name>` (sha256sum) or a bare `<sha>` token; validate.
    Returns None when invalid/empty (§7). Never trusts stdout."""
    if text is None:
        return None
    s = text.strip().replace("\r", "")
    if not s:
        return None
    first = s.split()[0]
    return first if SHA256_RE.match(first) else None


def sha_match(remote_sha: Optional[str], local_sha: Optional[str]) -> bool:
    """Fail-closed comparison. Empty/None operands NEVER match (§9)."""
    if not remote_sha or not local_sha:
        return False
    return remote_sha == local_sha


def local_sha256(path: str) -> Optional[str]:
    """Compute + validate a 64-hex SHA-256 over a file, or None (§8)."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    d = h.hexdigest()
    return d if SHA256_RE.match(d) else None


def member_is_safe(name: str) -> bool:
    """Reject absolute paths, drive letters, and any '..' component (§10)."""
    norm = name.replace("\\", "/")
    if norm.startswith("/"):
        return False
    if re.match(r"^[A-Za-z]:[\\/]", norm):
        return False
    for part in norm.split("/"):
        if part == "..":
            return False
    return True


def inventory_ok(observed: list, expected) -> bool:
    if not expected:
        return True
    for e in expected:
        if not any(o == e or o.endswith("/" + e) for o in observed):
            return False
    return True


def verify_archive(local_archive: str, evidence_dir: str) -> dict:
    """Validate gzip/tar integrity, member safety, extractability, and inventory
    containment. Never extracts outside evidence_dir (§10). Returns a dict."""
    os.makedirs(evidence_dir, exist_ok=True)
    result = {"integrity_ok": False, "traversal": False, "observed_inventory": [],
              "code": None}
    try:
        with tarfile.open(local_archive, "r:gz") as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            result["observed_inventory"] = [m.name for m in members]
            for m in members:
                if not member_is_safe(m.name):
                    result["traversal"] = True
                    result["code"] = ARCHIVE_TRAVERSAL if m.name.replace("\\", "/").find("..") >= 0 or m.name.startswith("/") else ARCHIVE_INVALID
                    return result
            extract_root = os.path.realpath(evidence_dir) + os.sep
            # Defense-in-depth extraction into a temp dir strictly under evidence.
            with tempfile.TemporaryDirectory(dir=evidence_dir) as td:
                real_td = os.path.realpath(td)
                for m in members:
                    target = os.path.realpath(os.path.join(real_td, m.name.replace("\\", "/")))
                    if not target.startswith(real_td + os.sep):
                        result["traversal"] = True
                        result["code"] = ARCHIVE_TRAVERSAL
                        return result
                try:
                    tf.extractall(real_td, filter="data")
                except TypeError:  # older tarfile without filter kwarg
                    tf.extractall(real_td)
    except (tarfile.TarError, OSError, EOFError, ValueError):
        result["integrity_ok"] = False
        result["code"] = ARCHIVE_INVALID
        return result
    result["integrity_ok"] = True
    return result


# ---------------------------------------------------------------------------
# Transport (§6 foreground downloads; §12 single stop authority)
# ---------------------------------------------------------------------------
class RemoteTransport:
    def remote_exists(self, path: str) -> bool:
        raise NotImplementedError

    def remote_size(self, path: str) -> int:
        raise NotImplementedError

    def download(self, remote_path: str, local_path: str) -> bool:
        raise NotImplementedError

    def session_active(self) -> bool:
        raise NotImplementedError

    def stop_session(self) -> bool:
        raise NotImplementedError


class ColabTransport(RemoteTransport):
    """Real Colab CLI transport. NOT exercised by unit tests."""

    def __init__(self, session: str):
        self.session = session

    def _cli(self, args, timeout=1800):
        return subprocess.run(["colab", *args], capture_output=True, text=True, timeout=timeout)

    def remote_exists(self, path: str) -> bool:
        # Existence is proven by a successful foreground download postcondition
        # instead of a stdout probe; this cheap marker avoids stdout authority.
        return False

    def remote_size(self, path: str) -> int:
        return -1

    def download(self, remote_path: str, local_path: str) -> bool:
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        r = self._cli(["download", "-s", self.session, remote_path, local_path])
        return r.returncode == 0 and os.path.exists(local_path) and os.path.getsize(local_path) > 0

    def session_active(self) -> bool:
        r = self._cli(["status", "-s", self.session])
        return r.returncode == 0

    def stop_session(self) -> bool:
        r = self._cli(["stop", "-s", self.session])
        return r.returncode == 0

    def exec_file(self, local_path: str, env=None, output_path: str = None) -> int:
        """Foreground remote exec of a local script via the CLI (§7).

        env VALUES are NEVER logged or persisted — only the key COUNT is logged.
        Remote stdout is diagnostic only and is never written into evidence.
        """
        args = ["exec", "-s", self.session, "-f", local_path]
        for k, v in sorted((env or {}).items()):
            args += ["--env", f"{k}={v}"]
        r = self._cli(args)
        if output_path:
            with open(output_path, "a", encoding="utf-8") as fh:
                fh.write(f"[exec] rc={r.returncode} file={os.path.basename(local_path)} "
                         f"env_keys={len(env or {})}\n")
        return r.returncode


def _retry(fn: Callable[[], bool], attempts: int = 3) -> bool:
    """Retry idempotent/safe operations (§17)."""
    for _ in range(max(1, attempts)):
        try:
            if fn():
                return True
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def check_producer_done(spec: TransactionSpec, transport: RemoteTransport) -> dict:
    """§5: producer complete == archive + sha both exist and are non-empty.
    A stdout BUNDLE_DONE sentinel is never consulted (zero authority)."""
    a_exists = transport.remote_exists(spec.remote_archive) and transport.remote_size(spec.remote_archive) > 0
    s_exists = transport.remote_exists(spec.remote_sha_file) and transport.remote_size(spec.remote_sha_file) > 0
    code = None
    if not a_exists:
        code = REMOTE_ARCHIVE_MISSING
    elif not s_exists:
        code = REMOTE_SHA_MISSING
    return {"producer_done": bool(a_exists and s_exists), "code": code}


def download_stage(spec: TransactionSpec, transport: RemoteTransport) -> dict:
    """§6 foreground downloads + local non-empty postconditions. Resumable: if a
    valid local copy already exists, it is accepted (restart-safety §16 B/C/D)."""
    os.makedirs(spec.local_evidence_dir, exist_ok=True)
    out = {"download_complete": False, "code": None, "local_archive_size": 0, "local_sha_size": 0}
    archive_ok = _retry(lambda: transport.download(spec.remote_archive, spec.local_archive))
    if not archive_ok and os.path.exists(spec.local_archive) and os.path.getsize(spec.local_archive) > 0:
        archive_ok = True
    if not archive_ok:
        out["code"] = DOWNLOAD_ARCHIVE_FAILED
        return out
    if not (os.path.exists(spec.local_archive) and os.path.getsize(spec.local_archive) > 0):
        out["code"] = LOCAL_ARCHIVE_EMPTY
        return out
    sha_ok = _retry(lambda: transport.download(spec.remote_sha_file, spec.local_sha_file))
    if not sha_ok and os.path.exists(spec.local_sha_file) and os.path.getsize(spec.local_sha_file) > 0:
        sha_ok = True
    if not sha_ok:
        out["code"] = DOWNLOAD_SHA_FAILED
        return out
    if not (os.path.exists(spec.local_sha_file) and os.path.getsize(spec.local_sha_file) > 0):
        out["code"] = LOCAL_SHA_FILE_EMPTY
        return out
    out["download_complete"] = True
    out["local_archive_size"] = os.path.getsize(spec.local_archive)
    out["local_sha_size"] = os.path.getsize(spec.local_sha_file)
    return out


def sha_stage(spec: TransactionSpec) -> dict:
    """§7/§8/§9: remote SHA read ONLY from the downloaded sibling file; local SHA
    computed; fail-closed match."""
    out = {"remote_sha": None, "local_sha": None, "remote_sha_available": False,
           "local_sha_available": False, "sha_match": False, "code": None}
    try:
        with open(spec.local_sha_file, "r") as fh:
            raw = fh.read()
    except OSError:
        raw = ""
    rsha = parse_remote_sha(raw)
    out["remote_sha"] = rsha
    out["remote_sha_available"] = rsha is not None
    lsha = local_sha256(spec.local_archive)
    out["local_sha"] = lsha
    out["local_sha_available"] = lsha is not None
    if not out["remote_sha_available"]:
        out["code"] = REMOTE_SHA_INVALID
    elif not out["local_sha_available"]:
        out["code"] = LOCAL_SHA_UNAVAILABLE
    else:
        out["sha_match"] = sha_match(rsha, lsha)
        if not out["sha_match"]:
            out["code"] = SHA_MISMATCH
    return out


def validate_stage(spec: TransactionSpec) -> dict:
    """§10 archive integrity + inventory validation."""
    out = {"archive_integrity_ok": False, "archive_inventory_ok": False,
           "observed_inventory": [], "traversal": False, "code": None}
    v = verify_archive(spec.local_archive, spec.local_evidence_dir)
    out["archive_integrity_ok"] = v["integrity_ok"]
    out["traversal"] = v["traversal"]
    out["observed_inventory"] = v["observed_inventory"]
    if v["code"]:
        out["code"] = v["code"]
        return out
    if not v["integrity_ok"]:
        out["code"] = ARCHIVE_INVALID
        return out
    ok = inventory_ok(v["observed_inventory"], spec.expected_inventory)
    out["archive_inventory_ok"] = bool(ok)
    if not ok:
        out["code"] = INVENTORY_MISMATCH
    return out


def compute_gates(download: dict, sha: dict, validate: dict) -> dict:
    """§11 explicit stage split. Any required gate false => stop never allowed."""
    transfer_verified = bool(
        download.get("download_complete")
        and sha.get("remote_sha_available")
        and sha.get("local_sha_available")
        and sha.get("sha_match")
    )
    complete = bool(transfer_verified and validate.get("archive_inventory_ok"))
    return {
        "DOWNLOAD_COMPLETE": bool(download.get("download_complete")),
        "REMOTE_SHA_AVAILABLE": bool(sha.get("remote_sha_available")),
        "LOCAL_SHA_AVAILABLE": bool(sha.get("local_sha_available")),
        "SHA_MATCH": bool(sha.get("sha_match")),
        "ARCHIVE_INVENTORY_OK": bool(validate.get("archive_inventory_ok")),
        "TRANSFER_VERIFIED": transfer_verified,
        "ARTIFACT_TRANSACTION_COMPLETE": complete,
        "SESSION_STOP_ALLOWED": complete,
    }


def _failure_code(download: dict, sha: dict, validate: dict, gates: dict) -> str:
    for stage in (download, sha, validate):
        if stage.get("code"):
            return stage["code"]
    if not gates["TRANSFER_VERIFIED"]:
        return SHA_MISMATCH if not gates["SHA_MATCH"] else LOCAL_SHA_UNAVAILABLE
    if not gates["ARTIFACT_TRANSACTION_COMPLETE"]:
        return INVENTORY_MISMATCH
    return ARTIFACT_TRANSACTION_PASS


# ---------------------------------------------------------------------------
# Transaction journal (§14) — atomic, no secrets
# ---------------------------------------------------------------------------
def journal_path_for(spec: TransactionSpec) -> str:
    return os.path.join(spec.local_evidence_dir, "artifact_transaction.json")


def write_journal(path: str, journal: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".journal.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(journal, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_journal(path: str) -> dict:
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def build_journal(spec: TransactionSpec, producer: dict, download: dict, sha: dict,
                  validate: dict, gates: dict) -> dict:
    status = ARTIFACT_TRANSACTION_PASS if gates["ARTIFACT_TRANSACTION_COMPLETE"] else "FAILED"
    failure_code = None if gates["ARTIFACT_TRANSACTION_COMPLETE"] else _failure_code(download, sha, validate, gates)
    return {
        "job_id": spec.job_id,
        "attempt_id": spec.attempt_id,
        "run_instance": spec.run_instance,
        "remote_run_dir": spec.remote_run_dir,
        "remote_archive": spec.remote_archive,
        "remote_sha_file": spec.remote_sha_file,
        "local_evidence_dir": spec.local_evidence_dir,
        "download_complete": bool(download.get("download_complete")),
        "remote_sha_available": bool(sha.get("remote_sha_available")),
        "local_sha_available": bool(sha.get("local_sha_available")),
        "sha_match": bool(sha.get("sha_match")),
        "archive_inventory_ok": bool(validate.get("archive_inventory_ok")),
        "archive_integrity_ok": bool(validate.get("archive_integrity_ok")),
        "traversal_detected": bool(validate.get("traversal")),
        "transfer_verified": bool(gates["TRANSFER_VERIFIED"]),
        "artifact_transaction_complete": bool(gates["ARTIFACT_TRANSACTION_COMPLETE"]),
        "session_stop_allowed": bool(gates["SESSION_STOP_ALLOWED"]),
        "remote_sha": sha.get("remote_sha"),
        "local_sha": sha.get("local_sha"),
        "archive_size": int(download.get("local_archive_size") or 0) or None,
        "sha_file_size": int(download.get("local_sha_size") or 0) or None,
        "producer_done": bool(producer.get("producer_done")),
        "producer_check_mode": producer.get("mode"),
        "expected_inventory": list(spec.expected_inventory),
        "observed_inventory": validate.get("observed_inventory", []),
        "status": status,
        "failure_code": failure_code,
    }


def run_transaction(spec: TransactionSpec, transport: RemoteTransport, *,
                    journal_path: Optional[str] = None, skip_producer_check: bool = False,
                    producer_check_mode: str = "remote_stat") -> dict:
    """Execute the full guarded transaction and persist the journal.

    Idempotent and resumable (§16): safe to call repeatedly; already-downloaded
    valid local copies are reused. On any unmet gate session_stop_allowed=false.

    producer_check_mode (§9): "remote_stat" verifies the durable remote files
    before download. "download_authoritative" (required for transports without
    reliable remote stat, e.g. ColabTransport) proves producer completion by the
    successful non-empty foreground downloads themselves. skip_producer_check is
    retained ONLY as a deprecated alias for "download_authoritative"; in NO mode
    is producer_done ever prefilled true.
    """
    jp = journal_path or journal_path_for(spec)
    if producer_check_mode not in ("remote_stat", "download_authoritative"):
        raise ValueError("producer_check_mode must be remote_stat or download_authoritative")
    if skip_producer_check:
        producer_check_mode = "download_authoritative"

    def _fail(producer, download):
        sha = {"remote_sha_available": False, "local_sha_available": False, "sha_match": False, "code": None}
        validate = {"archive_integrity_ok": False, "archive_inventory_ok": False, "observed_inventory": [], "traversal": False, "code": None}
        gates = compute_gates(download, sha, validate)
        journal = build_journal(spec, producer, download, sha, validate, gates)
        write_journal(jp, journal)
        return journal

    if producer_check_mode == "download_authoritative":
        download = download_stage(spec, transport)
        done = bool(download.get("download_complete"))
        producer = {"producer_done": done,
                    "code": None if done else download.get("code"),
                    "mode": producer_check_mode}
        if not done:
            return _fail(producer, download)
    else:
        producer = check_producer_done(spec, transport)
        producer["mode"] = producer_check_mode
        if not producer["producer_done"]:
            return _fail(producer, {"download_complete": False, "code": producer["code"]})
        download = download_stage(spec, transport)
    if download["download_complete"]:
        sha = sha_stage(spec)
        if sha.get("sha_match"):
            validate = validate_stage(spec)
        else:
            validate = {"archive_integrity_ok": False, "archive_inventory_ok": False,
                        "observed_inventory": [], "traversal": False, "code": None}
    else:
        sha = {"remote_sha_available": False, "local_sha_available": False, "sha_match": False, "code": None}
        validate = {"archive_integrity_ok": False, "archive_inventory_ok": False, "observed_inventory": [], "traversal": False, "code": None}

    gates = compute_gates(download, sha, validate)
    journal = build_journal(spec, producer, download, sha, validate, gates)
    write_journal(jp, journal)
    return journal


# ---------------------------------------------------------------------------
# Single guarded shutdown authority (§12/§15/§16)
# ---------------------------------------------------------------------------
REQUIRED_TRUE_FIELDS = (
    "download_complete", "remote_sha_available", "local_sha_available",
    "sha_match", "archive_inventory_ok", "artifact_transaction_complete",
    "session_stop_allowed",
)


def evaluate_stop_gate(journal: dict) -> tuple:
    """Reload-and-verify the persisted journal (§15). Returns (allowed, reason)."""
    if not journal:
        return False, "NO_JOURNAL"
    for f in REQUIRED_TRUE_FIELDS:
        if not journal.get(f):
            return False, f"GATE_FALSE:{f}"
    if journal.get("status") != ARTIFACT_TRANSACTION_PASS:
        return False, "STATUS_NOT_PASS"
    if journal.get("failure_code") is not None:
        return False, "FAILURE_CODE_SET"
    return True, "OK"


def guarded_stop(transport: RemoteTransport, journal_path: str) -> dict:
    """THE ONLY automatic shutdown path. Re-derives state from the PERSISTED
    journal (not a transient shell var) and stops only when fully satisfied."""
    journal = read_journal(journal_path)
    allowed, reason = evaluate_stop_gate(journal)
    result = {"stop_allowed": allowed, "reason": reason, "session_stopped": False}
    if not allowed:
        return result
    result["session_stopped"] = bool(transport.stop_session())
    return result


def reconcile_stop(transport: RemoteTransport, journal_path: str) -> dict:
    """§16 G: after a crash between stop and record, reconcile. Never create a
    new training attempt because teardown acknowledgement was lost."""
    journal = read_journal(journal_path)
    allowed, reason = evaluate_stop_gate(journal)
    result = {"stop_allowed": allowed, "reason": reason, "session_active": None, "session_stopped": False}
    if not allowed:
        # Cannot safely stop; do NOT treat as completed; do NOT start a new attempt.
        try:
            result["session_active"] = transport.session_active()
        except Exception:
            pass
        return result
    active = transport.session_active()
    result["session_active"] = active
    if active:
        result["session_stopped"] = bool(transport.stop_session())
    else:
        result["reason"] = "ALREADY_TORN_DOWN"
    return result


def operator_force_stop(transport: RemoteTransport, journal_path: str, reason: str) -> dict:
    """Separately-named EMERGENCY termination (§12). Explicit; never records a
    successful artifact completion."""
    journal = read_journal(journal_path)
    stopped = bool(transport.stop_session())
    journal["status"] = "OPERATOR_FORCED_STOP"
    journal["session_stop_allowed"] = False
    journal["artifact_transaction_complete"] = False
    journal["operator_force_reason"] = reason
    write_journal(journal_path, journal)
    return {"operator_forced": True, "session_stopped": stopped, "reason": reason}
