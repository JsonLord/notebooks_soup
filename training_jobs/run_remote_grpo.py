#!/usr/bin/env python3
"""Authoritative LOCAL orchestration entry point for a remote Colab GRPO run.

This is a controller/orchestrator — never training code:
  * never imports torch / transformers / trl / peft / datasets / accelerate
  * never executes GRPO, notebooks, or training locally
  * performs remote execution ONLY through the injected transport, which shells
    the Colab CLI in the foreground (exec_file / download)
  * owns NO shutdown path of its own: automatic session teardown goes
    exclusively through the guarded artifact transaction
    (guarded_stop / reconcile_stop / operator_force_stop). A direct stop
    subprocess is forbidden and statically tested (§19 anti-bypass).

Lifecycle:
  validate RunSpec -> session gate -> remote GRPO exec -> remote
  artifact_produce.py exec -> foreground archive+SHA download -> local SHA ->
  archive/inventory verification -> artifact_transaction.json -> guarded stop

Runner state is persisted atomically to remote_grpo_run.json (no secrets) so
--resume continues artifact retrieval/stop WITHOUT re-running training.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from typing import Callable, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import artifact_transaction as at  # noqa: E402

REPO_ROOT = os.path.dirname(_HERE)
STATE_FILENAME = "remote_grpo_run.json"
PRODUCER_SCRIPT = os.path.join(_HERE, "artifact_produce.py")
LOG_DIRNAME = "logs"

TRAINING_PASS = "TRAINING_PASS"
TRAINING_FAIL = "TRAINING_FAIL"
TRAINING_SKIPPED_RESUME = "TRAINING_SKIPPED_RESUME"

STATUS_UNTRACKED = "REMOTE_JOB_SOURCE_UNTRACKED"
STATUS_NO_SESSION = "COLAB_REMOTE_REQUIRED"
STATUS_RESUME_MISMATCH = "RESUME_SPEC_MISMATCH"
STATUS_STATE_EXISTS = "STATE_EXISTS_USE_RESUME"


def default_is_tracked(path: str, repo_root: str = REPO_ROOT) -> bool:
    """§5 tracked-source gate: git ls-files --error-unmatch <path>."""
    try:
        r = subprocess.run(["git", "ls-files", "--error-unmatch", path],
                           cwd=repo_root, capture_output=True, text=True, timeout=60)
    except Exception:
        return False
    return r.returncode == 0


@dataclasses.dataclass(frozen=True)
class RemoteGRPORunSpec:
    """Explicit run identity. No implicit defaults for identity-bearing fields."""

    session: str
    job_id: str
    attempt_id: str
    run_instance: str
    remote_job_source: str          # local, GIT-TRACKED script path executed remotely
    remote_run_dir: str
    remote_archive: str
    local_evidence_dir: str
    remote_env: dict                # explicit non-secret env (identity ids)
    expected_inventory: tuple = at.GRPO_EXPECTED_INVENTORY
    hf_token_env_var: str = "HF_TOKEN"

    def __post_init__(self):
        self._require("session", self.session)
        self._require("job_id", self.job_id)
        self._require("run_instance", self.run_instance)
        self._require("remote_job_source", self.remote_job_source)
        self._require("local_evidence_dir", self.local_evidence_dir)
        if str(self.attempt_id).strip().lower() in at.FORBIDDEN_ATTEMPT:
            raise at.SpecError(f"attempt_id {self.attempt_id!r} is a forbidden default/stale value")
        self._require("attempt_id", self.attempt_id)
        if not os.path.isabs(self.remote_run_dir):
            raise at.SpecError("remote_run_dir must be absolute")
        # Path rules are enforced by reusing the authoritative TransactionSpec.
        tspec = self.to_transaction_spec()
        # remote_env must carry the identity ids explicitly and no token.
        if not self.remote_env:
            raise at.SpecError("remote_env is required and must be explicit")
        expected_ids = {
            "JOB_ID": self.job_id,
            "ATTEMPT_ID": self.attempt_id,
            "RUN_INSTANCE": self.run_instance,
            "REMOTE_RUN_DIR": self.remote_run_dir,
        }
        for k, v in expected_ids.items():
            if k not in self.remote_env:
                raise at.SpecError(f"remote_env must explicitly contain {k}")
            if str(self.remote_env[k]) != str(v):
                raise at.SpecError(f"remote_env[{k!r}] conflicts with spec identity")
        for k in self.remote_env:
            if k.upper() in ("HF_TOKEN", "TOKEN", "SECRET", "PASSWORD", "AUTH"):
                raise at.SpecError("remote_env must not carry credential values; "
                                   "tokens are injected at exec time only")
        object.__setattr__(self, "_tspec", tspec)

    @staticmethod
    def _require(name: str, value) -> None:
        if value is None or not str(value).strip():
            raise at.SpecError(f"field {name!r} must be non-empty and explicit")
        if any(c.isspace() for c in str(value)):
            raise at.SpecError(f"field {name!r} must not contain whitespace")

    @property
    def remote_sha_file(self) -> str:
        return self.remote_archive + ".sha256"

    def to_transaction_spec(self) -> "at.TransactionSpec":
        return at.grpo_spec(
            job_id=self.job_id,
            attempt_id=self.attempt_id,
            run_instance=self.run_instance,
            remote_run_dir=self.remote_run_dir,
            remote_archive=self.remote_archive,
            local_evidence_dir=self.local_evidence_dir,
            expected_inventory=self.expected_inventory,
        )

    def state_fields(self) -> dict:
        """Redacted identity fields persisted for resume (no env VALUES)."""
        return {
            "session": self.session,
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
            "run_instance": self.run_instance,
            "remote_job_source": self.remote_job_source,
            "remote_run_dir": self.remote_run_dir,
            "remote_archive": self.remote_archive,
            "local_evidence_dir": self.local_evidence_dir,
            "remote_env_keys": sorted(self.remote_env.keys()),
            "expected_inventory": list(self.expected_inventory),
        }


def _save_state(path: str, state: dict) -> None:
    at.write_journal(path, state)


def exec_env_for(spec: RemoteGRPORunSpec, env: dict) -> tuple:
    """Identity env + optional HF token pulled from controller env at exec time.
    Returns (env_map, hf_token_present). Token value is never stored/logged."""
    out = dict(spec.remote_env)
    tok = (env or {}).get(spec.hf_token_env_var, "")
    present = bool(tok)
    if present:
        out[spec.hf_token_env_var] = tok
    return out, present


def run_run(spec: RemoteGRPORunSpec, transport, *,
            env: Optional[dict] = None, is_tracked: Callable[[str], bool] = default_is_tracked,
            resume: bool = False) -> dict:
    """Drive the guarded remote GRPO lifecycle. Pure orchestration."""
    env = dict(env if env is not None else os.environ)
    tspec = spec.to_transaction_spec()
    os.makedirs(spec.local_evidence_dir, exist_ok=True)
    logs_dir = os.path.join(spec.local_evidence_dir, LOG_DIRNAME)
    os.makedirs(logs_dir, exist_ok=True)
    journal_path = at.journal_path_for(tspec)
    state_path = os.path.join(spec.local_evidence_dir, STATE_FILENAME)

    state = at.read_journal(state_path)
    fields = spec.state_fields()
    if resume and state:
        if state.get("spec") != fields:
            return {"status": STATUS_RESUME_MISMATCH, "training_status": None,
                    "artifact_transaction_status": None, "stop_status": "SESSION_STOP_BLOCKED",
                    "journal_path": journal_path}
    elif not resume and state:
        # Never silently double-run training that a previous process started.
        return {"status": STATUS_STATE_EXISTS, "training_status": None,
                "artifact_transaction_status": None, "stop_status": "SESSION_STOP_BLOCKED",
                "journal_path": journal_path}
    if not state:
        state = {"job_id": spec.job_id, "attempt_id": spec.attempt_id,
                 "run_instance": spec.run_instance, "session": spec.session,
                 "remote_job_source": spec.remote_job_source, "spec": fields,
                 "training_started": False, "training_exit_code": None,
                 "producer_started": False, "transaction_status": None,
                 "journal_path": journal_path, "stop_status": None}
        _save_state(state_path, state)

    result = {"status": None, "training_status": None,
              "artifact_transaction_status": None, "stop_status": None,
              "journal_path": journal_path, "state_path": state_path,
              "remote_exec_attempted": False}

    # ---- §14 resume: transaction already PASS -> reconcile stop, no training
    if resume and state.get("transaction_status") == at.ARTIFACT_TRANSACTION_PASS:
        rec = at.reconcile_stop(transport, journal_path)
        if rec["session_stopped"]:
            stop_status = "SESSION_STOP_RECONCILED"
        elif rec["session_active"] is False:
            stop_status = "SESSION_ALREADY_TORN_DOWN"
        else:
            stop_status = "SESSION_STOP_FAIL"
        state["stop_status"] = stop_status
        _save_state(state_path, state)
        result.update({"status": "RESUMED_STOP_RECONCILED", "training_status": TRAINING_SKIPPED_RESUME,
                       "artifact_transaction_status": at.ARTIFACT_TRANSACTION_PASS,
                       "stop_status": stop_status})
        return result

    # ---- 1) GRPO training (executed at most once across resumes)
    if not state.get("training_started"):
        # §5 gate BEFORE any remote execution: tracked job source only.
        if not is_tracked(spec.remote_job_source):
            state["pre_exec_gate"] = STATUS_UNTRACKED
            _save_state(state_path, state)
            result["status"] = STATUS_UNTRACKED
            result["stop_status"] = "SESSION_STOP_BLOCKED"
            return result
        # §6 session gate: attach only to an existing named session.
        if not transport.session_active():
            state["pre_exec_gate"] = STATUS_NO_SESSION
            _save_state(state_path, state)
            result["status"] = STATUS_NO_SESSION
            result["stop_status"] = "SESSION_STOP_BLOCKED"
            return result
        job_env, hf_present = exec_env_for(spec, env)
        state["hf_token_present"] = hf_present
        rc = transport.exec_file(spec.remote_job_source, env=job_env,
                                 output_path=os.path.join(logs_dir, "grpo_exec.log"))
        state["training_started"] = True
        state["training_exit_code"] = rc
        result["remote_exec_attempted"] = True
        _save_state(state_path, state)
        result["training_status"] = TRAINING_PASS if rc == 0 else TRAINING_FAIL
    else:
        prev = state.get("training_exit_code")
        result["training_status"] = (TRAINING_PASS if prev == 0 else TRAINING_FAIL) if prev is not None else TRAINING_SKIPPED_RESUME

    # ---- 2) canonical artifact production (PASS or FAIL — failure evidence too)
    if not state.get("producer_started"):
        if transport.session_active():
            prod_ids = dict(spec.remote_env)
            # the producer also needs the explicit archive path (§8)
            prod_ids["REMOTE_ARCHIVE"] = spec.remote_archive
            transport.exec_file(PRODUCER_SCRIPT, env=prod_ids,
                                output_path=os.path.join(logs_dir, "artifact_produce.log"))
            state["producer_started"] = True
            result["remote_exec_attempted"] = True
        else:
            state["producer_gate"] = STATUS_NO_SESSION
        _save_state(state_path, state)

    # ---- 3) guarded artifact transaction (downloads are the producer proof)
    journal = at.run_transaction(tspec, transport,
                                 producer_check_mode="download_authoritative")
    state["transaction_status"] = journal["status"]
    _save_state(state_path, state)
    result["artifact_transaction_status"] = journal["status"]
    result["artifact_transaction_failure_code"] = journal.get("failure_code")
    result["training_and_artifact_are_separate"] = True

    # ---- 4) shutdown ONLY through the guarded transaction
    stop = at.guarded_stop(transport, journal_path)
    if stop["stop_allowed"]:
        state["stop_status"] = "SESSION_STOP_PASS" if stop["session_stopped"] else "SESSION_STOP_FAIL"
    else:
        state["stop_status"] = "SESSION_STOP_BLOCKED"
    _save_state(state_path, state)
    result["stop_status"] = state["stop_status"]
    result["status"] = ("ORCHESTRATION_COMPLETE" if journal["status"] == at.ARTIFACT_TRANSACTION_PASS
                        else "ARTIFACT_TRANSACTION_FAIL")
    return result


def operator_force_stop(spec: RemoteGRPORunSpec, transport, reason: str) -> dict:
    """§15 explicit emergency path. Requires non-empty reason; never success."""
    if not str(reason).strip():
        raise at.SpecError("--operator-force-stop requires a non-empty --reason")
    journal_path = at.journal_path_for(spec.to_transaction_spec())
    return at.operator_force_stop(transport, journal_path, str(reason).strip())


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_remote_grpo.py",
                                description="Guarded local orchestrator for a remote GRPO run. "
                                            "Never provisions compute; never stops sessions directly.")
    p.add_argument("--session", required=True)
    p.add_argument("--job-id", required=True, dest="job_id")
    p.add_argument("--attempt-id", required=True, dest="attempt_id")
    p.add_argument("--run-instance", required=True, dest="run_instance")
    p.add_argument("--job-source", required=True, dest="job_source",
                   help="GIT-TRACKED local script executed remotely via colab exec")
    p.add_argument("--remote-run-dir", required=True, dest="remote_run_dir")
    p.add_argument("--remote-archive", required=True, dest="remote_archive")
    p.add_argument("--evidence-dir", required=True, dest="evidence_dir")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--operator-force-stop", action="store_true", dest="force_stop")
    p.add_argument("--reason", default="")
    return p


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    spec = RemoteGRPORunSpec(
        session=args.session, job_id=args.job_id, attempt_id=args.attempt_id,
        run_instance=args.run_instance, remote_job_source=args.job_source,
        remote_run_dir=args.remote_run_dir, remote_archive=args.remote_archive,
        local_evidence_dir=args.evidence_dir,
        remote_env={"JOB_ID": args.job_id, "ATTEMPT_ID": args.attempt_id,
                    "RUN_INSTANCE": args.run_instance, "REMOTE_RUN_DIR": args.remote_run_dir},
    )
    transport = at.ColabTransport(args.session)
    if args.force_stop:
        res = operator_force_stop(spec, transport, args.reason)
        print(json.dumps(res, sort_keys=True))
        return 2
    res = run_run(spec, transport, resume=args.resume)
    print(json.dumps(res, sort_keys=True))
    return 0 if res.get("status") == "ORCHESTRATION_COMPLETE" else 3


if __name__ == "__main__":
    raise SystemExit(main())
