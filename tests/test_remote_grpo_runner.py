#!/usr/bin/env python3
"""Deterministic tests for the guarded remote-GRPO orchestrator (§18-§20).

No Colab, no GPU, no HF, no training framework: a fake transport records every
remote call, and AST scans prove the runner cannot bypass the artifact gate.
Run: /opt/venv/bin/python -m unittest -v tests/test_remote_grpo_runner.py
"""
import ast
import hashlib
import io
import os
import sys
import tarfile
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_JOBS = os.path.join(REPO, "training_jobs")
for _p in (TRAINING_JOBS, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import artifact_transaction as at  # noqa: E402
import run_remote_grpo as rgg  # noqa: E402

RUNNER_PATH = os.path.join(TRAINING_JOBS, "run_remote_grpo.py")
FORBIDDEN_IMPORTS = {"torch", "transformers", "trl", "peft", "datasets",
                     "accelerate", "unsloth"}


def make_tar(members: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in members.items():
            b = data.encode() if isinstance(data, str) else data
            info = tarfile.TarInfo(name=name)
            info.size = len(b)
            tf.addfile(info, io.BytesIO(b))
    return buf.getvalue()


class FakeRunnerTransport(at.RemoteTransport):
    """Virtual remote fs + full call recorder (execs, downloads, stops)."""

    def __init__(self, remote=None, *, exists=True, size=1024, active=True, exec_rc=0):
        self.remote = dict(remote or {})
        self._exists = exists
        self._size = size
        self.active = active
        self.exec_rc = exec_rc
        self.execs = []          # (local_path, env, output_path)
        self.stop_calls = 0
        self.status_calls = 0

    # --- RemoteTransport surface ---
    def remote_exists(self, path):
        return self._exists

    def remote_size(self, path):
        return self._size

    def download(self, remote_path, local_path):
        data = self.remote.get(remote_path)
        if data is None:
            return False
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        with open(local_path, "wb") as fh:
            fh.write(data)
        return True

    def session_active(self):
        self.status_calls += 1
        return self.active

    def stop_session(self):
        self.stop_calls += 1
        self.active = False
        return True

    # --- orchestrator-side surface (mirrors ColabTransport.exec_file) ---
    def exec_file(self, local_path, env=None, output_path=None):
        self.execs.append((local_path, dict(env or {}), output_path))
        return self.exec_rc


def _spec(tmp, **over):
    base = dict(
        session="opentraining-1", job_id="grpo-smoldataenvs-smoke-001",
        attempt_id="attempt-5-t4-orchestrated", run_instance="g-t4-abc123",
        remote_job_source="training_jobs/grpo_smoke.py",
        remote_run_dir="/content/opentraining/grpo-smoldataenvs-smoke-001/attempt-5-t4-orchestrated",
        remote_archive="/content/opentraining/grpo-smoldataenvs-smoke-001/attempt-5-t4-orchestrated.tar.gz",
        local_evidence_dir=tmp,
        remote_env={"JOB_ID": "grpo-smoldataenvs-smoke-001",
                    "ATTEMPT_ID": "attempt-5-t4-orchestrated",
                    "RUN_INSTANCE": "g-t4-abc123",
                    "REMOTE_RUN_DIR": "/content/opentraining/grpo-smoldataenvs-smoke-001/attempt-5-t4-orchestrated"},
    )
    base.update(over)
    return rgg.RemoteGRPORunSpec(**base)


def _remote(spec, members=None):
    """Virtual remote fs: archive + sibling sha file for `spec`."""
    if members is None:
        members = {m: "{}\n" for m in at.GRPO_EXPECTED_INVENTORY}
    blob = make_tar(members)
    sha = hashlib.sha256(blob).hexdigest() + "  " + os.path.basename(spec.remote_archive) + "\n"
    return {spec.remote_archive: blob,
            spec.remote_sha_file: sha.encode()}


GOOD_RC = 0
BAD_RC = 1


class RunnerLifecycle(unittest.TestCase):
    def _mk(self, spec, transport):
        return rgg.run_run(spec, transport, env={"HF_TOKEN": "hf_top_secret_123"},
                           is_tracked=lambda p: True)

    def test_1_happy_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = _spec(tmp)
            tr = FakeRunnerTransport(remote=_remote(spec), exec_rc=GOOD_RC)
            res = self._mk(spec, tr)
            self.assertEqual(res["training_status"], rgg.TRAINING_PASS)
            self.assertEqual(res["artifact_transaction_status"], at.ARTIFACT_TRANSACTION_PASS)
            self.assertEqual(res["stop_status"], "SESSION_STOP_PASS")
            self.assertEqual(res["status"], "ORCHESTRATION_COMPLETE")
            self.assertEqual(tr.stop_calls, 1)
            # token persisted only as a boolean; value never hits the state file
            state = open(os.path.join(tmp, rgg.STATE_FILENAME)).read()
            self.assertIn('"hf_token_present": true', state)
            self.assertNotIn("hf_top_secret_123", state)

    def test_2_training_fail_artifacts_survive(self):
        """§13: TRAINING_FAIL + artifact PASS may still stop (evidence retrieved)."""
        with tempfile.TemporaryDirectory() as tmp:
            spec = _spec(tmp)
            tr = FakeRunnerTransport(remote=_remote(spec), exec_rc=BAD_RC)
            res = self._mk(spec, tr)
            self.assertEqual(res["training_status"], rgg.TRAINING_FAIL)
            self.assertEqual(res["artifact_transaction_status"], at.ARTIFACT_TRANSACTION_PASS)
            self.assertEqual(res["stop_status"], "SESSION_STOP_PASS")
            self.assertEqual(tr.stop_calls, 1)
            self.assertTrue(res["training_and_artifact_are_separate"])

    def test_3_training_ok_artifact_fail_keeps_session(self):
        """§13: TRAINING_PASS + failed transaction must NOT stop."""
        with tempfile.TemporaryDirectory() as tmp:
            spec = _spec(tmp)
            tr = FakeRunnerTransport(remote={}, exec_rc=GOOD_RC)
            res = self._mk(spec, tr)
            self.assertEqual(res["training_status"], rgg.TRAINING_PASS)
            self.assertEqual(res["artifact_transaction_status"], "FAILED")
            self.assertEqual(res["stop_status"], "SESSION_STOP_BLOCKED")
            self.assertEqual(tr.stop_calls, 0)

    def test_4_sha_mismatch_no_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = _spec(tmp)
            remote = _remote(spec)
            remote = dict(remote)
            remote[spec.remote_archive] = b"tampered-after-signing"
            tr = FakeRunnerTransport(remote=remote, exec_rc=GOOD_RC)
            res = self._mk(spec, tr)
            self.assertEqual(res["artifact_transaction_failure_code"], at.SHA_MISMATCH)
            self.assertEqual(tr.stop_calls, 0)

    def test_5_inventory_mismatch_no_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = _spec(tmp, expected_inventory=("does_not_exist.json",))
            tr = FakeRunnerTransport(remote=_remote(_spec(tmp)), exec_rc=GOOD_RC)
            res = self._mk(spec, tr)
            self.assertEqual(res["artifact_transaction_failure_code"], at.INVENTORY_MISMATCH)
            self.assertEqual(tr.stop_calls, 0)

    def test_6_empty_sha_no_stop(self):
        """A sha file of zeros is valid hex but mismatches => no stop (fail-closed)."""
        with tempfile.TemporaryDirectory() as tmp:
            spec = _spec(tmp)
            remote = dict(_remote(spec))
            remote[spec.remote_sha_file] = ("0" * 64 + "  x\n").encode()
            tr = FakeRunnerTransport(remote=remote, exec_rc=GOOD_RC)
            res = self._mk(spec, tr)
            self.assertEqual(res["artifact_transaction_failure_code"], at.SHA_MISMATCH)
            self.assertEqual(tr.stop_calls, 0)

    def test_7_traversal_archive_no_stop(self):
        """Checksum matches but archive escapes evidence dir => ARCHIVE_TRAVERSAL."""
        with tempfile.TemporaryDirectory() as tmp:
            spec = _spec(tmp)
            blob = make_tar({"x": "data"})
            evil = os.path.join(tmp, "evil")
            os.makedirs(evil)
            with open(os.path.join(evil, "pwn.txt"), "w") as fh:
                fh.write("x")
            p = os.path.join(tmp, "e.tar.gz")
            with tarfile.open(p, "w:gz") as t:
                t.add(os.path.join(evil, "pwn.txt"), arcname="../pwn.txt")
            with open(p, "rb") as fh:
                blob = fh.read()
            remote = {spec.remote_archive: blob,
                      spec.remote_sha_file: (hashlib.sha256(blob).hexdigest() + "  e.tar.gz\n").encode()}
            tr = FakeRunnerTransport(remote=remote, exec_rc=GOOD_RC)
            res = self._mk(spec, tr)
            self.assertEqual(res["artifact_transaction_failure_code"], at.ARCHIVE_TRAVERSAL)
            self.assertEqual(tr.stop_calls, 0)

    def test_8_untracked_job_source_blocks_exec(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = _spec(tmp)
            tr = FakeRunnerTransport(remote=_remote(spec))
            res = rgg.run_run(spec, tr, env={}, is_tracked=lambda p: False)
            self.assertEqual(res["status"], rgg.STATUS_UNTRACKED)
            self.assertEqual(tr.execs, [])   # zero colab exec, zero stop
            self.assertEqual(tr.stop_calls, 0)

    def test_9_missing_session_blocks_exec(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = _spec(tmp)
            tr = FakeRunnerTransport(remote=_remote(spec), active=False)
            res = rgg.run_run(spec, tr, env={}, is_tracked=lambda p: True)
            self.assertEqual(res["status"], rgg.STATUS_NO_SESSION)
            self.assertEqual(tr.execs, [])
            self.assertEqual(tr.stop_calls, 0)

    def test_10_producer_gets_exact_ids_and_no_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = _spec(tmp)
            tr = FakeRunnerTransport(remote=_remote(spec), exec_rc=GOOD_RC)
            self._mk(spec, tr)
            prod = [e for e in tr.execs if e[0].endswith("artifact_produce.py")]
            self.assertEqual(len(prod), 1)
            env = prod[0][1]
            self.assertEqual(env["JOB_ID"], spec.job_id)
            self.assertEqual(env["ATTEMPT_ID"], spec.attempt_id)
            self.assertEqual(env["RUN_INSTANCE"], spec.run_instance)
            self.assertEqual(env["REMOTE_RUN_DIR"], spec.remote_run_dir)
            self.assertEqual(env["REMOTE_ARCHIVE"], spec.remote_archive)
            self.assertNotIn("HF_TOKEN", env)          # token goes to the job exec only
            job = [e for e in tr.execs if e[0] == spec.remote_job_source]
            self.assertEqual(len(job), 1)
            self.assertEqual(job[0][1]["HF_TOKEN"], "hf_top_secret_123")
            # §16: inventory comes from the single authoritative source
            self.assertEqual(spec.expected_inventory, at.GRPO_EXPECTED_INVENTORY)

    def test_11_no_silent_double_run_and_no_stale_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = _spec(tmp)
            rgg.run_run(spec, FakeRunnerTransport(remote={}, exec_rc=GOOD_RC),
                        env={}, is_tracked=lambda p: True)  # leaves FAIL state
            tr2 = FakeRunnerTransport(remote=_remote(spec), exec_rc=GOOD_RC)
            res2 = rgg.run_run(spec, tr2, env={}, is_tracked=lambda p: True)
            self.assertEqual(res2["status"], rgg.STATUS_STATE_EXISTS)   # requires --resume
            self.assertEqual(tr2.execs, [])

    def test_12_resume_after_training_does_not_retrain(self):
        """§16: crash after training -> resume retries artifacts only."""
        with tempfile.TemporaryDirectory() as tmp:
            spec = _spec(tmp)
            rgg.run_run(spec, FakeRunnerTransport(remote={}, exec_rc=BAD_RC),
                        env={}, is_tracked=lambda p: True)  # txn FAILED
            tr2 = FakeRunnerTransport(remote=_remote(spec), exec_rc=GOOD_RC)
            res = rgg.run_run(spec, tr2, env={}, is_tracked=lambda p: True, resume=True)
            self.assertEqual(tr2.execs, [])   # no re-training, no re-produce
            self.assertEqual(res["training_status"], rgg.TRAINING_FAIL)  # remembered code
            self.assertEqual(res["artifact_transaction_status"], at.ARTIFACT_TRANSACTION_PASS)
            self.assertEqual(res["stop_status"], "SESSION_STOP_PASS")

    def test_13_resume_after_pass_reconciles_stop(self):
        """§16 F/G: stop acknowledged -> resume reconciles, never retrains."""
        with tempfile.TemporaryDirectory() as tmp:
            spec = _spec(tmp)
            tr = FakeRunnerTransport(remote=_remote(spec), exec_rc=GOOD_RC)
            rgg.run_run(spec, tr, env={}, is_tracked=lambda p: True)
            self.assertEqual(tr.stop_calls, 1)
            tr2 = FakeRunnerTransport(remote=_remote(spec), exec_rc=GOOD_RC)
            res = rgg.run_run(spec, tr2, env={}, is_tracked=lambda p: True, resume=True)
            self.assertEqual(res["status"], "RESUMED_STOP_RECONCILED")
            self.assertEqual(res["training_status"], rgg.TRAINING_SKIPPED_RESUME)
            self.assertEqual(tr2.execs, [])
            self.assertEqual(tr2.stop_calls, 1)   # exactly one reconciled stop

    def test_14_blocked_shapes_never_reach_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = _spec(tmp)
            for remote, rc, expect_code in (
                    ({}, GOOD_RC, at.DOWNLOAD_ARCHIVE_FAILED),
                    ({spec.remote_archive: b"x"}, GOOD_RC, at.DOWNLOAD_SHA_FAILED)):
                tr = FakeRunnerTransport(remote=dict(remote), exec_rc=rc)
                res = rgg.run_run(_spec(tempfile.mkdtemp(), remote_archive=spec.remote_archive,
                                        remote_run_dir=spec.remote_run_dir),
                                  tr, env={}, is_tracked=lambda p: True)
                self.assertEqual(res["stop_status"], "SESSION_STOP_BLOCKED")
                self.assertEqual(tr.stop_calls, 0)

    def test_15_operator_force_stop_is_explicit_and_never_success(self):
        """§15: emergency stop requires a reason and never records success."""
        with tempfile.TemporaryDirectory() as tmp:
            spec = _spec(tmp)
            tr = FakeRunnerTransport(remote={})
            with self.assertRaises(at.SpecError):
                rgg.operator_force_stop(spec, tr, "   ")
            res = rgg.operator_force_stop(spec, tr, "budget exhausted")
            self.assertTrue(res["operator_forced"])
            self.assertEqual(tr.stop_calls, 1)
            journal = at.read_journal(at.journal_path_for(spec.to_transaction_spec()))
            self.assertEqual(journal["status"], "OPERATOR_FORCED_STOP")
            self.assertFalse(journal["session_stop_allowed"])
            self.assertFalse(journal["artifact_transaction_complete"])


class RunnerSpecValidation(unittest.TestCase):
    def test_16_forbidden_attempt_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bad in ("attempt", "attempt-1", "latest", "current", "default",
                        "none", "null", ""):
                with self.assertRaises(at.SpecError, msg=bad):
                    _spec(tmp, attempt_id=bad,
                          remote_env={"JOB_ID": "grpo-smoldataenvs-smoke-001",
                                      "ATTEMPT_ID": bad, "RUN_INSTANCE": "g-t4-abc123",
                                      "REMOTE_RUN_DIR": "/content/opentraining/x"})

    def test_17_path_and_env_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(at.SpecError):
                _spec(tmp, remote_run_dir="relative/dir")
            with self.assertRaises(at.SpecError):
                _spec(tmp, remote_env={})
            with self.assertRaises(at.SpecError):
                _spec(tmp, remote_env={"JOB_ID": "other", "ATTEMPT_ID": "a",
                                       "RUN_INSTANCE": "r", "REMOTE_RUN_DIR": "/c/x"})
            with self.assertRaises(at.SpecError):
                _spec(tmp, remote_env={"JOB_ID": "grpo-smoldataenvs-smoke-001",
                                       "ATTEMPT_ID": "attempt-5-t4-orchestrated",
                                       "RUN_INSTANCE": "g-t4-abc123",
                                       "REMOTE_RUN_DIR": "/content/opentraining/"
                                                        "grpo-smoldataenvs-smoke-001/"
                                                        "attempt-5-t4-orchestrated",
                                       "HF_TOKEN": "should-not-be-here"})


class RunnerStaticInvariants(unittest.TestCase):
    def _tree(self):
        with open(RUNNER_PATH, encoding="utf-8") as fh:
            return ast.parse(fh.read(), filename=RUNNER_PATH)

    def test_18_no_training_framework_imports(self):
        mods = set()
        for node in ast.walk(self._tree()):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module.split(".")[0])
        self.assertFalse(mods & FORBIDDEN_IMPORTS, sorted(mods & FORBIDDEN_IMPORTS))

    def test_19_no_direct_cli_stop_or_provision_argv(self):
        """§19/§20 anti-bypass: the runner's only subprocess argv is git."""
        offenders = []
        for node in ast.walk(self._tree()):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr in ("run", "Popen", "call", "check_output", "check_call"):
                if node.args and isinstance(node.args[0], ast.List):
                    toks = [getattr(e, "value", None) for e in node.args[0].elts]
                    if toks and toks[0] != "git":
                        offenders.append(toks[0])
                    for tok in toks:
                        if tok in ("colab", "stop", "jupyter", "torchrun", "accelerate"):
                            offenders.append(tok)
        self.assertEqual(offenders, [])

    def test_20_stop_authority_only_through_guarded_api(self):
        src = open(RUNNER_PATH, encoding="utf-8").read()
        for api in ("guarded_stop", "reconcile_stop", "operator_force_stop"):
            self.assertIn(api, src)
        called = set()
        for node in ast.walk(self._tree()):
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Attribute):
                    called.add(f.attr)
                elif isinstance(f, ast.Name):
                    called.add(f.id)
        self.assertNotIn("stop", called)      # never a bare/transport stop() call
        self.assertNotIn("stop_session", called)
        self.assertIn("guarded_stop", called)
        self.assertIn("reconcile_stop", called)

    def test_21_runner_never_references_legacy_bundlers(self):
        src = open(RUNNER_PATH, encoding="utf-8").read()
        self.assertNotIn("grpo_bundle", src)
        self.assertNotIn("grpo_artifacts_bundle", src)


class RunnerExperimentEnv(unittest.TestCase):
    """§5/§12 wiring: hyperparameters reach the worker; producer stays clean."""

    def test_experiment_values_reach_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = _spec(tmp, experiment_env={"MAX_STEPS": "10", "LEARNING_RATE": "1e-6",
                                              "TEMPERATURE": "0.6"})
            tr = FakeRunnerTransport(remote=_remote(spec), exec_rc=GOOD_RC)
            rgg.run_run(spec, tr, env={"HF_TOKEN": "hf_top_secret_123"}, is_tracked=lambda p: True)
            _path, env, _out = tr.execs[0]  # first exec is the GRPO worker
            self.assertEqual(env.get("MAX_STEPS"), "10")
            self.assertEqual(env.get("LEARNING_RATE"), "1e-6")
            self.assertEqual(env.get("TEMPERATURE"), "0.6")
            self.assertEqual(env.get("JOB_ID"), spec.job_id)
            self.assertIn("HF_TOKEN", env)

    def test_producer_exec_has_no_experiment_or_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = _spec(tmp, experiment_env={"MAX_STEPS": "10"})
            tr = FakeRunnerTransport(remote=_remote(spec), exec_rc=GOOD_RC)
            rgg.run_run(spec, tr, env={"HF_TOKEN": "hf_top_secret_123"}, is_tracked=lambda p: True)
            prod_env = tr.execs[1][1]  # producer exec
            self.assertNotIn("MAX_STEPS", prod_env)
            self.assertNotIn("HF_TOKEN", prod_env)
            self.assertEqual(prod_env.get("REMOTE_ARCHIVE"), spec.remote_archive)

    def test_credential_experiment_key_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(at.SpecError):
                _spec(tmp, experiment_env={"HF_TOKEN": "leak"})

    def test_identity_experiment_key_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(at.SpecError):
                _spec(tmp, experiment_env={"JOB_ID": "other"})

    def test_empty_experiment_value_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(at.SpecError):
                _spec(tmp, experiment_env={"MAX_STEPS": "  "})

    def test_experiment_keys_persisted_not_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = _spec(tmp, experiment_env={"LEARNING_RATE": "1e-6"})
            self.assertIn("LEARNING_RATE", spec.state_fields().get("experiment_env_keys", []))


if __name__ == "__main__":
    unittest.main(verbosity=2)
