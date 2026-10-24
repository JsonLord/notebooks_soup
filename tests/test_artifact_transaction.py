#!/usr/bin/env python3
"""Dependency-free deterministic tests for the guarded artifact-transaction
controller (§18). Run:  /opt/venv/bin/python -m unittest -v <thisfile>
No GPU / GRPO / dataset / model / HF Sandbox / Colab session is required (§20).
"""
import gzip
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training_jobs"))
import artifact_transaction as at  # noqa: E402


def make_tar(members: dict) -> bytes:
    import tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            b = data.encode() if isinstance(data, str) else data
            info.size = len(b)
            tf.addfile(info, io.BytesIO(b))
    return buf.getvalue()


def sha_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class FakeTransport(at.RemoteTransport):
    """In-memory remote filesystem + call recorder."""

    def __init__(self, remote=None, *, stdout_sha=None, stdout_done=False):
        self.remote = dict(remote or {})   # remote path -> bytes
        self.downloads = []
        self.stop_calls = 0
        self.active = True
        self.stdout_sha = stdout_sha       # diagnostic; must never be authority
        self.stdout_done = stdout_done

    def remote_exists(self, path):
        return path in self.remote

    def remote_size(self, path):
        return len(self.remote.get(path, b""))

    def download(self, remote_path, local_path):
        self.downloads.append(remote_path)
        if remote_path not in self.remote:
            return False
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        with open(local_path, "wb") as fh:
            fh.write(self.remote[remote_path])
        return True

    def session_active(self):
        return self.active

    def stop_session(self):
        self.stop_calls += 1
        self.active = False
        return True


GOOD_MEMBERS = {"manifest.json": "{}", "results.json": "{\"status\":\"ok\"}",
                "metrics.jsonl": "{\"loss\":0}", "nested/example.txt": "hi"}
EXPECTED = ("manifest.json", "results.json", "metrics.jsonl", "nested/example.txt")


class Case(unittest.TestCase):
    def _spec(self, tmp, attempt_id="attempt-5-t4-x", remote_archive="/content/opentraining/run.tar.gz"):
        return at.TransactionSpec(
            job_id="grpo-smoldataenvs-smoke-001", attempt_id=attempt_id,
            run_instance="m-s-abc", remote_run_dir="/content/opentraining/runs/att",
            remote_archive=remote_archive, local_evidence_dir=os.path.join(tmp, "evidence"),
            expected_inventory=EXPECTED)

    def _good_transport(self, spec):
        archive = make_tar(GOOD_MEMBERS)
        sha_file = ("%s  %s\n" % (sha_hex(archive), os.path.basename(spec.remote_archive))).encode()
        return FakeTransport({spec.remote_archive: archive, spec.remote_sha_file: sha_file})

    # 1 normal PASS
    def test_01_normal_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(tmp)
            t = self._good_transport(spec)
            j = at.run_transaction(spec, t, skip_producer_check=True)
            self.assertTrue(j["artifact_transaction_complete"])
            self.assertTrue(j["session_stop_allowed"])
            self.assertEqual(j["status"], at.ARTIFACT_TRANSACTION_PASS)
            g = at.guarded_stop(t, at.journal_path_for(spec))
            self.assertTrue(g["session_stopped"]) ; self.assertEqual(t.stop_calls, 1)

    # 2 remote archive one directory too deep -> FAIL no stop
    def test_02_wrong_nested_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(tmp)
            archive = make_tar(GOOD_MEMBERS)
            # producer wrote one dir deeper than spec expects -> archive missing
            deeper = spec.remote_archive + "/extra"
            t = FakeTransport({deeper: archive, deeper + ".sha256": b"x"})
            j = at.run_transaction(spec, t, skip_producer_check=False)
            self.assertFalse(j["session_stop_allowed"])
            self.assertEqual(j["failure_code"], at.REMOTE_ARCHIVE_MISSING)
            g = at.guarded_stop(t, at.journal_path_for(spec))
            self.assertFalse(g["stop_allowed"]) ; self.assertEqual(t.stop_calls, 0)

    # 3 archive download 404
    def test_03_archive_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(tmp)
            t = self._good_transport(spec)
            del t.remote[spec.remote_archive]
            j = at.run_transaction(spec, t, skip_producer_check=True)
            self.assertFalse(j["download_complete"])
            self.assertEqual(j["failure_code"], at.DOWNLOAD_ARCHIVE_FAILED)
            self.assertFalse(j["session_stop_allowed"])

    # 4 sha download 404
    def test_04_sha_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(tmp)
            t = self._good_transport(spec)
            del t.remote[spec.remote_sha_file]
            j = at.run_transaction(spec, t, skip_producer_check=True)
            self.assertEqual(j["failure_code"], at.DOWNLOAD_SHA_FAILED)
            self.assertFalse(j["session_stop_allowed"])

    # 5 remote sha empty
    def test_05_remote_sha_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(tmp)
            t = self._good_transport(spec)
            t.remote[spec.remote_sha_file] = b""
            j = at.run_transaction(spec, t, skip_producer_check=True)
            self.assertFalse(j["remote_sha_available"])
            self.assertEqual(j["failure_code"], at.LOCAL_SHA_FILE_EMPTY)
            self.assertFalse(j["session_stop_allowed"])

    # 5b remote sha invalid (not 64 hex) but file present & non-empty
    def test_05b_remote_sha_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(tmp)
            t = self._good_transport(spec)
            t.remote[spec.remote_sha_file] = b"notahash  run.tar.gz\n"
            j = at.run_transaction(spec, t, skip_producer_check=True)
            self.assertEqual(j["failure_code"], at.REMOTE_SHA_INVALID)
            self.assertFalse(j["session_stop_allowed"])

    # 6/7 fail-closed sha_match incl. empty==empty MUST FAIL
    def test_06_sha_match_failclosed(self):
        self.assertFalse(at.sha_match("", ""))
        self.assertFalse(at.sha_match(None, None))
        self.assertFalse(at.sha_match("", "a" * 64))
        self.assertFalse(at.sha_match("a" * 64, ""))
        self.assertFalse(at.sha_match("a" * 64, "b" * 64))
        self.assertTrue(at.sha_match("a" * 64, "a" * 64))

    # 8 sha mismatch
    def test_08_sha_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(tmp)
            t = self._good_transport(spec)
            t.remote[spec.remote_sha_file] = ("b" * 64 + "  run.tar.gz\n").encode()
            j = at.run_transaction(spec, t, skip_producer_check=True)
            self.assertEqual(j["failure_code"], at.SHA_MISMATCH)
            self.assertFalse(j["session_stop_allowed"])

    # 9 valid SHA but missing required member -> FAIL no stop
    def test_09_valid_sha_missing_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(tmp)
            members = dict(GOOD_MEMBERS); members.pop("results.json")
            archive = make_tar(members)
            sha_file = ("%s  %s\n" % (sha_hex(archive), "run.tar.gz")).encode()
            t = FakeTransport({spec.remote_archive: archive, spec.remote_sha_file: sha_file})
            j = at.run_transaction(spec, t, skip_producer_check=True)
            self.assertTrue(j["sha_match"])  # integrity of transfer is fine
            self.assertFalse(j["archive_inventory_ok"])
            self.assertEqual(j["failure_code"], at.INVENTORY_MISMATCH)
            self.assertFalse(j["session_stop_allowed"])
            g = at.guarded_stop(t, at.journal_path_for(spec))
            self.assertFalse(g["stop_allowed"]) ; self.assertEqual(t.stop_calls, 0)

    # 10 corrupt archive with matching remote checksum -> integrity FAIL no stop
    def test_10_corrupt_archive_matching_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(tmp)
            garbage = gzip.compress(b"this is not a valid tar stream at all")
            # not a tar.gz even though gzipped; tarfile.open(r:gz) will fail on read
            sha_file = ("%s  %s\n" % (sha_hex(garbage), "run.tar.gz")).encode()
            t = FakeTransport({spec.remote_archive: garbage, spec.remote_sha_file: sha_file})
            j = at.run_transaction(spec, t, skip_producer_check=True)
            self.assertTrue(j["sha_match"])  # checksum agrees on the (corrupt) bytes
            self.assertEqual(j["failure_code"], at.ARCHIVE_INVALID)
            self.assertFalse(j["session_stop_allowed"])

    # 11 path-traversal member -> FAIL
    def test_11_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = os.path.join(tmp, "evidence"); os.makedirs(ev, exist_ok=True)
            evil = os.path.join(tmp, "evil.tar.gz")
            import tarfile
            with tarfile.open(evil, "w:gz") as tf:
                info = tarfile.TarInfo(name="../escape.txt"); data = b"x"; info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            v = at.verify_archive(evil, ev)
            self.assertTrue(v["traversal"]) ; self.assertFalse(v["integrity_ok"])
            self.assertEqual(v["code"], at.ARCHIVE_TRAVERSAL)
            self.assertFalse(at.member_is_safe("../escape.txt"))
            self.assertFalse(at.member_is_safe("/abs/escape.txt"))

    # 12 duplicate download retry -> safe
    def test_12_duplicate_retry_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(tmp)
            t = self._good_transport(spec)
            j1 = at.run_transaction(spec, t, skip_producer_check=True)
            j2 = at.run_transaction(spec, t, skip_producer_check=True)
            self.assertTrue(j1["artifact_transaction_complete"] and j2["artifact_transaction_complete"])

    # 13 restart after downloads -> verification resumes (remote already reclaimed)
    def test_13_restart_after_downloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(tmp)
            good = self._good_transport(spec)
            at.run_transaction(spec, good, skip_producer_check=True)  # populate local copies
            # simulate restart: session gone (empty remote), local copies survive
            t2 = FakeTransport({})
            j = at.run_transaction(spec, t2, skip_producer_check=True)
            self.assertTrue(j["download_complete"])  # resumed from local files
            self.assertTrue(j["artifact_transaction_complete"])
            g = at.guarded_stop(t2, at.journal_path_for(spec))
            self.assertTrue(g["session_stopped"])

    # 14 restart after SHA match but before inventory -> inventory resumes
    def test_14_restart_before_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(tmp)
            good = self._good_transport(spec)
            at.download_stage(spec, good)  # only downloads performed
            s = at.sha_stage(spec); self.assertTrue(s["sha_match"])
            v = at.validate_stage(spec)    # inventory performed on resume
            self.assertTrue(v["archive_inventory_ok"])

    # 15 restart after transaction PASS -> stop allowed (persisted state re-read)
    def test_15_restart_after_pass_stop_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(tmp)
            good = self._good_transport(spec)
            at.run_transaction(spec, good, skip_producer_check=True)
            fresh = FakeTransport({})  # new "process" only has the journal
            g = at.guarded_stop(fresh, at.journal_path_for(spec))
            self.assertTrue(g["stop_allowed"]) ; self.assertTrue(g["session_stopped"])

    # 16 stdout BUNDLE_DONE without files -> zero authority, no stop
    def test_16_stdout_done_no_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(tmp)
            t = FakeTransport({})  # no durable files, but pretends stdout done
            t.stdout_done = True
            j = at.run_transaction(spec, t, skip_producer_check=True)
            self.assertFalse(j["session_stop_allowed"])
            self.assertEqual(j["failure_code"], at.DOWNLOAD_ARCHIVE_FAILED)

    # 17 stdout SHA disagrees with sibling SHA file -> sibling FILE wins
    def test_17_stdout_vs_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(tmp)
            t = self._good_transport(spec)
            archive = t.remote[spec.remote_archive]
            t.stdout_sha = "f" * 64  # bogus stdout value
            j = at.run_transaction(spec, t, skip_producer_check=True)
            self.assertEqual(j["remote_sha"], sha_hex(archive))  # from FILE, not stdout
            self.assertNotEqual(j["remote_sha"], t.stdout_sha)

    # 18 explicit attempt identifiers survive process reconstruction (journal)
    def test_18_identifiers_survive_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(tmp, attempt_id="attempt-9-t4-realid")
            t = self._good_transport(spec)
            at.run_transaction(spec, t, skip_producer_check=True)
            jr = at.read_journal(at.journal_path_for(spec))
            self.assertEqual(jr["attempt_id"], "attempt-9-t4-realid")
            rebuilt = at.TransactionSpec(
                job_id=jr["job_id"], attempt_id=jr["attempt_id"],
                run_instance=jr["run_instance"], remote_run_dir=jr["remote_run_dir"],
                remote_archive=jr["remote_archive"], local_evidence_dir=jr["local_evidence_dir"],
                expected_inventory=tuple(jr["expected_inventory"]))
            self.assertEqual(rebuilt.remote_sha_file, spec.remote_sha_file)

    # 19 no default/stale attempt can leak from prior shell state
    def test_19_no_stale_attempt(self):
        for bad in ("", "attempt-1", "latest", "default", "current"):
            with self.assertRaises(at.SpecError):
                self._spec("/tmp/na", attempt_id=bad)
        # legacy bundlers must refuse without explicit GRPO_ATTEMPT
        repo = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
        env = dict(os.environ); env.pop("GRPO_ATTEMPT", None)
        import subprocess
        for script in ("training_jobs/grpo_bundle.py", "training_jobs/grpo_artifacts_bundle.py"):
            r = subprocess.run([sys.executable, os.path.join(repo, script)],
                               env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, script)
            self.assertIn("ATTEMPT_MISSING", r.stdout, script)

    # 20 shutdown unreachable with ANY required gate false
    def test_20_shutdown_gate_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(tmp)
            t = self._good_transport(spec)
            at.run_transaction(spec, t, skip_producer_check=True)
            jp = at.journal_path_for(spec)
            good = at.read_journal(jp)
            for field in at.REQUIRED_TRUE_FIELDS:
                bad = dict(good); bad[field] = False
                at.write_journal(jp, bad)
                t.stop_calls = 0
                g = at.guarded_stop(t, jp)
                self.assertFalse(g["stop_allowed"], field)
                self.assertEqual(t.stop_calls, 0, field)
            # restore + verify a clean pass stops once
            at.write_journal(jp, good)
            self.assertTrue(at.guarded_stop(t, jp)["session_stopped"])

    # extra: §3 path validation guards (absolute / not run-dir)
    def test_21_spec_path_validation(self):
        with self.assertRaises(at.SpecError):
            self._spec("/tmp/na", remote_archive="run.tar.gz")  # not absolute
        with self.assertRaises(at.SpecError):
            at.TransactionSpec(job_id="j", attempt_id="attempt-3-x", run_instance="i",
                               remote_run_dir="/content/opentraining/runs/att",
                               remote_archive="/content/opentraining/runs/att",
                               local_evidence_dir="/tmp/e", expected_inventory=EXPECTED)
        self.assertTrue(at.parse_remote_sha("a" * 64 + "  run.tar.gz\n"))
        self.assertIsNone(at.parse_remote_sha(""))
        self.assertIsNone(at.parse_remote_sha("zz" * 32 + "  x"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
