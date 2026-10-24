#!/usr/bin/env python3
"""Deterministic regression test for the real Colab transport exec timeout (§18).

Root cause guarded: ColabTransport.exec_file must pass an explicit generous
`--timeout` to the colab CLI. The CLI's own default is 30s, which silently
aborts a real GRPO run mid-training (preflight written, metrics/results missing
-> INVENTORY_MISMATCH, the only remote copy then destroyed on stop). The local
subprocess cap is held strictly ABOVE the remote cap so it never truncates.

No Colab, no GPU, no HF: subprocess.run is monkeypatched to record argv.
Run: /opt/venv/bin/python -m unittest -v tests/test_transport_exec_timeout.py
"""
import os
import sys
import types
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_JOBS = os.path.join(REPO, "training_jobs")
if TRAINING_JOBS not in sys.path:
    sys.path.insert(0, TRAINING_JOBS)

import artifact_transaction as at  # noqa: E402


class TransportExecTimeout(unittest.TestCase):
    def _capture(self, **exec_kwargs):
        rec = {}

        def fake_run(argv, **kw):
            rec["argv"] = list(argv)
            rec["timeout_kwarg"] = kw.get("timeout")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        orig = at.subprocess.run
        at.subprocess.run = fake_run
        try:
            tr = at.ColabTransport("sess-xyz")
            rc = tr.exec_file("/tmp/job.py", env={"JOB_ID": "j1", "MAX_STEPS": "20"},
                              output_path=None, **exec_kwargs)
        finally:
            at.subprocess.run = orig
        return rec, rc

    @staticmethod
    def _argv_timeout(argv):
        assert "--timeout" in argv, "exec_file must pass an explicit --timeout to colab"
        return int(argv[argv.index("--timeout") + 1])

    def test_default_timeout_not_the_30s_cli_default(self):
        rec, rc = self._capture()
        self.assertEqual(rc, 0)
        t = self._argv_timeout(rec["argv"])
        self.assertNotEqual(t, 30, "must not rely on the colab CLI 30s default")
        self.assertGreater(t, 300, "remote exec cap must allow a full GRPO run")

    def test_local_cap_above_remote_cap(self):
        rec, _ = self._capture()
        remote_t = self._argv_timeout(rec["argv"])
        local_t = rec["timeout_kwarg"]
        self.assertIsNotNone(local_t, "_cli must bound the local subprocess too")
        self.assertGreater(local_t, remote_t,
                           "local subprocess cap must exceed the remote --timeout "
                           "so it never truncates the run")

    def test_env_forwarded_and_no_stop_in_exec_argv(self):
        rec, _ = self._capture()
        argv = rec["argv"]
        self.assertIn("JOB_ID=j1", argv)
        self.assertIn("MAX_STEPS=20", argv)
        self.assertNotIn("stop", argv, "exec path must never carry a stop verb")

    def test_explicit_timeout_configurable(self):
        rec, _ = self._capture(timeout=600)
        self.assertEqual(self._argv_timeout(rec["argv"]), 600)
        self.assertEqual(rec["timeout_kwarg"], 720,
                         "local cap should be the remote cap plus headroom")

    def test_argv_targets_exec_verb(self):
        rec, _ = self._capture()
        self.assertEqual(rec["argv"][:2], ["colab", "exec"])
        self.assertIn("-s", rec["argv"])
        self.assertIn("sess-xyz", rec["argv"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
