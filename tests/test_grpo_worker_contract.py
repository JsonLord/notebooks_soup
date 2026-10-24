#!/usr/bin/env python3
"""Deterministic contract tests for the canonical GRPO worker (§18/§20).

No GPU, no HF, no torch/trl execution: the worker is parsed statically (it
imports torch by design and runs only on Colab), the pure eval-split/scoring
helpers are unit-tested from the dependency-free module, and the worker's inline
copies are AST-compared to the module to kill drift. Run:
  /opt/venv/bin/python -m unittest -v tests/test_grpo_worker_contract.py
"""
import ast
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_JOBS = os.path.join(REPO, "training_jobs")
if TRAINING_JOBS not in sys.path:
    sys.path.insert(0, TRAINING_JOBS)

import grpo_eval_split as ges  # noqa: E402

WORKER = os.path.join(TRAINING_JOBS, "grpo_smoldataenvs.py")
MODULE = os.path.join(TRAINING_JOBS, "grpo_eval_split.py")


def _funcs(path):
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src, filename=path)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("disjoint_split", "score_from_rewards"):
            out[node.name] = ast.get_source_segment(src, node)
    return out


class EvalSplitMath(unittest.TestCase):
    def test_determinism(self):
        self.assertEqual(ges.disjoint_split(10, 2, 3), ges.disjoint_split(10, 2, 3))
        self.assertEqual(ges.disjoint_split(10, 2, 3), ([0, 1], [2, 3, 4]))

    def test_disjoint_and_bounds(self):
        tr, ev = ges.disjoint_split(8, 4, 4)
        self.assertEqual(set(tr) & set(ev), set())
        self.assertEqual(tr + ev, list(range(8)))
        with self.assertRaises(ValueError):
            ges.disjoint_split(5, 3, 3)  # exceeds total
        with self.assertRaises(ValueError):
            ges.disjoint_split(5, -1, 1)

    def test_zero_eval_allowed_for_smoke(self):
        tr, ev = ges.disjoint_split(3, 3, 0)
        self.assertEqual(tr, [0, 1, 2])
        self.assertEqual(ev, [])

    def test_score_no_fabrication(self):
        self.assertEqual(ges.score_from_rewards([]), {"n": 0, "mean_reward": None, "nonzero_reward_rate": None})
        s = ges.score_from_rewards([1.0, 0.0, 0.5])
        self.assertAlmostEqual(s["mean_reward"], 0.5)
        self.assertAlmostEqual(s["nonzero_reward_rate"], 2.0 / 3.0)


class WorkerStaticContract(unittest.TestCase):
    def setUp(self):
        with open(WORKER, encoding="utf-8") as fh:
            self.src = fh.read()
        self.tree = ast.parse(self.src, filename=WORKER)

    def test_execution_target_print(self):
        self.assertIn("EXECUTION_TARGET=COLAB_REMOTE", self.src)

    def test_identity_env_required(self):
        for key in ("JOB_ID", "ATTEMPT_ID", "RUN_INSTANCE"):
            self.assertIn('_req("%s")' % key, self.src, key)

    def test_fixed_boundaries_present(self):
        self.assertIn('"use_vllm": USE_VLLM', self.src)
        self.assertIn("forward_hf_token=False", self.src)
        self.assertIn("cpu-basic", self.src)

    def test_no_trainer_callback(self):
        self.assertNotIn("TrainerCallback", self.src)
        self.assertIn("custom_callbacks = []", self.src)

    def test_model_guard_for_2b_on_t4(self):
        self.assertIn("MODEL_NOT_SUPPORTED_ON_T4", self.src)

    def test_eval_metric_names(self):
        for key in ("pre_train_eval", "post_train_eval", "reward_delta",
                    "pre_train_nonzero_reward_rate", "post_train_nonzero_reward_rate",
                    "valid_program_rate", "sandbox_execution_rate", "mean_completion_length"):
            self.assertIn(key, self.src, key)

    def test_writes_all_grpo_inventory(self):
        for f in ("manifest.json", "environment.json", "hardware.json", "sandbox.json",
                  "auth.json", "grpo_api_preflight.json", "metrics.jsonl", "results.json",
                  "heartbeat.json"):
            self.assertIn('"/" + "%s"' % f if False else ('/" + "%s"' % f), self.src, f) \
                if False else self.assertIn(f, self.src, f)

    def test_blocked_paths_exit_zero(self):
        # every blocked path must exit(0) so the artifact producer still bundles
        self.assertIn("def blocked(", self.src)
        self.assertIn("sys.exit(0)  # exit(0) so the artifact producer still bundles evidence", self.src)


class InlineCopyDrift(unittest.TestCase):
    def test_inline_helpers_match_module(self):
        w = _funcs(WORKER)
        m = _funcs(MODULE)
        self.assertEqual(set(w.keys()), {"disjoint_split", "score_from_rewards"})
        self.assertEqual(set(m.keys()), {"disjoint_split", "score_from_rewards"})
        for name in ("disjoint_split", "score_from_rewards"):
            self.assertEqual(w[name], m[name], "worker inline %s drifted from module" % name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
