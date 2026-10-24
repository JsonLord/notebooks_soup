#!/usr/bin/env python3
"""Deterministic contract tests for the canonical GRPO worker (§18/§20).

No GPU, no HF, no torch/trl execution: the worker is parsed statically (it
imports torch by design and runs only on Colab), the pure eval-split/scoring
helpers are unit-tested from the dependency-free module, and the worker's inline
copies are AST-compared to the module to kill drift. Run:
  /opt/venv/bin/python -m unittest -v tests/test_grpo_worker_contract.py
"""
import ast
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_JOBS = os.path.join(REPO, "training_jobs")
if TRAINING_JOBS not in sys.path:
    sys.path.insert(0, TRAINING_JOBS)

import grpo_eval_split as ges  # noqa: E402
import grpo_prompt_contract as gpc  # noqa: E402

WORKER = os.path.join(TRAINING_JOBS, "grpo_smoldataenvs.py")
MODULE = os.path.join(TRAINING_JOBS, "grpo_eval_split.py")
MODULE_PC = os.path.join(TRAINING_JOBS, "grpo_prompt_contract.py")
NAMES_PC = ("inspect_staged_contract", "stage_bucket", "build_prompt", "task_validity_for")


def _funcs(path):
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src, filename=path)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("disjoint_split", "score_from_rewards"):
            out[node.name] = ast.get_source_segment(src, node)
    return out


def _funcs_named(path, names):
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src, filename=path)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            out[node.name] = ast.get_source_segment(src, node)
    return out


class R:
    def __init__(self, out):
        self.stdout = out


class FakeSb:
    """In-process stand-in for huggingface_hub.Sandbox.

    inspect_staged_contract builds:
        printf '%s' '<escaped snippet>' > /tmp/_ic.py && python3 /tmp/_ic.py '<inp>'
    We recover the printf script arg and the trailing input path with shlex
    (which correctly undoes the POSIX '\\'' quoting, so no manual escape strings),
    then run the module's OWN snippet verbatim with the current interpreter
    against the fixture dir. This proves the snippet logic with no bash/pyenv/
    printf-in-shell; the shell escaping itself is verified separately (rc 0,
    valid JSON) so this test focuses on what the contract actually contains.
    """

    def run(self, cmd, check=False):
        import shlex
        left, _, right = cmd.partition(" > /tmp/_ic.py && ")
        script = shlex.split(left)[2]
        inp = shlex.split(right)[-1]
        dd = tempfile.mkdtemp(prefix="ic_")
        sp = os.path.join(dd, "_ic.py")
        with open(sp, "w", encoding="utf-8") as fh:
            fh.write(script)
        p = subprocess.run([sys.executable, sp, inp], capture_output=True, text=True, timeout=120)
        # A clean production sandbox prints ONLY the program's stdout, but a
        # spawned child under some interpreter launches can prefix/suffix a
        # banner/noise line. The snippet emits exactly one JSON object on its
        # own line, so tolerate surrounding noise by taking the LAST
        # JSON-parseable line (a robust caller-side reader). This hardens the
        # TEST fake only; the module stays correct for the real sandbox.
        best = ""
        for line in (p.stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                best = line
            except Exception:
                continue
        return R(best)


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

    def test_inline_prompt_contract_match_module(self):
        # §17: the worker and the standalone inference probe must share ONE
        # builder/inspector. Enforce it by AST-drifting the four prompt-contract
        # defs between the worker's inline copy and grpo_prompt_contract.py.
        w = _funcs_named(WORKER, NAMES_PC)
        m = _funcs_named(MODULE_PC, NAMES_PC)
        self.assertEqual(set(w.keys()), set(NAMES_PC))
        self.assertEqual(set(m.keys()), set(NAMES_PC))
        for name in NAMES_PC:
            self.assertEqual(w[name], m[name], "worker inline %s drifted from module" % name)


class PromptContractLogic(unittest.TestCase):
    # §12: the prompt/file contract must equal the ACTUAL staged contract, with
    # no gold leakage, robust to messy real-world files, and with the two
    # historical guessing failures (invented 'Matches' table / invented filename)
    # regressed. Runs the module's OWN snippet via FakeSb against a fixture dir.

    def _fixture(self):
        d = tempfile.mkdtemp(prefix="pc_")
        with open(os.path.join(d, "single.csv"), "w", encoding="utf-8") as fh:
            fh.write("Item,Trans Fat\nMcDouble,3.5\n")
        with open(os.path.join(d, "ca_offenses_by_city.csv"), "w", encoding="utf-8") as fh:
            fh.write("City, Population, Violent crime\nImperial3, 33,005, 5\n")
        with open(os.path.join(d, "ca_offenses_by_year.csv"), "w", encoding="utf-8") as fh:
            fh.write("Year,Total\n2020,10\n")
        with open(os.path.join(d, "ragged.csv"), "w", encoding="utf-8") as fh:
            fh.write("a_single_very_long_unnamed_field_header_without_any_comma_at_all_x\n1,2,3\n")
        with open(os.path.join(d, "embedded.csv"), "wb") as fh:
            fh.write(b"County\rCity\r\nLosAngeles\rLACity\r\n")
        with open(os.path.join(d, "malformed.csv"), "wb") as fh:
            fh.write(bytes([0x00, 0x01, 0xff, 0x80, 0x02, 0x03]))
        con = sqlite3.connect(os.path.join(d, "database.sqlite"))
        con.execute("CREATE TABLE Match(id INTEGER, home_team_goal INTEGER, away_team_goal INTEGER)")
        con.execute("CREATE TABLE Player(id INTEGER, player_name TEXT)")
        con.execute("CREATE TABLE seqholder(id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT)")
        con.execute("INSERT INTO seqholder(v) VALUES ('x')")
        con.commit()
        con.close()
        contract = gpc.inspect_staged_contract(FakeSb(), inp=d)
        return d, contract

    def test_all_staged_filenames_surfaced_none_invented(self):
        d, contract = self._fixture()
        names = {f["name"] for f in contract["files"]}
        for n in ("single.csv", "ca_offenses_by_city.csv", "ca_offenses_by_year.csv",
                  "ragged.csv", "embedded.csv", "malformed.csv", "database.sqlite"):
            self.assertIn(n, names)
        for n in names:
            self.assertTrue(os.path.isfile(os.path.join(d, n)), n)

    def test_single_csv_columns_exact(self):
        d, contract = self._fixture()
        by = {f["name"]: f for f in contract["files"]}
        self.assertEqual(by["single.csv"]["columns"], ["Item", "Trans Fat"])
        self.assertEqual(by["single.csv"]["schema_read"], "SCHEMA_READ_OK")

    def test_composite_second_file_not_hidden(self):
        d, contract = self._fixture()
        row = {"question": "q", "files": ["ca_offenses_by_city.csv"]}
        text = "\n".join(m["content"] for m in gpc.build_prompt(row, contract=contract))
        self.assertIn("ca_offenses_by_city.csv", text)
        self.assertIn("ca_offenses_by_year.csv", text)

    def test_ragged_partial_no_crash(self):
        d, contract = self._fixture()
        by = {f["name"]: f for f in contract["files"]}
        self.assertEqual(by["ragged.csv"].get("schema_read"), "SCHEMA_PARTIAL")

    def test_malformed_tagged_contract_survives(self):
        d, contract = self._fixture()
        by = {f["name"]: f for f in contract["files"]}
        self.assertEqual(by["malformed.csv"]["kind"], "csv")
        self.assertNotEqual(contract["files"], [])

    def test_sqlite_real_table_not_invented_plural(self):
        d, contract = self._fixture()
        by = {f["name"]: f for f in contract["files"]}
        self.assertEqual(by["database.sqlite"]["kind"], "sqlite")
        tabs = by["database.sqlite"]["tables"]
        self.assertIn("Match", tabs)
        self.assertIn("home_team_goal", tabs["Match"])
        self.assertIn("away_team_goal", tabs["Match"])
        self.assertNotIn("Matches", tabs)          # regression: 'no such table: Matches'
        self.assertNotIn("sqlite_sequence", tabs)   # internal table never surfaced

    def test_prompt_discloses_schema_no_gold_leak(self):
        d, contract = self._fixture()
        row = {"question": "What is the draw ratio?", "files": ["database.sqlite"],
               "answer": "25.4%", "bucket_prefix": "hugomathien__soccer"}
        text = "\n".join(m["content"] for m in gpc.build_prompt(row, contract=contract))
        self.assertIn("Match", text)
        self.assertIn("home_team_goal", text)
        self.assertNotIn("Matches", text)          # never invents the plural
        self.assertNotIn("25.4", text)             # NO GOLD LEAK
        self.assertNotIn("reward_mode", text)      # no grader internals
        self.assertNotIn("abs_tol", text)
        self.assertIn("last non-empty stdout", text)
        self.assertIn("exact filenames", text.lower())

    def test_legacy_fallback_when_no_contract(self):
        row = {"question": "q", "files": ["database.sqlite"], "answer": "25.4%",
               "bucket_prefix": "hugomathien__soccer"}
        text = "\n".join(m["content"] for m in gpc.build_prompt(row))  # contract=None
        self.assertIn("database.sqlite", text)
        self.assertNotIn("Match", text)            # unchanged old (schema-hidden) prompt


class TaskValidity(unittest.TestCase):
    # §3/§4/§13: a GOLD_UNREPRODUCIBLE task stays in evidence but is excluded
    # from the reward denominator; it is counted as neither model success nor
    # failure. This is an annotation, not a dataset edit, deletion, or grading
    # change.

    def test_california_crime_quarantined(self):
        self.assertEqual(gpc.task_validity_for("fbi-us__california-crime"), "GOLD_UNREPRODUCIBLE")
        self.assertEqual(gpc.task_validity_for("SomeOrg__California-Crime"), "GOLD_UNREPRODUCIBLE")

    def test_soccer_and_blank_validated(self):
        self.assertEqual(gpc.task_validity_for("hugomathien__soccer"), "VALIDATED")
        self.assertEqual(gpc.task_validity_for(""), "VALIDATED")

    def test_quarantined_excluded_from_reward_denominator(self):
        rows = [
            {"bucket_prefix": "hugomathien__soccer", "reward": 1.0},       # validated, scores
            {"bucket_prefix": "fbi-us__california-crime", "reward": 0.0},  # quarantined
            {"bucket_prefix": "org__other", "reward": 0.0},               # validated, no reward
        ]
        elig = [r for r in rows if gpc.task_validity_for(r["bucket_prefix"]) == "VALIDATED"]
        quar = [r for r in rows if gpc.task_validity_for(r["bucket_prefix"]) != "VALIDATED"]
        self.assertEqual(len(quar), 1)                                  # kept in evidence
        self.assertEqual(quar[0]["bucket_prefix"], "fbi-us__california-crime")
        sc = ges.score_from_rewards([r["reward"] for r in elig])        # eligible-only denom
        self.assertEqual(sc["n"], 2)
        self.assertAlmostEqual(sc["nonzero_reward_rate"], 0.5)
        self.assertNotAlmostEqual(sc["nonzero_reward_rate"], 1.0 / 3.0)  # not dragged by quarantined 0


if __name__ == "__main__":
    unittest.main(verbosity=2)
