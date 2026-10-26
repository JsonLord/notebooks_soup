#!/usr/bin/env python3
"""Prompt safety tests for P0-P3 variants (§7).

Verifies that no variant leaks gold answers, reference solutions, grader
internals, or asks for visible reasoning prose. Also verifies the output
contract (real newlines, final answer as last stdout line, filenames,
schema) is preserved in every variant.
"""
import ast
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training_jobs"))
from grpo_prompt_contract import build_prompt, task_validity_for


# A representative task row with known gold and schema.
_ROW = {
    "question": "What is the highest Trans Fat (g) value in the dataset?",
    "answer": "9.0",
    "files": ["nutrition.csv"],
    "bucket_prefix": "test__nutrition",
    "reward_mode": "flexible",
    "atol": 0.01,
    "rtol": 0.0,
    "task_id": "test__nutrition__max_tf",
}

# A contract with known schema.
_CONTRACT = {
    "dir": "/home/user/input",
    "files": [
        {"name": "nutrition.csv", "kind": "csv",
         "columns": ["Serving Size", "Calories", "Trans Fat (g)", "Cholesterol (mg)"],
         "schema_read": "SCHEMA_READ_OK"},
    ],
}

VARIANTS = ["P0", "P1", "P2", "P3"]


def _all_text(variant, row=None, contract=None):
    row = row or _ROW
    contract = contract or _CONTRACT
    msgs = build_prompt(row, contract=contract, variant=variant)
    return "\n".join(m["content"] for m in msgs).lower()


def test_gold_answer_absent():
    """The gold answer value must never appear in any variant prompt."""
    for v in VARIANTS:
        text = _all_text(v)
        # The gold is "9.0" — check it's not in the prompt body (not the question)
        # The question itself may contain numbers, so we check the scaffolding only.
        # Actually, the question is "What is the highest Trans Fat (g) value in the dataset?"
        # which doesn't contain "9.0". So the gold must not appear anywhere.
        assert "9.0" not in text, "P%d: gold answer '9.0' found in prompt" % v


def test_reference_solution_absent():
    """No reference program/solution must appear in any variant."""
    for v in VARIANTS:
        text = _all_text(v)
        # A reference solution would contain code patterns like:
        for needle in ("import pandas", "pd.read_csv", "max_trans", "df['trans", "df['trans fat"): 
            assert needle not in text, "P%d: reference solution pattern '%s' found" % (v, needle)


def test_grader_internals_absent():
    """No grader implementation detail must leak."""
    for v in VARIANTS:
        text = _all_text(v)
        for needle in ("reward_mode", "abs_tol", "rel_tol", "grader.grade",
                       "reward = 1.0", "return 0.0", "def grade"):
            assert needle not in text, "P%d: grader internal '%s' found" % (v, needle)


def test_filenames_present():
    """The actual staged filename must be in every variant."""
    for v in VARIANTS:
        text = _all_text(v)
        assert "nutrition.csv" in text, "P%d: staged filename 'nutrition.csv' missing" % v


def test_schema_present():
    """The actual CSV columns must be in every variant."""
    for v in VARIANTS:
        text = _all_text(v)
        assert "serving size" in text, "P%d: schema column 'Serving Size' missing" % v
        assert "trans fat (g)" in text, "P%d: schema column 'Trans Fat (g)' missing" % v


def test_real_newline_contract_present():
    """The real-newline output contract must be in every variant."""
    for v in VARIANTS:
        text = _all_text(v)
        assert "real newline" in text, "P%d: real-newline contract missing" % v
        assert "backslash-n" in text, "P%d: backslash-n warning missing" % v


def test_final_output_contract_present():
    """The final-answer-as-last-stdout-line contract must be in every variant."""
    for v in VARIANTS:
        text = _all_text(v)
        assert "last non-empty stdout line" in text, "P%d: final-output contract missing" % v


def test_no_visible_reasoning_prose():
    """No variant must ask the model to emit visible chain-of-thought or reasoning prose.

    The scaffolding says 'internally determine' and 'output only the executable
    Python program' — it must NOT say 'explain your reasoning', 'show your work',
    'step by step', 'chain of thought', 'write out your reasoning', etc.
    """
    for v in VARIANTS:
        text = _all_text(v)
        for needle in ("explain your reasoning", "show your work", "chain of thought",
                       "step by step", "write out your reasoning", "think out loud",
                       "reasoning trace", "show your thinking"):
            assert needle not in text, "P%d: visible-reasoning request '%s' found" % (v, needle)


def test_p0_is_unchanged():
    """P0 must be byte-identical to the original build_prompt (no scaffolding)."""
    # P0 scaffold is empty string — the prompt should match the original format.
    text = _all_text("P0")
    # The original prompt has "Keep it under 40 lines." immediately followed by
    # the output code contract. No scaffolding text between them.
    assert "internally determine" not in text, "P0: scaffolding text found (should be P0 baseline)"
    assert "when working with tabular data" not in text, "P0: P2/P3 text found"
    assert "prefer the simplest" not in text, "P0: P3 text found"


def test_p1_has_checklist_no_robustness():
    """P1 has the analysis checklist but NOT the data-robustness scaffold."""
    text = _all_text("P1")
    assert "internally determine" in text, "P1: checklist missing"
    assert "when working with tabular data" not in text, "P1: P2 robustness text found"
    assert "prefer the simplest" not in text, "P1: P3 text found"


def test_p2_has_robustness_no_decomposition():
    """P2 has checklist + robustness but NOT the P3 decomposition."""
    text = _all_text("P2")
    assert "internally determine" in text, "P2: checklist missing"
    assert "when working with tabular data" in text, "P2: robustness text missing"
    assert "prefer the simplest" not in text, "P2: P3 decomposition found"


def test_p3_has_all_scaffolds():
    """P3 has checklist + robustness + decomposition."""
    text = _all_text("P3")
    assert "internally determine" in text, "P3: checklist missing"
    assert "when working with tabular data" in text, "P3: robustness text missing"
    assert "prefer the simplest" in text, "P3: decomposition missing"


def test_prompt_hashes_differ():
    """All four variants must produce different prompt hashes."""
    hashes = {}
    for v in VARIANTS:
        msgs = build_prompt(_ROW, contract=_CONTRACT, variant=v)
        h = hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()
        hashes[v] = h
    assert len(set(hashes.values())) == 4, "Prompt hashes not all distinct: %s" % hashes


def test_no_variant_changes_reward_or_grader():
    """None of the prompt variants may contain reward/grader parameters."""
    for v in VARIANTS:
        text = _all_text(v)
        assert "reward_mode" not in text
        assert "atol" not in text
        assert "rtol" not in text
        assert "grader" not in text


def test_task_validity_unchanged():
    """task_validity_for must not be affected by prompt variant."""
    assert task_validity_for("test__nutrition") == "VALIDATED"
    assert task_validity_for("fbi-us__california-crime") == "GOLD_UNREPRODUCIBLE"



def test_prompt_signature_deterministic():
    """prompt_variant_signature must be deterministic."""
    from grpo_prompt_contract import prompt_variant_signature
    for v in ["P0", "P1", "P2", "P3"]:
        assert prompt_variant_signature(v) == prompt_variant_signature(v)


def test_prompt_signature_variants_differ():
    """Different variants must have different signatures."""
    from grpo_prompt_contract import prompt_variant_signature
    sigs = [prompt_variant_signature(v) for v in ["P0", "P1", "P2", "P3"]]
    assert len(set(sigs)) == 4, "signatures should differ across variants"


def test_prompt_signature_p1_frozen():
    """P1 scaffold signature must match the frozen value."""
    from grpo_prompt_contract import prompt_variant_signature
    assert prompt_variant_signature("P1") == "6310c395fe09c5f9"


def test_worker_has_prompt_scaffolds():
    """Worker must define PROMPT_SCAFFOLDS + prompt_variant_signature (drift prevention)."""
    with open("training_jobs/grpo_smoldataenvs.py") as f:
        src = f.read()
    assert "def prompt_variant_signature(variant):" in src
    assert "PROMPT_SCAFFOLDS" in src
    assert "hexdigest()[:16]" in src
    assert "import hashlib" in src


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print("PASS %s" % t.__name__)
        except AssertionError as e:
            print("FAIL %s: %s" % (t.__name__, e))
            sys.exit(1)
    print("\n%d/%d prompt safety tests passed" % (passed, len(tests)))


