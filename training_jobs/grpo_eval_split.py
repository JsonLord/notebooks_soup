#!/usr/bin/env python3
"""Dependency-free helpers for the OpenTraining T4 GRPO evaluation contract
(§7 fixed held-out eval slice; §20 reproducibility).

Pure stdlib ONLY: importable and unit-testable WITHOUT torch / trl / GPU / HF /
Colab (§20). The training worker keeps a byte-identical inline copy of these
two functions so it remains a single self-contained file for `colab exec -f`;
a contract test AST-compares the two definitions to kill drift.

The eval slice is chosen so that, for a fixed seed and a deterministically
shuffled dataset pool, EVERY candidate sees the identical held-out eval rows,
and those eval rows are DISJOINT from the training rows (§7).
"""


def disjoint_split(total, train_n, eval_n):
    """Return (train_ids, eval_ids): strictly disjoint index lists over range(total).

    train_ids are the first train_n indices, eval_ids the next eval_n. Raising if
    the requested counts exceed the pool prevents a silently overlapping or
    truncated eval slice (which would break cross-candidate comparability).
    """
    total = int(total)
    train_n = int(train_n)
    eval_n = int(eval_n)
    if total < 0 or train_n < 0 or eval_n < 0:
        raise ValueError("counts must be non-negative")
    if train_n + eval_n > total:
        raise ValueError("train_n + eval_n (%d) exceeds total (%d)" % (train_n + eval_n, total))
    train_ids = list(range(train_n))
    eval_ids = list(range(train_n, train_n + eval_n))
    assert not (set(train_ids) & set(eval_ids))
    return train_ids, eval_ids


def score_from_rewards(rewards):
    """Deterministic mean + nonzero-rate over a reward list. No fabrication: an
    empty list yields None metrics, never a fake 0.0 mean."""
    vals = [float(r) for r in rewards]
    n = len(vals)
    mean = (sum(vals) / n) if n else None
    nonzero = sum(1 for v in vals if v > 0.0)
    nonzero_rate = (nonzero / n) if n else None
    return {"n": n, "mean_reward": mean, "nonzero_reward_rate": nonzero_rate}
