#!/usr/bin/env python3
"""Generate deterministic P1_SEARCH_V2 task splits on the T4 (where `datasets` is available).

Runs as a standalone script inside the Colab session BEFORE any probe/training.
Writes:
  - p1_confirmation_split.json   (32 new tasks for P0-vs-P1 confirmation)
  - p1_lora_search_v2_split.json (24 TRAIN_V2 + 32 EVAL_V2 for training)

All task IDs are disjoint from:
  - PROMPT_DEV (12 tasks, pool[0:12] after shuffle seed=42)
  - PROMPT_HOLDOUT (12 tasks, pool[12:24] after shuffle seed=42)
  - which are the SAME as SEARCH_V1 train/eval (12+12)
"""
import json
import os
import sys

def main():
    out_dir = os.environ.get("SPLIT_OUT_DIR", "/content/opentraining/pilot/p1-v2/splits")
    os.makedirs(out_dir, exist_ok=True)

    from datasets import load_dataset

    ds = load_dataset("FineEnvs/SmolDataEnvs", split="train")
    tierset = {"easy", "medium"}
    pool = ds.filter(lambda r: (r.get("difficulty_tier") in tierset))

    # Validity filter (same as worker)
    def validity(bp):
        bp = (bp or "").strip().lower()
        if "california-crime" in bp or "fbi-us__california" in bp:
            return "GOLD_UNREPRODUCIBLE"
        return "VALIDATED"

    pool = pool.filter(lambda r: (validity(r.get("bucket_prefix", "")) == "VALIDATED"))
    pool = pool.shuffle(seed=42)
    total = len(pool)
    print(f"Pool: {total} validated easy/medium tasks", flush=True)

    # Exclude the first 24 (PROMPT_DEV + PROMPT_HOLDOUT = SEARCH_V1 train+eval)
    # These are deterministic: pool[0:12] = dev, pool[12:24] = holdout
    prior_ids = set()
    for i in range(24):
        r = pool[i]
        tid = r.get("task_id", r.get("id", f"idx_{i}"))
        prior_ids.add(tid)
    print(f"Excluded prior: {len(prior_ids)} tasks", flush=True)

    # Remaining pool (skip first 24)
    remaining = pool.select(range(24, total))
    remaining = remaining.shuffle(seed=43)

    # Take: 32 confirmation + 24 TRAIN_V2 + 32 EVAL_V2 = 88 total
    n_confirm, n_train, n_eval = 32, 24, 32
    n_total = n_confirm + n_train + n_eval
    assert len(remaining) >= n_total, f"Pool too small: {len(remaining)} < {n_total}"

    def row_info(r, i):
        return {
            "task_id": r.get("task_id", r.get("id", f"idx_{24+i}")),
            "difficulty": r.get("difficulty_tier", "unknown"),
            "bucket": r.get("bucket_prefix", "unknown"),
            "validity": "VALIDATED",
        }

    confirm = [row_info(remaining[i], i) for i in range(n_confirm)]
    train_v2 = [row_info(remaining[i], i) for i in range(n_confirm, n_confirm + n_train)]
    eval_v2 = [row_info(remaining[i], i) for i in range(n_confirm + n_train, n_total)]

    # Verify disjointness
    c_ids = set(x["task_id"] for x in confirm)
    t_ids = set(x["task_id"] for x in train_v2)
    e_ids = set(x["task_id"] for x in eval_v2)
    assert not (c_ids & t_ids), "confirm/train overlap"
    assert not (c_ids & e_ids), "confirm/eval overlap"
    assert not (t_ids & e_ids), "train/eval overlap"
    assert not (c_ids & prior_ids), "confirm overlaps prior"
    assert not (t_ids & prior_ids), "train overlaps prior"
    assert not (e_ids & prior_ids), "eval overlaps prior"

    from collections import Counter
    for name, items in [("confirm", confirm), ("train_v2", train_v2), ("eval_v2", eval_v2)]:
        d = Counter(x["difficulty"] for x in items)
        print(f"{name}: {len(items)} tasks (easy={d.get('easy',0)}, medium={d.get('medium',0)})", flush=True)

    # Write confirmation split
    with open(os.path.join(out_dir, "p1_confirmation_split.json"), "w") as f:
        json.dump({
            "provenance": "p1_confirmation",
            "seed": 43,
            "excluded_prior": sorted(prior_ids),
            "confirmation_tasks": confirm,
            "task_ids": [x["task_id"] for x in confirm],
        }, f, indent=2)

    # Write V2 training split
    with open(os.path.join(out_dir, "p1_lora_search_v2_split.json"), "w") as f:
        json.dump({
            "provenance": "p1_lora_search_v2",
            "seed": 43,
            "excluded_prior": sorted(prior_ids),
            "train_task_ids": [x["task_id"] for x in train_v2],
            "eval_task_ids": [x["task_id"] for x in eval_v2],
            "train_details": train_v2,
            "eval_details": eval_v2,
            "disjoint": True,
        }, f, indent=2)

    print(f"Splits written to {out_dir}", flush=True)
    print("SPLITS_GENERATED", flush=True)

if __name__ == "__main__":
    main()
