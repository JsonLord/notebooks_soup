#!/usr/bin/env python3
"""Bounded 1.7B LoRA-GRPO search driver (local orchestrator, no torch).

Drives run_remote_grpo.py for each candidate, extracts results, ranks,
promotes survivors, and enforces the 6 CU / 10-execution budget.
"""
import json, os, subprocess, sys, time, tarfile, shutil, hashlib

SESSION = "opentraining-1"
JOB_ID = "pilot-002"
MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
EVIDENCE_ROOT = "training_evidence/pilot-002"
LEDGER_PATH = os.path.join(EVIDENCE_ROOT, "search_ledger.jsonl")
SPLIT_PATH = os.path.join(EVIDENCE_ROOT, "search_task_split.json")
BASE_EVAL_PATH = os.path.join(EVIDENCE_ROOT, "base_eval.json")

MAX_EXECUTIONS = 10
MAX_CU = 6.0
CU_PER_HOUR = 1.0  # T4 = 1 CU/hr

FIXED_ENV = {
    "TRAINING_METHOD": "lora",
    "LORA_DROPOUT": "0.05",
    "LORA_TARGET_MODE": "all-linear",
    "MODEL": MODEL,
    "MAX_NEW_TOKENS": "256",
    "FP16": "true",
    "BF16": "false",
    "USE_VLLM": "false",
    "SEED": "42",
    "TRAIN_TASK_LIMIT": "12",
    "EVAL_TASK_LIMIT": "12",
    "VALIDITY_FILTER": "true",
}

def log(msg):
    print("[SEARCH] %s" % msg, flush=True)

def ensure_dirs():
    os.makedirs(EVIDENCE_ROOT, exist_ok=True)

def append_ledger(record):
    with open(LEDGER_PATH, "a") as fh:
        fh.write(json.dumps(record) + "\n")

def read_ledger():
    if not os.path.exists(LEDGER_PATH):
        return []
    records = []
    with open(LEDGER_PATH) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records

def check_budget(exec_count, cu_used, projected_cu=0.15):
    if exec_count >= MAX_EXECUTIONS:
        log("BUDGET: max executions reached (%d/%d)" % (exec_count, MAX_EXECUTIONS))
        return False
    if cu_used + projected_cu > MAX_CU:
        log("BUDGET: CU would exceed %s (used=%.2f + projected=%.2f)" % (MAX_CU, cu_used, projected_cu))
        return False
    return True

def verify_no_active_session():
    r = subprocess.run(["colab", "status", "-s", SESSION], capture_output=True, text=True, timeout=30)
    return r.returncode != 0  # no session = returncode != 0

def run_candidate(candidate_id, lr, lora_r, lora_alpha, steps, base_eval):
    """Run one candidate via run_remote_grpo.py. Returns (results_dict, cu_elapsed)."""
    attempt_id = "%s-lr%0.0e-r%d-s%d" % (candidate_id, lr, lora_r, steps)
    remote_run_dir = "/content/opentraining/pilot/%s/%s" % (JOB_ID, attempt_id)
    remote_archive = "/content/opentraining/artifacts/%s-%s.tar.gz" % (JOB_ID, attempt_id)
    local_evidence = os.path.join(EVIDENCE_ROOT, attempt_id)

    env = dict(FIXED_ENV)
    env["LEARNING_RATE"] = str(lr)
    env["LORA_R"] = str(lora_r)
    env["LORA_ALPHA"] = str(lora_alpha)
    env["MAX_STEPS"] = str(steps)

    log("RUNNING %s: lr=%s r=%d alpha=%d steps=%d" % (candidate_id, lr, lora_r, lora_alpha, steps))
    t0 = time.time()

    cmd = ["/opt/venv/bin/python", "training_jobs/run_remote_grpo.py",
           "--session", SESSION,
           "--job-id", JOB_ID,
           "--attempt-id", attempt_id,
           "--run-instance", "search-20261026",
           "--job-source", "training_jobs/grpo_smoldataenvs.py",
           "--remote-run-dir", remote_run_dir,
           "--remote-archive", remote_archive,
           "--evidence-dir", local_evidence]
    for k, v in sorted(env.items()):
        cmd += ["--env", "%s=%s" % (k, v)]

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    wall = time.time() - t0
    cu_elapsed = (wall / 3600.0) * CU_PER_HOUR

    if r.returncode != 0:
        log("CANDIDATE %s FAILED rc=%d" % (candidate_id, r.returncode))
        log(r.stdout[-500:] if r.stdout else "")
        log(r.stderr[-500:] if r.stderr else "")
        return None, cu_elapsed

    # Parse the orchestration result
    try:
        orch = json.loads(r.stdout.strip().split("\n")[-1])
    except Exception:
        orch = {}

    log("CANDIDATE %s complete: wall=%.0fs cu=%.3f train=%s txn=%s stop=%s" % (
        candidate_id, wall, cu_elapsed, orch.get("training_status"),
        orch.get("artifact_transaction_status"), orch.get("stop_status")))

    # Extract archive and read results
    results = extract_candidate_results(local_evidence, attempt_id)
    if results is None:
        log("CANDIDATE %s: could not extract results" % candidate_id)
        return None, cu_elapsed

    results["candidate_id"] = candidate_id
    results["lr"] = lr
    results["lora_r"] = lora_r
    results["lora_alpha"] = lora_alpha
    results["steps"] = steps
    results["wall_time"] = round(wall, 1)
    results["cu_elapsed"] = round(cu_elapsed, 4)

    # Compute deltas vs BASE_EVAL
    if base_eval:
        results["eval_reward_delta"] = round((results.get("post_eval_mean_reward") or 0) - base_eval.get("mean_reward", 0), 4)
        results["eval_nonzero_delta"] = round((results.get("post_eval_nonzero_rate") or 0) - base_eval.get("nonzero_rate", 0), 4)
        results["eval_exec_delta"] = round((results.get("post_eval_execution_rate") or 0) - base_eval.get("execution_rate", 0), 4)

    # Eligibility check (§9)
    eligible = True
    reasons = []
    if results.get("global_step", 0) < steps:
        eligible = False; reasons.append("incomplete steps")
    if results.get("nan_inf_detected"):
        eligible = False; reasons.append("NaN/Inf")
    if not results.get("base_frozen", True):
        eligible = False; reasons.append("base not frozen")
    if not results.get("adapter_saved"):
        eligible = False; reasons.append("adapter not saved")
    if orch.get("artifact_transaction_status") != "ARTIFACT_TRANSACTION_PASS":
        eligible = False; reasons.append("txn fail")
    if orch.get("stop_status") != "SESSION_STOP_PASS":
        eligible = False; reasons.append("stop fail")
    if results.get("informative_group_count", 0) < 1:
        eligible = False; reasons.append("no GRPO signal")
    results["eligible"] = eligible
    results["ineligible_reasons"] = reasons
    if reasons:
        log("CANDIDATE %s INELIGIBLE: %s" % (candidate_id, "; ".join(reasons)))

    # Adapter SHA
    adapter_sha = compute_adapter_sha(local_evidence, attempt_id)
    results["adapter_sha"] = adapter_sha

    append_ledger(results)
    return results, cu_elapsed

def extract_candidate_results(local_evidence, attempt_id):
    """Extract archive from evidence dir and parse results.json."""
    archives = [f for f in os.listdir(local_evidence) if f.endswith(".tar.gz")]
    if not archives:
        return None
    archive = os.path.join(local_evidence, archives[0])
    xdir = os.path.join(local_evidence, "_x")
    if os.path.exists(xdir):
        shutil.rmtree(xdir)
    os.makedirs(xdir, exist_ok=True)
    try:
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(xdir)
    except Exception as e:
        log("Archive extract failed: %s" % e)
        return None

    # Find results.json
    results_path = os.path.join(xdir, JOB_ID, attempt_id, "results.json")
    if not os.path.exists(results_path):
        # Try alternative paths
        for root, _, files in os.walk(xdir):
            if "results.json" in files:
                results_path = os.path.join(root, "results.json")
                break
    if not os.path.exists(results_path):
        return None

    with open(results_path) as fh:
        d = json.load(fh)

    pre = d.get("pre_train_eval") or {}
    post = d.get("post_train_eval") or {}
    return {
        "status": d.get("status"),
        "global_step": d.get("global_step"),
        "nan_inf_detected": d.get("nan_inf_detected"),
        "peak_allocated_bytes": d.get("peak_allocated_bytes"),
        "peak_reserved_bytes": d.get("peak_reserved_bytes"),
        "host_ram_peak_mb": d.get("host_ram_peak_mb"),
        "mean_reward": d.get("mean_reward"),
        "reward_std": d.get("reward_std"),
        "informative_group_count": d.get("informative_group_count"),
        "zero_advantage_group_count": d.get("zero_advantage_group_count"),
        "nonzero_reward_count": d.get("nonzero_reward_count"),
        "grpo_signal_present": d.get("grpo_signal_present"),
        "memory_feasible": d.get("memory_feasible"),
        "training_execution_valid": d.get("training_execution_valid"),
        "quality_improved": d.get("quality_improved"),
        "adapter_saved": d.get("adapter_saved"),
        "adapter_size_bytes": d.get("adapter_size_bytes"),
        "base_frozen": (d.get("lora_param") or {}).get("base_frozen"),
        "pre_eval_mean_reward": pre.get("mean_reward"),
        "post_eval_mean_reward": post.get("mean_reward"),
        "pre_eval_nonzero_rate": pre.get("nonzero_reward_rate"),
        "post_eval_nonzero_rate": post.get("nonzero_reward_rate"),
        "pre_eval_execution_rate": pre.get("sandbox_execution_rate"),
        "post_eval_execution_rate": post.get("sandbox_execution_rate"),
        "training_groups": d.get("nonzero_reward_count", 0) // 2 if d.get("nonzero_reward_count") else 0,
        "lora_verdict": d.get("lora_verdict"),
        "observed_reserve_gb": d.get("observed_reserve_gb"),
        "reference_behavior": (d.get("lora_mem") or {}).get("reference_behavior"),
    }

def compute_adapter_sha(local_evidence, attempt_id):
    """Compute SHA-256 of the adapter weights file."""
    xdir = os.path.join(local_evidence, "_x")
    for root, _, files in os.walk(xdir):
        for f in files:
            if f.startswith("adapter_model") and f.endswith(".safetensors"):
                path = os.path.join(root, f)
                h = hashlib.sha256()
                with open(path, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                return h.hexdigest()[:16]
    return None

def rank_candidates(candidates, base_eval):
    """Rank eligible candidates by held-out reward (§10)."""
    eligible = [c for c in candidates if c.get("eligible")]
    def score(c):
        return (
            c.get("post_eval_mean_reward") or 0,
            c.get("eval_reward_delta") or 0,
            c.get("post_eval_nonzero_rate") or 0,
            (c.get("informative_group_count", 0) / max(1, c.get("training_groups", 1))),
            c.get("post_eval_execution_rate") or 0,
            -(c.get("wall_time") or 0),
        )
    eligible.sort(key=score, reverse=True)
    return eligible

def main():
    ensure_dirs()
    log("=== 1.7B LoRA-GRPO BOUNDED SEARCH START ===")
    log("Budget: %d executions, %.1f CU" % (MAX_EXECUTIONS, MAX_CU))

    exec_count = 0
    cu_used = 0.0
    ledger = []

    # Check for existing ledger (resume)
    if os.path.exists(LEDGER_PATH):
        ledger = read_ledger()
        exec_count = len(ledger)
        cu_used = sum(r.get("cu_elapsed", 0) for r in ledger)
        log("Resumed: %d prior executions, %.2f CU used" % (exec_count, cu_used))

    # ---- BASE EVAL (§7): use the lora-001 pilot's pre_train_eval ----
    # The pilot (lora-001) already ran pre_train_eval on 6 eval tasks.
    # For the search we need 12 eval tasks. We'll run a dedicated base eval
    # as the first execution (inference-only, no training) — but that's not
    # how run_remote_grpo.py works (it always trains).
    #
    # Alternative: use the pilot's pre_eval as an approximation, then each
    # candidate's own pre_eval (on the same 12-task eval) serves as the
    # authoritative baseline. Since all candidates use the same seed + eval
    # split, their pre_evals should be identical (base model, no LoRA).
    # We'll take the first candidate's pre_eval as BASE_EVAL.
    base_eval = None
    if os.path.exists(BASE_EVAL_PATH):
        with open(BASE_EVAL_PATH) as fh:
            base_eval = json.load(fh)
        log("BASE_EVAL loaded: mean=%s nonzero=%s exec=%s" % (
            base_eval.get("mean_reward"), base_eval.get("nonzero_rate"),
            base_eval.get("execution_rate")))

    # ---- ROUND 1: LR Screen (§11) ----
    log("=== ROUND 1: LR SCREEN (r=16, alpha=32, 20 steps) ===")
    r1_candidates = [
        ("R1-A", 5e-6, 16, 32, 20),
        ("R1-B", 1e-5, 16, 32, 20),
        ("R1-C", 2e-5, 16, 32, 20),
    ]

    r1_results = []
    for cid, lr, r, a, s in r1_candidates:
        # Check if already done
        existing = [x for x in ledger if x.get("candidate_id") == cid]
        if existing:
            log("SKIP %s (already in ledger)" % cid)
            r1_results.append(existing[0])
            continue

        if not check_budget(exec_count, cu_used):
            log("TERMINAL: SEARCH_COMPUTE_BUDGET_REACHED")
            break

        if not verify_no_active_session():
            log("Wait: session %s still active" % SESSION)
            time.sleep(30)
            if not verify_no_active_session():
                log("SESSION_TIMEOUT: cannot start candidate")
                break

        results, cu = run_candidate(cid, lr, r, a, s, base_eval)
        exec_count += 1
        cu_used += cu
        if results:
            r1_results.append(results)
            # Set BASE_EVAL from first candidate's pre_eval
            if base_eval is None and results.get("pre_eval_mean_reward") is not None:
                base_eval = {
                    "mean_reward": results["pre_eval_mean_reward"],
                    "nonzero_rate": results.get("pre_eval_nonzero_rate"),
                    "execution_rate": results.get("pre_eval_execution_rate"),
                }
                with open(BASE_EVAL_PATH, "w") as fh:
                    json.dump(base_eval, fh, indent=2)
                log("BASE_EVAL set from %s pre_eval: mean=%s" % (cid, base_eval["mean_reward"]))

    # ROUND-1 EARLY TERMINATION (§12)
    log("=== ROUND-1 EARLY TERMINATION CHECK ===")
    r1_eligible = [r for r in r1_results if r.get("eligible")]
    if r1_eligible and base_eval:
        all_no_improve = all(
            (r.get("post_eval_mean_reward") or 0) <= base_eval.get("mean_reward", 0)
            for r in r1_eligible
        )
        any_exec_improve = any(
            (r.get("post_eval_execution_rate") or 0) > base_eval.get("execution_rate", 0)
            or (r.get("post_eval_nonzero_rate") or 0) > base_eval.get("nonzero_rate", 0)
            for r in r1_eligible
        )
        if all_no_improve and not any_exec_improve:
            log("TERMINAL: LORA_SEARCH_NO_GENERALIZATION_SIGNAL")
            log("All R1 candidates flat or worse, no execution/nonzero improvement.")
            return

    # ROUND-1 PROMOTION (§13)
    r1_ranked = rank_candidates(r1_results, base_eval)
    promoted_lrs = r1_ranked[:2]
    log("PROMOTED LRs: %s" % ["%s(lr=%g)" % (r["candidate_id"], r["lr"]) for r in promoted_lrs])

    if len(promoted_lrs) < 2:
        log("TERMINAL: insufficient promotable LRs")
        return

    # ---- ROUND 2: Rank Screen (§14) ----
    log("=== ROUND 2: RANK SCREEN (20 steps) ===")
    r2_candidates = []
    for i, pr in enumerate(promoted_lrs):
        lr = pr["lr"]
        # r=16 already exists from R1; add r=8 and r=32
        r2_candidates.append(("R2-%d-r8" % i, lr, 8, 16, 20))
        r2_candidates.append(("R2-%d-r32" % i, lr, 32, 64, 20))

    r2_results = []
    for cid, lr, r, a, s in r2_candidates:
        existing = [x for x in ledger if x.get("candidate_id") == cid]
        if existing:
            log("SKIP %s (already in ledger)" % cid)
            r2_results.append(existing[0])
            continue

        if not check_budget(exec_count, cu_used):
            log("TERMINAL: SEARCH_COMPUTE_BUDGET_REACHED")
            break

        if not verify_no_active_session():
            time.sleep(30)
            if not verify_no_active_session():
                log("SESSION_TIMEOUT: cannot start candidate")
                break

        # R32 memory gate (§15): r32 has more params, check projected reserve
        results, cu = run_candidate(cid, lr, r, a, s, base_eval)
        exec_count += 1
        cu_used += cu
        if results:
            # Check r32 memory
            if r == 32 and results.get("observed_reserve_gb") and results["observed_reserve_gb"] < 2.5:
                results["r32_memory_blocked"] = True
                results["eligible"] = False
                results["ineligible_reasons"].append("R32_MEMORY_BLOCKED")
                log("R32_MEMORY_BLOCKED: reserve=%.2fGB < 2.5GB" % results["observed_reserve_gb"])
            r2_results.append(results)

    # ROUND-2 PROMOTION (§16): rank all 6 effective combos
    all_r2 = r1_results[:2] + r2_results  # r16 from R1 + r8/r32 from R2
    r2_ranked = rank_candidates(all_r2, base_eval)
    top2_configs = r2_ranked[:2]
    log("PROMOTED top-2 configs: %s" % [
        "%s(lr=%g,r=%d)" % (r["candidate_id"], r["lr"], r["lora_r"]) for r in top2_configs])

    if len(top2_configs) < 2:
        log("TERMINAL: insufficient promotable configs")
        return

    # ---- ROUND 3: 50-step validation (§17) ----
    log("=== ROUND 3: 50-STEP VALIDATION ===")
    r3_results = []
    for i, cfg in enumerate(top2_configs):
        cid = "R3-%d" % i
        existing = [x for x in ledger if x.get("candidate_id") == cid]
        if existing:
            log("SKIP %s (already in ledger)" % cid)
            r3_results.append(existing[0])
            continue

        if not check_budget(exec_count, cu_used, projected_cu=0.3):
            log("TERMINAL: SEARCH_COMPUTE_BUDGET_REACHED")
            break

        if not verify_no_active_session():
            time.sleep(30)
            if not verify_no_active_session():
                log("SESSION_TIMEOUT: cannot start candidate")
                break

        results, cu = run_candidate(cid, cfg["lr"], cfg["lora_r"], cfg["lora_alpha"], 50, base_eval)
        exec_count += 1
        cu_used += cu
        if results:
            r3_results.append(results)

    # OVERFITTING CHECK (§18)
    for r in r3_results:
        train_r = r.get("mean_reward") or 0
        eval_r = r.get("post_eval_mean_reward") or 0
        base_r = base_eval.get("mean_reward", 0) if base_eval else 0
        if train_r > 0.5 and eval_r < base_r:
            r["possible_overfit"] = True
            log("POSSIBLE_OVERFIT in %s: train=%.2f > 0.5 but eval=%.2f < base=%.2f" % (
                r["candidate_id"], train_r, eval_r, base_r))

    # WINNER SELECTION (§19)
    r3_ranked = rank_candidates(r3_results, base_eval)
    provisional_winner = r3_ranked[0] if r3_ranked else None

    if provisional_winner and base_eval:
        has_improvement = (
            (provisional_winner.get("post_eval_mean_reward") or 0) > base_eval.get("mean_reward", 0)
            or (provisional_winner.get("post_eval_nonzero_rate") or 0) > base_eval.get("nonzero_rate", 0)
        )
        exec_ok = (provisional_winner.get("post_eval_execution_rate") or 0) >= (base_eval.get("execution_rate", 0) - 0.1)
        if not has_improvement or not exec_ok:
            log("TERMINAL: NO_PROMOTABLE_LORA_CONFIG")
            log("Winner %s: has_improvement=%s exec_ok=%s" % (provisional_winner["candidate_id"], has_improvement, exec_ok))
            return
    elif not provisional_winner:
        log("TERMINAL: NO_PROMOTABLE_LORA_CONFIG")
        return

    log("PROVISIONAL WINNER: %s (lr=%g, r=%d)" % (
        provisional_winner["candidate_id"], provisional_winner["lr"], provisional_winner["lora_r"]))

    # ---- ROUND 4: 100-step confirmation (§20) ----
    log("=== ROUND 4: 100-STEP CONFIRMATION ===")
    if check_budget(exec_count, cu_used, projected_cu=0.6):
        cid = "R4-WINNER"
        existing = [x for x in ledger if x.get("candidate_id") == cid]
        if existing:
            log("SKIP %s (already in ledger)" % cid)
            r4_results = [existing[0]]
        else:
            if not verify_no_active_session():
                time.sleep(30)
            results, cu = run_candidate(cid, provisional_winner["lr"],
                                          provisional_winner["lora_r"],
                                          provisional_winner["lora_alpha"], 100, base_eval)
            exec_count += 1
            cu_used += cu
            r4_results = [results] if results else []
    else:
        log("BUDGET: skipping R4 (insufficient CU for 100-step run)")
        r4_results = []

    # ---- FINAL REPORT DATA ----
    log("=== SEARCH COMPLETE ===")
    final_ledger = read_ledger()
    log("Total executions: %d, CU used: %.2f" % (len(final_ledger), cu_used))
    log("Terminal: LORA_SEARCH_WINNER_CONFIRMED" if r4_results else "Terminal: NO_R4_CONFIRMATION")

    # Write summary
    summary = {
        "total_executions": len(final_ledger),
        "total_cu": round(cu_used, 3),
        "base_eval": base_eval,
        "candidates": final_ledger,
    }
    with open(os.path.join(EVIDENCE_ROOT, "search_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    log("Summary written to %s" % os.path.join(EVIDENCE_ROOT, "search_summary.json"))

if __name__ == "__main__":
    main()
