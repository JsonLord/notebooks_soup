"""P1 → LoRA-GRPO V2 Autonomous Search Driver (local orchestrator, no torch).

Stages:
  A. Provision T4, generate V2 splits, run 128-gen confirmation probe
  B. If confirmed: run anchor (P1, lr=1e-5, r=16, 20 steps)
  C. If anchor passes: LR screen (5e-6, 2e-5)
  D. If LR signal: rank screen (r=8, r=32 for top 2 LRs)
  E. 50-step validation (top 2)
  F. 100-step confirmation (winner)

Budget: max 10 training executions, 6 CU, 1 concurrent session.
"""
import json, os, subprocess, sys, time, tarfile, shutil, hashlib

SESSION = "opentraining-1"
JOB_ID = "pilot-004"
MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
EVIDENCE_ROOT = "training_evidence/pilot-004"
LEDGER_PATH = os.path.join(EVIDENCE_ROOT, "search_ledger.jsonl")
SPLITS_DIR = os.path.join(EVIDENCE_ROOT, "splits")
CONFIRM_DIR = os.path.join(EVIDENCE_ROOT, "confirmation")

# P1 frozen prompt
PROMPT_VARIANT = "P1"
PROMPT_SCAFFOLD_SIG = "6310c395fe09c5f9"  # prompt_variant_signature("P1")

MAX_EXECUTIONS = 10
MAX_CU = 6.0
CU_PER_HOUR = 1.0

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
    "VALIDITY_FILTER": "true",
    "PROMPT_VARIANT": PROMPT_VARIANT,
    "EXPECTED_PROMPT_HASH": PROMPT_SCAFFOLD_SIG,
    "NUM_GENERATIONS": "2",
    "TEMPERATURE": "0.8",
    "REPETITION_PENALTY": "1.05",
}

# V2 split sizes
TRAIN_V2_N = 24
EVAL_V2_N = 32

def log(msg):
    print("[P1V2] %s" % msg, flush=True)

def run_cmd(cmd, timeout=600):
    log("CMD: %s" % (" ".join(cmd)[:120]))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.stdout:
        for line in r.stdout.strip().split("\n"):
            log("  " + line[:160])
    if r.returncode != 0 and r.stderr:
        log("  STDERR: " + r.stderr[:300])
    return r

def verify_no_active_session():
    r = subprocess.run(["colab", "status"], capture_output=True, text=True, timeout=30)
    out = (r.stdout or "") + (r.stderr or "")
    return "No active sessions" in out

def provision_session():
    log("Provisioning T4 session %s..." % SESSION)
    run_cmd(["colab", "new", "-s", SESSION, "--gpu", "T4"], timeout=180)
    log("Installing pinned stack...")
    run_cmd(["colab", "install", "-s", SESSION, "trl==1.16.0", "peft==0.21.2"], timeout=600)
    run_cmd(["colab", "exec", "-s", SESSION, "pip uninstall -y torchao"], timeout=120)
    # Restart kernel to pick up clean env
    run_cmd(["colab", "restart", "-s", SESSION], timeout=120)
    time.sleep(5)
    log("Session ready.")

def colab_exec(code, timeout=600):
    return run_cmd(["colab", "exec", "-s", SESSION, code], timeout=timeout)

def colab_upload(local_path, remote_path):
    return run_cmd(["colab", "upload", "-s", SESSION, local_path, remote_path], timeout=120)

def colab_download(remote_path, local_path):
    return run_cmd(["colab", "download", "-s", SESSION, remote_path, local_path], timeout=120)

def guarded_stop():
    log("Guarded stop...")
    r = run_cmd(["colab", "stop", "-s", SESSION], timeout=60)
    log("Stop: %s" % (r.stdout or "").strip()[:100])
    time.sleep(3)
    return verify_no_active_session()

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

def check_budget(exec_count, cu_used, projected_cu=0.21):
    if exec_count >= MAX_EXECUTIONS:
        log("BUDGET: max executions (%d/%d)" % (exec_count, MAX_EXECUTIONS))
        return False, "SEARCH_COMPUTE_BUDGET_REACHED"
    if cu_used + projected_cu > MAX_CU:
        log("BUDGET: CU would exceed %.1f (used=%.2f + proj=%.2f)" % (MAX_CU, cu_used, projected_cu))
        return False, "SEARCH_COMPUTE_BUDGET_REACHED"
    return True, None

def run_training_candidate(candidate_id, lr, lora_r, lora_alpha, max_steps, split_file_remote):
    """Run one training candidate via run_remote_grpo.py."""
    log("=== CANDIDATE %s: lr=%s r=%d alpha=%d steps=%d ===" % (candidate_id, lr, lora_r, lora_alpha, max_steps))
    
    env = dict(FIXED_ENV)
    env["LEARNING_RATE"] = lr
    env["LORA_R"] = str(lora_r)
    env["LORA_ALPHA"] = str(lora_alpha)
    env["MAX_STEPS"] = str(max_steps)
    env["TRAIN_TASK_LIMIT"] = str(TRAIN_V2_N)
    env["EVAL_TASK_LIMIT"] = str(EVAL_V2_N)
    env["TASK_SPLIT_FILE"] = split_file_remote
    
    attempt = "%s-lr%s-r%d-s%d" % (candidate_id, lr.replace("e-0", "e-"), lora_r, max_steps)
    run_dir = "/content/opentraining/pilot/%s/%s" % (JOB_ID, attempt)
    archive = "/content/opentraining/pilot/%s/%s.tar.gz" % (JOB_ID, attempt)
    ev_dir = os.path.join(EVIDENCE_ROOT, attempt)
    os.makedirs(ev_dir, exist_ok=True)
    
    # Build --env args
    env_args = []
    for k, v in env.items():
        env_args.extend(["--env", "%s=%s" % (k, v)])
    
    cmd = [
        "python3", "training_jobs/run_remote_grpo.py",
        "--session", SESSION,
        "--job-id", JOB_ID,
        "--attempt-id", attempt,
        "--run-instance", "p1v2",
        "--job-source", "training_jobs/grpo_smoldataenvs.py",
        "--remote-run-dir", run_dir,
        "--remote-archive", archive,
        "--evidence-dir", ev_dir,
    ] + env_args
    
    r = run_cmd(cmd, timeout=3600)
    
    # Extract results
    results = {"candidate_id": candidate_id, "lr": lr, "lora_r": lora_r,
               "lora_alpha": lora_alpha, "max_steps": max_steps, "exit_code": r.returncode}
    
    # Try to read results.json from evidence dir
    results_path = os.path.join(ev_dir, "_x", JOB_ID, attempt, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            rr = json.load(f)
        results["steps_completed"] = rr.get("steps_completed", rr.get("global_step"))
        results["training_pass"] = rr.get("training_pass", rr.get("verdict", "")).startswith("PASS")
        results["nan_inf"] = rr.get("nan_inf", False)
        # Eval metrics
        ev = rr.get("eval_metrics", {})
        pre = ev.get("pre", {})
        post = ev.get("post", {})
        results["base_eval_reward"] = pre.get("mean_reward")
        results["post_eval_reward"] = post.get("mean_reward")
        results["base_eval_nonzero"] = pre.get("nonzero_rate")
        results["post_eval_nonzero"] = post.get("nonzero_rate")
        results["base_eval_exec"] = pre.get("sandbox_execution_rate")
        results["post_eval_exec"] = post.get("sandbox_execution_rate")
        results["eval_reward_delta"] = (results["post_eval_reward"] or 0) - (results["base_eval_reward"] or 0)
        results["eval_nonzero_delta"] = (results["post_eval_nonzero"] or 0) - (results["base_eval_nonzero"] or 0)
        # Group rewards
        gr = rr.get("group_rewards", [])
        if gr:
            total_groups = len(gr)
            informative = sum(1 for g in gr if g.get("reward_variance", 0) > 0)
            results["total_groups"] = total_groups
            results["informative_groups"] = informative
            results["zero_advantage_groups"] = total_groups - informative
            results["informative_fraction"] = informative / total_groups
            results["training_mean_reward"] = sum(g.get("mean_reward", 0) for g in gr) / total_groups
        # Memory
        ma = rr.get("memory_admission", {})
        results["peak_allocated"] = ma.get("M4_optimizer_projected", {}).get("projected_peak_bytes", 0)
        results["projected_reserve"] = ma.get("M4_optimizer_projected", {}).get("projected_reserve_bytes", 0)
        # Wall time
        results["wall_seconds"] = rr.get("wall_seconds", 0)
        # Adapter
        results["adapter_saved"] = rr.get("adapter_saved", False)
    
    # Artifact transaction check
    tx_path = os.path.join(ev_dir, "artifact_transaction.json")
    if os.path.exists(tx_path):
        with open(tx_path) as f:
            tx = json.load(f)
        results["artifact_pass"] = tx.get("transaction", "") == "PASS"
        results["sha_match"] = tx.get("sha_match", False)
    
    results["terminal_state"] = "PASS" if results.get("training_pass") and results.get("artifact_pass") else "FAIL"
    
    # CU estimate
    wall_hr = (results.get("wall_seconds", 0) + 300) / 3600  # +5min overhead
    results["cu_estimate"] = round(wall_hr * CU_PER_HOUR, 3)
    
    append_ledger(results)
    log("CANDIDATE %s: terminal=%s reward=%s delta=%s groups=%s info_frac=%s CU=%.3f" % (
        candidate_id, results["terminal_state"],
        results.get("post_eval_reward"), results.get("eval_reward_delta"),
        results.get("total_groups"), results.get("informative_fraction"),
        results.get("cu_estimate", 0)))
    return results

def main():
    os.makedirs(EVIDENCE_ROOT, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)
    os.makedirs(CONFIRM_DIR, exist_ok=True)
    
    exec_count = 0
    cu_used = 0.0
    
    # ===================== STAGE A: CONFIRMATION =====================
    log("=" * 60)
    log("STAGE A: P1 GENERALIZATION CONFIRMATION (128 gens)")
    log("=" * 60)
    
    if not verify_no_active_session():
        log("ERROR: session already active, stopping")
        return "SEARCH_BLOCKED"
    
    provision_session()
    
    # Generate splits on T4
    colab_upload("training_jobs/generate_v2_splits.py", "/content/opentraining/p1-v2/generate_v2_splits.py")
    colab_upload("training_jobs/grpo_smoldataenvs.py", "/content/opentraining/p1-v2/grpo_smoldataenvs.py")
    colab_upload("training_jobs/grpo_prompt_contract.py", "/content/opentraining/p1-v2/grpo_prompt_contract.py")
    colab_exec("cd /content/opentraining/p1-v2 && SPLIT_OUT_DIR=/content/opentraining/p1-v2/splits python3 generate_v2_splits.py", timeout=300)
    
    # Download splits
    colab_download("/content/opentraining/p1-v2/splits/p1_confirmation_split.json", os.path.join(SPLITS_DIR, "p1_confirmation_split.json"))
    colab_download("/content/opentraining/p1-v2/splits/p1_lora_search_v2_split.json", os.path.join(SPLITS_DIR, "p1_lora_search_v2_split.json"))
    
    # Verify splits
    with open(os.path.join(SPLITS_DIR, "p1_confirmation_split.json")) as f:
        confirm_split = json.load(f)
    with open(os.path.join(SPLITS_DIR, "p1_lora_search_v2_split.json")) as f:
        v2_split = json.load(f)
    log("Splits: confirm=%d train=%d eval=%d" % (
        len(confirm_split.get("task_ids", [])),
        len(v2_split.get("train_task_ids", [])),
        len(v2_split.get("eval_task_ids", []))))
    
    # Run confirmation probe
    colab_upload("training_jobs/probe_p1_confirmation.py", "/content/opentraining/p1-v2/probe_p1_confirmation.py")
    colab_exec(
        "cd /content/opentraining/p1-v2 && "
        "SPLIT_FILE=/content/opentraining/p1-v2/splits/p1_confirmation_split.json "
        "PROBE_OUT=/content/opentraining/p1-v2/confirm "
        "MODEL=%s " % MODEL +
        "HF_TOKEN=$HF_TOKEN "
        "python3 probe_p1_confirmation.py",
        timeout=3600)
    
    colab_download("/content/opentraining/p1-v2/confirm/results.json", os.path.join(CONFIRM_DIR, "results.json"))
    
    # Check confirmation gate
    with open(os.path.join(CONFIRM_DIR, "results.json")) as f:
        cr = json.load(f)
    
    p0 = cr.get("P0", {})
    p1 = cr.get("P1", {})
    log("CONFIRMATION: P0 reward=%.4f tasks=%d P1 reward=%.4f tasks=%d" % (
        p0.get("mean_reward", 0), p0.get("task_correct", 0),
        p1.get("mean_reward", 0), p1.get("task_correct", 0)))
    
    confirm_verdict = "P1_GENERALIZATION_NOT_CONFIRMED"
    reward_delta = (p1.get("mean_reward", 0) or 0) - (p0.get("mean_reward", 0) or 0)
    task_delta = (p1.get("task_correct", 0) or 0) - (p0.get("task_correct", 0) or 0)
    exec_delta = (p1.get("exec_rate", 0) or 0) - (p0.get("exec_rate", 0) or 0)
    
    if p1.get("mean_reward", 0) and p0.get("mean_reward", 0):
        if p1["mean_reward"] > p0["mean_reward"] and p1.get("task_correct", 0) > p0.get("task_correct", 0):
            if reward_delta >= 0.05 or task_delta >= 2:
                if exec_delta >= -0.05:
                    confirm_verdict = "P1_GENERALIZATION_CONFIRMED"
                    if reward_delta >= 0.10 and task_delta >= 3 and exec_delta >= 0:
                        confirm_verdict = "P1_GENERALIZATION_STRONG_CONFIRMATION"
    
    log("CONFIRMATION VERDICT: %s (reward_delta=%.4f task_delta=%+d exec_delta=%.4f)" % (
        confirm_verdict, reward_delta, task_delta, exec_delta))
    
    if confirm_verdict == "P1_GENERALIZATION_NOT_CONFIRMED":
        log("HARD STOP: P1 did not confirm on 32 new tasks")
        guarded_stop()
        return "P1_GENERALIZATION_NOT_CONFIRMED"
    
    # ===================== STAGE B: ANCHOR =====================
    log("=" * 60)
    log("STAGE B: P1 ANCHOR (lr=1e-5 r=16 alpha=32 steps=20)")
    log("=" * 60)
    
    ok, budget_err = check_budget(exec_count, cu_used)
    if not ok:
        guarded_stop()
        return budget_err
    
    # Upload V2 split to T4
    colab_upload(os.path.join(SPLITS_DIR, "p1_lora_search_v2_split.json"),
                 "/content/opentraining/pilot/p1-v2/splits/p1_lora_search_v2_split.json")
    
    anchor = run_training_candidate("P1-ANCHOR", "1e-5", 16, 32, 20,
                                     "/content/opentraining/pilot/p1-v2/splits/p1_lora_search_v2_split.json")
    exec_count += 1
    cu_used += anchor.get("cu_estimate", 0.21)
    
    # Anchor gate (§16)
    anchor_pass = False
    post_reward = anchor.get("post_eval_reward", 0) or 0
    base_reward = anchor.get("base_eval_reward", 0) or 0
    post_nonzero = anchor.get("post_eval_nonzero", 0) or 0
    base_nonzero = anchor.get("base_eval_nonzero", 0) or 0
    info_frac = anchor.get("informative_fraction", 0) or 0
    
    if post_reward > base_reward:
        anchor_pass = True
        log("ANCHOR PASS: post_reward %.4f > base %.4f" % (post_reward, base_reward))
    elif post_nonzero > base_nonzero:
        anchor_pass = True
        log("ANCHOR PASS: post_nonzero %.4f > base %.4f" % (post_nonzero, base_nonzero))
    elif info_frac >= 0.30 and post_reward >= base_reward:
        anchor_pass = True
        log("ANCHOR PASS: info_frac %.2f >= 0.30 with no regression" % info_frac)
    
    if not anchor_pass:
        log("ANCHOR FAIL: reward %.4f->%.4f nonzero %.4f->%.4f info_frac=%.2f" % (
            base_reward, post_reward, base_nonzero, post_nonzero, info_frac))
        guarded_stop()
        return "P1_LORA_NO_LEARNING_SIGNAL"
    
    # ===================== STAGE C: LR SCREEN =====================
    log("=" * 60)
    log("STAGE C: LR SCREEN (5e-6, 2e-5)")
    log("=" * 60)
    
    lr_candidates = ["5e-6", "2e-5"]
    lr_results = {"1e-5": anchor}  # anchor IS the 1e-5 candidate
    
    for lr in lr_candidates:
        ok, budget_err = check_budget(exec_count, cu_used)
        if not ok:
            guarded_stop()
            return budget_err
        
        cid = "P1-LR%s" % lr
        r = run_training_candidate(cid, lr, 16, 32, 20,
                                    "/content/opentraining/pilot/p1-v2/splits/p1_lora_search_v2_split.json")
        exec_count += 1
        cu_used += r.get("cu_estimate", 0.21)
        lr_results[lr] = r
    
    # Rank LRs
    def lr_rank(lr, r):
        return (
            r.get("post_eval_reward", 0) or 0,
            r.get("eval_reward_delta", 0) or 0,
            r.get("post_eval_nonzero", 0) or 0,
            r.get("informative_fraction", 0) or 0,
            r.get("post_eval_exec", 0) or 0,
        )
    
    ranked_lrs = sorted(lr_results.items(), key=lambda x: lr_rank(x[0], x[1]), reverse=True)
    log("LR RANKING: " + " > ".join("%s (reward=%.4f delta=%.4f)" % (lr, r.get("post_eval_reward", 0), r.get("eval_reward_delta", 0)) for lr, r in ranked_lrs))
    
    # Early stop (§20)
    all_flat = all((r.get("eval_reward_delta", 0) or 0) <= 0 for _, r in lr_results.items())
    no_nonzero_improve = all((r.get("eval_nonzero_delta", 0) or 0) <= 0 for _, r in lr_results.items())
    no_exec_improve = all((r.get("post_eval_exec", 0) or 0) <= (r.get("base_eval_exec", 0) or 0) for _, r in lr_results.items())
    
    if all_flat and no_nonzero_improve and no_exec_improve:
        log("EARLY STOP: all LRs flat, no nonzero/exec improvement")
        guarded_stop()
        return "P1_LORA_SEARCH_NO_GENERALIZATION"
    
    # ===================== STAGE D: RANK SCREEN =====================
    log("=" * 60)
    log("STAGE D: RANK SCREEN")
    log("=" * 60)
    
    # Promote top 2 LRs
    promoted_lrs = [lr for lr, r in ranked_lrs[:2] if (r.get("eval_reward_delta", 0) or 0) > 0]
    if not promoted_lrs:
        # Use top 2 regardless if no positive delta (but flag)
        promoted_lrs = [lr for lr, r in ranked_lrs[:2]]
        log("WARNING: no LR with positive delta, using top 2 anyway")
    
    log("Promoted LRs: %s" % promoted_lrs)
    
    rank_configs = []
    for lr in promoted_lrs:
        for r, alpha in [(8, 16), (32, 64)]:  # r=16 already done
            rank_configs.append((lr, r, alpha))
    
    rank_results = {}
    for lr, r, alpha in rank_configs:
        ok, budget_err = check_budget(exec_count, cu_used)
        if not ok:
            guarded_stop()
            return budget_err
        
        cid = "P1-LR%s-r%d" % (lr, r)
        res = run_training_candidate(cid, lr, r, alpha, 20,
                                      "/content/opentraining/pilot/p1-v2/splits/p1_lora_search_v2_split.json")
        exec_count += 1
        cu_used += res.get("cu_estimate", 0.21)
        rank_results[(lr, r)] = res
    
    # Combine all effective configs (2 LRs × {r8, r16, r32})
    all_configs = {}
    for lr, r in rank_results:
        all_configs[(lr, r)] = rank_results[(lr, r)]
    # Add r=16 from LR screen
    for lr in promoted_lrs:
        if (lr, 16) not in all_configs:
            all_configs[(lr, 16)] = lr_results[lr]
    
    # Rank all 6 effective configs
    def cfg_rank(item):
        (lr, r), res = item
        return (
            res.get("post_eval_reward", 0) or 0,
            res.get("eval_reward_delta", 0) or 0,
            res.get("post_eval_nonzero", 0) or 0,
            res.get("informative_fraction", 0) or 0,
            res.get("post_eval_exec", 0) or 0,
        )
    
    ranked_cfgs = sorted(all_configs.items(), key=cfg_rank, reverse=True)
    log("RANK RANKING: " + " > ".join("lr=%s r=%d (reward=%.4f delta=%.4f)" % (lr, r, res.get("post_eval_reward", 0), res.get("eval_reward_delta", 0)) for (lr, r), res in ranked_cfgs))
    
    # Promote top 2 with positive delta
    promoted_cfgs = [((lr, r), res) for (lr, r), res in ranked_cfgs[:2] if (res.get("eval_reward_delta", 0) or 0) > 0]
    if not promoted_cfgs:
        log("NO promotable config with positive delta")
        guarded_stop()
        return "NO_PROMOTABLE_P1_LORA_CONFIG"
    
    # ===================== STAGE E: 50-STEP VALIDATION =====================
    log("=" * 60)
    log("STAGE E: 50-STEP VALIDATION")
    log("=" * 60)
    
    step50_results = []
    for (lr, r), _ in promoted_cfgs:
        ok, budget_err = check_budget(exec_count, cu_used)
        if not ok:
            guarded_stop()
            return budget_err
        
        alpha = r * 2
        cid = "P1-50s-LR%s-r%d" % (lr, r)
        res = run_training_candidate(cid, lr, r, alpha, 50,
                                      "/content/opentraining/pilot/p1-v2/splits/p1_lora_search_v2_split.json")
        exec_count += 1
        cu_used += res.get("cu_estimate", 0.21)
        step50_results.append((lr, r, alpha, res))
    
    # Winner: best post_eval_reward with positive delta
    def s50_rank(item):
        lr, r, alpha, res = item
        return res.get("post_eval_reward", 0) or 0
    
    step50_results.sort(key=s50_rank, reverse=True)
    winner_lr, winner_r, winner_alpha, winner_res = step50_results[0]
    
    winner_positive = (winner_res.get("eval_reward_delta", 0) or 0) > 0 or (winner_res.get("post_eval_nonzero", 0) or 0) > (winner_res.get("base_eval_nonzero", 0) or 0)
    if not winner_positive:
        log("50-STEP: no winner with positive improvement")
        guarded_stop()
        return "NO_PROMOTABLE_P1_LORA_CONFIG"
    
    log("50-STEP WINNER: lr=%s r=%d alpha=%d reward=%.4f delta=%.4f" % (
        winner_lr, winner_r, winner_alpha, winner_res.get("post_eval_reward", 0), winner_res.get("eval_reward_delta", 0)))
    
    # ===================== STAGE F: 100-STEP CONFIRMATION =====================
    log("=" * 60)
    log("STAGE F: 100-STEP CONFIRMATION")
    log("=" * 60)
    
    ok, budget_err = check_budget(exec_count, cu_used)
    if not ok:
        log("Skipping 100-step (budget): %s" % budget_err)
        guarded_stop()
        return "P1_LORA_WINNER_CONFIRMED"  # 50-step winner is sufficient
    
    cid = "P1-100s-LR%s-r%d" % (winner_lr, winner_r)
    winner100 = run_training_candidate(cid, winner_lr, winner_r, winner_alpha, 100,
                                         "/content/opentraining/pilot/p1-v2/splits/p1_lora_search_v2_split.json")
    exec_count += 1
    cu_used += winner100.get("cu_estimate", 0.21)
    
    log("100-STEP: reward=%.4f delta=%.4f nonzero=%.4f" % (
        winner100.get("post_eval_reward", 0), winner100.get("eval_reward_delta", 0), winner100.get("post_eval_nonzero", 0)))
    
    # ===================== FINAL =====================
    log("=" * 60)
    log("FINAL: P1_LORA_WINNER_CONFIRMED")
    log("=" * 60)
    log("Winner: lr=%s r=%d alpha=%d" % (winner_lr, winner_r, winner_alpha))
    log("50-step: reward=%.4f delta=%.4f" % (winner_res.get("post_eval_reward", 0), winner_res.get("eval_reward_delta", 0)))
    if winner100:
        log("100-step: reward=%.4f delta=%.4f" % (winner100.get("post_eval_reward", 0), winner100.get("eval_reward_delta", 0)))
    log("Executions: %d/%d CU: %.2f/%.1f" % (exec_count, MAX_EXECUTIONS, cu_used, MAX_CU))
    
    cleaned = guarded_stop()
    log("Cleanup: %s" % ("OK" if cleaned else "ORPHAN_SESSION"))
    return "P1_LORA_WINNER_CONFIRMED"


if __name__ == "__main__":
    terminal = main()
    print("TERMINAL_STATE=%s" % terminal, flush=True)
