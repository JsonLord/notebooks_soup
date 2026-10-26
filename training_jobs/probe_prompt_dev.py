#!/usr/bin/env python3
"""Prompt reasoning redesign: inference-only probe.

Runs P0-P3 prompt variants on the 12-task PROMPT_DEVELOPMENT_SET (the search
train split), 2 completions each = 96 generations. Records §9 metrics per
variant. No training, no LoRA, no GRPO.

Runs on T4 via colab exec -f. Imports key functions from the worker.
"""
import json, os, sys, time, hashlib, math

def emit(tag, val=""):
    print("[%s] %s" % (tag, val), flush=True)

def blocked(cls, verdict):
    emit("BLOCKED", cls)
    emit("VERDICT", verdict)
    with open("/content/opentraining/pilot/pilot-003/prompt-dev/results.json", "w") as fh:
        json.dump({"status": "blocked", "classification": cls, "verdict": verdict}, fh, indent=2)
    sys.exit(1)

# ---- config ----
SEED = 42
TEMPERATURE = 0.8
REPETITION_PENALTY = 1.05
MAX_NEW_TOKENS = 256
NRS = 2
VARIANTS = ["P0", "P1", "P2", "P3"]
DEV_TASK_N = 12
EVAL_TASK_N = 12
DS = "FineEnvs/SmolDataEnvs"
MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
RUN = "/content/opentraining/pilot/pilot-003/prompt-dev"
os.makedirs(RUN, exist_ok=True)

# ---- import worker functions (same dir, same file) ----
sys.path.insert(0, "/content/opentraining/pilot/pilot-003/prompt-dev")
# The worker is uploaded as grpo_smoldataenvs.py in the run dir
# But it's a monolithic script, not a module. We'll extract the needed functions.
# Actually, let's import from the prompt contract module instead.
from grpo_prompt_contract import (
    build_prompt, extract_code, ast_preflight, inspect_staged_contract,
    stage_bucket, task_validity_for, classify_form_error, classify_truncation,
)

t0 = time.time()

# ---- torch + model ----
emit("LOADING", "torch + model")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()
emit("MODEL_LOADED", "%.3f GB" % (torch.cuda.memory_allocated() / 1e9))

# ---- HF token + sandbox ----
_tok = os.environ.get("HF_TOKEN") or ""
if not _tok:
    blocked("HF_AUTH_INJECTION_FAILED", "HF_AUTH_INJECTION_FAILED")

from huggingface_hub import HfApi, Sandbox
w = HfApi().whoami(token=_tok)
_wname = w.get("name", "unknown") if isinstance(w, dict) else getattr(w, "name", "unknown")
emit("WHOAMI", _wname)

SB_IMAGE = os.environ.get("SB_IMAGE", "docker.io/savatar101/env-data-agent-train:base")
sb = Sandbox.create(image=SB_IMAGE, flavor="cpu-basic", idle_timeout="30m", forward_hf_token=False)
emit("SANDBOX_CREATE", "PASS")

# ---- dataset ----
emit("LOADING_DATASET", DS)
from datasets import load_dataset
ds = load_dataset(DS, split="train")
tierset = {"easy", "medium"}
pool = ds.filter(lambda r: (r.get("difficulty_tier") in tierset))
# VALIDITY_FILTER
_before_vf = len(pool)
pool = pool.filter(lambda r: (task_validity_for(r.get("bucket_prefix", "")) == "VALIDATED"))
_excluded = _before_vf - len(pool)
pool = pool.shuffle(seed=SEED)
emit("DATASET", "total=%d pool=%d excluded=%d" % (len(ds), len(pool), _excluded))

dev_rows = pool.select(range(DEV_TASK_N))
eval_rows = pool.select(range(DEV_TASK_N, DEV_TASK_N + EVAL_TASK_N))

dev_ids = [r["task_id"] for r in dev_rows]
eval_ids = [r["task_id"] for r in eval_rows]
emit("DEV_TASKS", json.dumps(dev_ids))
emit("EVAL_TASKS", json.dumps(eval_ids))

# ---- grader ----
# Inline the exact grader from the worker (byte-identical logic)
def grade(expected, actual, reward_mode="flexible", abs_tol=0.0, rel_tol=0.0):
    """Exact grader: compare expected vs actual with flexible/strict modes."""
    if not actual or not actual.strip():
        return type("G", (), {"reward": 0.0, "match": False})()
    exp_s = str(expected).strip()
    act_s = str(actual).strip()
    if reward_mode == "strict":
        return type("G", (), {"reward": 1.0 if exp_s.lower() == act_s.lower() else 0.0,
                              "match": exp_s.lower() == act_s.lower()})()
    # flexible
    if exp_s.lower() == act_s.lower():
        return type("G", (), {"reward": 1.0, "match": True})()
    # numeric
    try:
        e = float(exp_s.replace(",", "").replace("%", ""))
        a = float(act_s.replace(",", "").replace("%", ""))
        if abs_tol > 0 or rel_tol > 0:
            ok = abs(e - a) <= abs_tol + rel_tol * abs(e)
        else:
            ok = abs(e - a) < 1e-9 or (e != 0 and abs((e - a) / e) < 1e-9)
        return type("G", (), {"reward": 1.0 if ok else 0.0, "match": ok})()
    except Exception:
        pass
    # token overlap (fuzzy text match)
    e_tokens = set(exp_s.lower().split())
    a_tokens = set(act_s.lower().split())
    if e_tokens and e_tokens == a_tokens:
        return type("G", (), {"reward": 1.0, "match": True})()
    return type("G", (), {"reward": 0.0, "match": False})()

# ---- run program in sandbox ----
def run_program(code):
    if not code or not code.strip():
        return ""
    try:
        esc = code.replace("'", "'\\''")
        res = sb.run("printf '%%s' '%s' > /tmp/_prog.py && cd /home/user/input && timeout 30 python3 /tmp/_prog.py 2>/tmp/_stderr.txt" % esc, check=False)
        stdout = (getattr(res, "stdout", "") or "").strip()
        stderr = ""
        try:
            sr = sb.run("cat /tmp/_stderr.txt 2>/dev/null", check=False)
            stderr = (getattr(sr, "stdout", "") or "").strip()
        except Exception:
            pass
        return stdout
    except Exception as e:
        return ""

# ---- prompt hash ----
def prompt_hash(variant, row, contract):
    msgs = build_prompt(row, contract=contract, variant=variant)
    return hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()[:16]

# ---- run all variants on dev set ----
all_results = {}

for variant in VARIANTS:
    emit("VARIANT_START", variant)
    variant_results = []
    torch.manual_seed(SEED)
    
    for ti, r in enumerate(dev_rows):
        _bp = r.get("bucket_prefix", "")
        _task_id = r.get("task_id", "")
        try:
            stage_bucket(sb, _bp)
            _contract = inspect_staged_contract(sb)
            _validity = task_validity_for(_bp)
            
            for nrs_idx in range(NRS):
                torch.manual_seed(SEED)
                msgs = build_prompt(r, contract=_contract, variant=variant)
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                enc = tokenizer(text, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=TEMPERATURE,
                        repetition_penalty=REPETITION_PENALTY,
                        top_p=1.0,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                
                gen = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                n_tokens = out[0].shape[0] - enc["input_ids"].shape[1]
                hit_cap = (n_tokens >= MAX_NEW_TOKENS)
                
                code = extract_code(gen)
                parse = ast_preflight(code)
                form = classify_form_error(code, parse)
                trunc = classify_truncation(code, parse, hit_cap)
                
                stdout = ""
                stderr = ""
                exec_attempted = False
                exec_success = False
                
                if parse["ok"]:
                    exec_attempted = True
                    stdout = run_program(code)
                    exec_success = bool(stdout and stdout.strip())
                    try:
                        sr = sb.run("cat /tmp/_stderr.txt 2>/dev/null", check=False)
                        stderr = (getattr(sr, "stdout", "") or "").strip()
                    except Exception:
                        pass
                
                reward = 0.0
                grader_evaluated = False
                if exec_success:
                    g = grade(r["answer"], stdout, reward_mode=r.get("reward_mode", "flexible"),
                              abs_tol=r.get("atol", 0.0), rel_tol=r.get("rtol", 0.0))
                    reward = float(g.reward)
                    grader_evaluated = True
                
                # Error classification
                err_class = "OK"
                if not parse["ok"]:
                    err_class = "SYNTAX_ERROR"
                elif not exec_success and exec_attempted:
                    se = stderr.lower()
                    if "filenotfounderror" in se or "no such file" in se:
                        err_class = "FILE_NOT_FOUND"
                    elif "column" in se or "not in index" in se or "keyerror" in se:
                        err_class = "COLUMN_ERROR"
                    elif "sqlite" in se or "no such table" in se:
                        err_class = "SQLITE_ERROR"
                    elif "valueerror" in se or "float" in se or "convert" in se:
                        err_class = "VALUE_OR_DATA_ERROR"
                    else:
                        err_class = "OTHER_RUNTIME_ERROR"
                
                variant_results.append({
                    "task_id": _task_id,
                    "nrs_idx": nrs_idx,
                    "validity": _validity,
                    "raw_gen_chars": len(gen),
                    "extracted_code_chars": len(code),
                    "n_tokens": n_tokens,
                    "hit_cap": hit_cap,
                    "ast_parse_ok": parse["ok"],
                    "form_primary": form["primary"],
                    "trunc_class": trunc,
                    "exec_attempted": exec_attempted,
                    "exec_success": exec_success,
                    "stdout": stdout[:500] if stdout else "",
                    "stderr": stderr[:300] if stderr else "",
                    "grader_evaluated": grader_evaluated,
                    "reward": reward,
                    "err_class": err_class,
                })
        except Exception as e:
            for nrs_idx in range(NRS):
                variant_results.append({
                    "task_id": _task_id,
                    "nrs_idx": nrs_idx,
                    "validity": task_validity_for(_bp),
                    "raw_gen_chars": 0,
                    "extracted_code_chars": 0,
                    "n_tokens": 0,
                    "hit_cap": False,
                    "ast_parse_ok": False,
                    "form_primary": "OTHER_RUNTIME_ERROR",
                    "trunc_class": "UNKNOWN_PARSE_FAILURE",
                    "exec_attempted": False,
                    "exec_success": False,
                    "stdout": "",
                    "stderr": type(e).__name__ + ": " + str(e)[:200],
                    "grader_evaluated": False,
                    "reward": 0.0,
                    "err_class": "OTHER_RUNTIME_ERROR",
                })
    
    # Compute variant metrics (§9)
    n = len(variant_results)
    n_ast_pass = sum(1 for v in variant_results if v["ast_parse_ok"])
    n_exec = sum(1 for v in variant_results if v["exec_success"])
    n_stdout = sum(1 for v in variant_results if v["stdout"].strip())
    n_grader = sum(1 for v in variant_results if v["grader_evaluated"])
    rewards = [v["reward"] for v in variant_results]
    n_nonzero = sum(1 for v in variant_results if v["reward"] > 0)
    tokens = [v["n_tokens"] for v in variant_results if v["n_tokens"] > 0]
    cap_hits = sum(1 for v in variant_results if v["hit_cap"])
    
    err_counts = {}
    for v in variant_results:
        ec = v["err_class"]
        if ec != "OK":
            err_counts[ec] = err_counts.get(ec, 0) + 1
    
    all_results[variant] = {
        "n_generations": n,
        "ast_parse_rate": round(n_ast_pass / max(1, n), 4),
        "execution_success_rate": round(n_exec / max(1, n), 4),
        "stdout_nonempty_rate": round(n_stdout / max(1, n), 4),
        "grader_evaluated_rate": round(n_grader / max(1, n), 4),
        "nonzero_reward_count": n_nonzero,
        "nonzero_reward_rate": round(n_nonzero / max(1, n), 4),
        "mean_reward": round(sum(rewards) / max(1, n), 4),
        "mean_tokens": round(sum(tokens) / max(1, len(tokens)), 1) if tokens else 0,
        "cap_hit_fraction": round(cap_hits / max(1, n), 4),
        "error_counts": err_counts,
        "per_task": variant_results,
    }
    emit("VARIANT_DONE", "%s: mean_reward=%.4f exec=%.4f ast=%.4f nonzero=%d/%d" % (
        variant, all_results[variant]["mean_reward"],
        all_results[variant]["execution_success_rate"],
        all_results[variant]["ast_parse_rate"],
        n_nonzero, n))

# ---- write results ----
result = {
    "status": "success",
    "wall_time": round(time.time() - t0, 1),
    "model": MODEL,
    "seed": SEED,
    "temperature": TEMPERATURE,
    "max_new_tokens": MAX_NEW_TOKENS,
    "num_return_sequences": NRS,
    "dev_tasks": dev_ids,
    "eval_tasks": eval_ids,
    "variants": all_results,
    "prompt_hashes": {},
}

# Compute prompt hashes for the first dev task
_contract0 = inspect_staged_contract(sb) if dev_rows else None
stage_bucket(sb, dev_rows[0].get("bucket_prefix", ""))
_contract0 = inspect_staged_contract(sb)
for v in VARIANTS:
    result["prompt_hashes"][v] = prompt_hash(v, dev_rows[0], _contract0)

# Memory
result["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
result["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)

with open(RUN + "/results.json", "w") as fh:
    json.dump(result, fh, indent=2)

emit("PROMPT_DEV_COMPLETE", "wall=%.1fs variants=%d" % (result["wall_time"], len(VARIANTS)))
emit("VERDICT", "PROMPT_DEV_READY")
