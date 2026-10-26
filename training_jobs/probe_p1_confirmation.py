"""P1 Generalization Confirmation: P0 vs P1 on 32 NEW untouched tasks.

128 generations: 32 tasks x 2 prompts (P0, P1) x 2 completions each.
Inference-only, no training, no backward, no optimizer.

Mirrors probe_prompt_dev.py execution/sandbox/grader logic exactly.
"""
import json, os, sys, time, hashlib
t0 = time.time()
OUT = os.environ.get("PROBE_OUT", "/content/opentraining/pilot/p1-v2/confirm")
os.makedirs(OUT, exist_ok=True)

def emit(tag, val=""):
    print("%s %s" % (tag, val), flush=True)

def blocked(code, verdict):
    emit("BLOCKED", code)
    emit("VERDICT", verdict)
    sys.exit(0)

# ---- config ----
SPLIT_FILE = os.environ.get("SPLIT_FILE", "/content/opentraining/pilot/p1-v2/splits/p1_confirmation_split.json")
MODEL = os.environ.get("MODEL", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
DS = os.environ.get("DATASET", "FineEnvs/SmolDataEnvs")
SEED = int(os.environ.get("SEED", "42"))
TEMP = float(os.environ.get("TEMPERATURE", "0.8"))
NRS = int(os.environ.get("NUM_RETURN_SEQUENCES", "2"))
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "256"))
SB_IMAGE = os.environ.get("SB_IMAGE", "docker.io/savatar101/env-data-agent-train:base")
VARIANTS = ["P0", "P1"]

# ---- imports ----
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grpo_smoldataenvs import (
    build_prompt, extract_code, ast_preflight, inspect_staged_contract,
    stage_bucket, task_validity_for, classify_form_error, classify_truncation,
)
from grpo_prompt_contract import prompt_variant_signature

# ---- model ----
emit("LOADING_MODEL", MODEL)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
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
sb = Sandbox.create(image=SB_IMAGE, flavor="cpu-basic", idle_timeout="30m", forward_hf_token=False)
emit("SANDBOX_CREATE", "PASS")

# ---- load split ----
with open(SPLIT_FILE) as f:
    split = json.load(f)
task_ids = split["task_ids"]
emit("CONFIRMATION_TASKS", str(len(task_ids)))

# ---- dataset ----
emit("LOADING_DATASET", DS)
from datasets import load_dataset
ds = load_dataset(DS, split="train")
tierset = {"easy", "medium"}
pool = ds.filter(lambda r: (r.get("difficulty_tier") in tierset))
_before_vf = len(pool)
pool = pool.filter(lambda r: (task_validity_for(r.get("bucket_prefix", "")) == "VALIDATED"))
pool = pool.shuffle(seed=SEED)
# Build task_id -> row map
tid_to_row = {}
for i in range(len(pool)):
    r = pool[i]
    tid = r.get("task_id", r.get("id", "idx_%d" % i))
    tid_to_row[tid] = r
rows = []
missing = []
for tid in task_ids:
    if tid in tid_to_row:
        rows.append(tid_to_row[tid])
    else:
        missing.append(tid)
assert not missing, "Missing task_ids in pool: %s" % missing[:5]
emit("ROWS_LOADED", str(len(rows)))

# ---- grader (inline, byte-identical to dev probe) ----
def grade(expected, actual, reward_mode="flexible", abs_tol=0.0, rel_tol=0.0):
    if not actual or not actual.strip():
        return type("G", (), {"reward": 0.0, "match": False})()
    exp_s = str(expected).strip()
    act_s = str(actual).strip()
    if reward_mode == "strict":
        ok = exp_s.lower() == act_s.lower()
        return type("G", (), {"reward": 1.0 if ok else 0.0, "match": ok})()
    if exp_s.lower() == act_s.lower():
        return type("G", (), {"reward": 1.0, "match": True})()
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
        return stdout
    except Exception:
        return ""

# ---- prompt hash ----
def prompt_hash(variant, row, contract):
    msgs = build_prompt(row, contract=contract, variant=variant)
    return hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()[:16]

# ---- prompt provenance ----
results = {"task_ids": task_ids, "model": MODEL, "temperature": TEMP,
           "num_return_sequences": NRS, "max_new_tokens": MAX_NEW, "seed": SEED,
           "prompt_hashes": {}, "prompt_sigs": {}, "per_task": []}
try:
    _bp0 = rows[0].get("bucket_prefix", "")
    if _bp0:
        stage_bucket(sb, _bp0)
    _c0 = inspect_staged_contract(sb) if _bp0 else None
    for v in VARIANTS:
        results["prompt_hashes"][v] = prompt_hash(v, rows[0], _c0)
        results["prompt_sigs"][v] = prompt_variant_signature(v)
    emit("PROMPT_PROVENANCE", json.dumps({v: results["prompt_hashes"][v] for v in VARIANTS}))
except Exception as e:
    emit("PROMPT_HASH_WARN", str(e))

# ---- inference loop ----
for variant in VARIANTS:
    emit("VARIANT_START", variant)
    torch.manual_seed(SEED)
    for ti, r in enumerate(rows):
        _bp = r.get("bucket_prefix", "")
        _task_id = r.get("task_id", "")
        _contract = None
        try:
            if _bp:
                stage_bucket(sb, _bp)
                _contract = inspect_staged_contract(sb)
        except Exception as e:
            emit("STAGE_WARN", "%s: %s" % (_task_id, str(e)[:80]))
        _validity = task_validity_for(_bp)
        _reward_mode = r.get("reward_mode", "flexible")
        _atol = r.get("atol", 0.0)
        _rtol = r.get("rtol", 0.0)
        _gold = r.get("answer", "")
        if _task_id not in [t["task_id"] for t in results["per_task"]]:
            results["per_task"].append({"task_id": _task_id, "difficulty": r.get("difficulty_tier"),
                                         "bucket": _bp, "validity": _validity, "prompts": {}})
        task_entry = [t for t in results["per_task"] if t["task_id"] == _task_id][0]
        try:
            msgs = build_prompt(r, contract=_contract, variant=variant)
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True,
                                     temperature=TEMP, num_return_sequences=NRS,
                                     repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
            inputs_len = enc["input_ids"].shape[1]
            completions = []
            for gi in range(gen.shape[0]):
                cid = gen[gi][inputs_len:]
                ctext = tok.decode(cid, skip_special_tokens=True)
                code = extract_code(ctext)
                ast_res = ast_preflight(code)
                entry = {"gen": gi, "completion": ctext[:500], "code_len": len(code),
                         "ast_ok": ast_res["ok"], "ast_err": ast_res.get("syntax_error_type"),
                         "tokens": len(cid)}
                stdout = ""
                if code and ast_res["ok"]:
                    stdout = run_program(code)
                entry["exec_ok"] = bool(stdout)
                entry["stdout"] = stdout[-200:]
                # reward
                lines = [l.strip() for l in stdout.split("\n") if l.strip()]
                final = lines[-1] if lines else ""
                g = grade(_gold, final, _reward_mode, _atol, _rtol)
                entry["reward"] = g.reward
                # error classification
                if not code:
                    entry["error_class"] = "NO_CODE"
                elif not ast_res["ok"]:
                    entry["error_class"] = "SYNTAX_ERROR"
                elif not stdout:
                    entry["error_class"] = "RUNTIME_ERROR"
                elif g.reward == 0.0:
                    entry["error_class"] = "VALUE_OR_DATA_ERROR"
                if (ti + 1) % 8 == 0:
                    emit("PROGRESS", "%s %d/%d (%.0fs)" % (variant, ti + 1, len(rows), time.time() - t0))
                completions.append(entry)
            task_entry["prompts"][variant] = {"completions": completions}
        except Exception as e:
            task_entry["prompts"][variant] = {"error": str(e)[:200]}

# ---- aggregate ----
def agg(prompt_data):
    all_c = []
    for t in prompt_data:
        for c in t.get("completions", []):
            all_c.append(c)
    n = len(all_c)
    if n == 0:
        return {"n": 0}
    ast_ok = sum(1 for c in all_c if c.get("ast_ok"))
    exec_ok = sum(1 for c in all_c if c.get("exec_ok"))
    nonzero = sum(1 for c in all_c if c.get("reward", 0) > 0)
    rewards = [c.get("reward", 0) for c in all_c]
    tokens = [c.get("tokens", 0) for c in all_c]
    cap_hit = sum(1 for c in tokens if c >= MAX_NEW)
    # error classes
    err_counts = {}
    for c in all_c:
        ec = c.get("error_class", "OK" if c.get("reward", 0) > 0 else "OTHER")
        err_counts[ec] = err_counts.get(ec, 0) + 1
    # task-level
    task_correct = 0
    n_tasks = len(prompt_data)
    for t in prompt_data:
        if any(c.get("reward", 0) > 0 for c in t.get("completions", [])):
            task_correct += 1
    return {"n": n, "ast_pass": ast_ok / n, "exec_rate": exec_ok / n,
            "nonzero": nonzero, "nonzero_rate": nonzero / n,
            "mean_reward": sum(rewards) / n, "task_correct": task_correct,
            "task_correct_rate": task_correct / n_tasks if n_tasks else 0,
            "mean_tokens": sum(tokens) / n, "cap_hit": cap_hit / n,
            "errors": err_counts}

results["P0"] = agg([t["prompts"].get("P0", {"completions": []}) for t in results["per_task"]])
results["P1"] = agg([t["prompts"].get("P1", {"completions": []}) for t in results["per_task"]])
delta = {}
for k in ["mean_reward", "nonzero_rate", "exec_rate", "task_correct_rate", "cap_hit"]:
    p0v = results["P0"].get(k)
    p1v = results["P1"].get(k)
    delta[k] = (p1v - p0v) if (p0v is not None and p1v is not None) else None
results["delta_P1_minus_P0"] = delta
results["wall_seconds"] = round(time.time() - t0, 1)
results["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)

with open(OUT + "/results.json", "w") as f:
    json.dump(results, f, indent=2)
emit("CONFIRMATION_DONE", "wall=%.0fs" % results["wall_seconds"])
emit("P0", "reward=%.4f tasks=%d/%d exec=%.3f ast=%.3f" % (
    results["P0"]["mean_reward"], results["P0"]["task_correct"], len(task_ids),
    results["P0"]["exec_rate"], results["P0"]["ast_pass"]))
emit("P1", "reward=%.4f tasks=%d/%d exec=%.3f ast=%.3f" % (
    results["P1"]["mean_reward"], results["P1"]["task_correct"], len(task_ids),
    results["P1"]["exec_rate"], results["P1"]["ast_pass"]))
emit("DELTA", "reward=%+.4f tasks=%+d exec=%+.3f" % (
    delta["mean_reward"], results["P1"]["task_correct"] - results["P0"]["task_correct"],
    delta["exec_rate"] or 0.0))
