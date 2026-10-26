"""P1 V2 Bootstrap: self-contained script for T4.

1. Creates remote directories
2. Generates V2 splits (32 confirmation + 24 train + 32 eval)
3. Runs 128-gen confirmation probe (P0 vs P1)
4. Writes results

Self-contained: all needed functions inlined. No external module imports
beyond torch/transformers/datasets/huggingface_hub (all on T4).

Run via: colab exec -s S -f p1_v2_bootstrap.py --env HF_TOKEN=... --timeout 7200
"""
import json, os, sys, time, hashlib

t0 = time.time()
def emit(tag, val=""):
    print("%s %s" % (tag, val), flush=True)

OUT_ROOT = "/content/opentraining/p1-v2"
SPLIT_DIR = OUT_ROOT + "/splits"
CONFIRM_DIR = OUT_ROOT + "/confirm"
TRAIN_SPLIT_DIR = "/content/opentraining/pilot/p1-v2/splits"
for d in [SPLIT_DIR, CONFIRM_DIR, TRAIN_SPLIT_DIR]:
    os.makedirs(d, exist_ok=True)
emit("DIRS_CREATED", OUT_ROOT)

# ============================================================
# INLINED: prompt contract functions (from grpo_prompt_contract.py)
# ============================================================
PROMPT_SCAFFOLDS = {
    "P0": "",
    "P1": ("\nBefore writing the program, internally determine:\n"
           "1. Which supplied file/database contains the needed information.\n"
           "2. Which exact columns/tables correspond to the question.\n"
           "3. What operation is required: filter / min / max / count / sum / average / ratio / join / SQL query.\n"
           "4. What exact answer format the question requests.\n"
           "Then output only the executable Python program.\n"),
}

def prompt_variant_signature(variant):
    scaffold = PROMPT_SCAFFOLDS.get(variant, "__UNKNOWN_VARIANT__")
    return hashlib.sha256((variant + "\x00" + scaffold).encode("utf-8")).hexdigest()[:16]

def build_prompt(r, contract=None, variant="P0"):
    system = "You are a data analyst. You answer questions about data files by writing a short Python program and reading what it prints."
    files = (contract or {}).get("files") or []
    if not files:
        files = [{"name": f, "kind": "unknown"} for f in r.get("files", [])]
    parts = ["Available files in the input directory (use these exact names):", ""]
    for fr in files:
        parts.append("- " + str(fr.get("name")))
        kind = fr.get("kind")
        if kind == "csv" and fr.get("columns"):
            parts.append("  Columns: " + ", ".join(fr["columns"]))
        elif kind == "csv":
            parts.append("  Schema: unavailable (this file is malformed; open it and inspect it yourself).")
        elif kind == "sqlite" and fr.get("tables"):
            for t, cols in fr["tables"].items():
                parts.append("  Table " + str(t) + ": " + ", ".join(cols))
    contract_text = ("\n".join(parts) + "\n\n") if files else ""
    scaffold = PROMPT_SCAFFOLDS.get(variant, "")
    user = ("%s\n\n%s"
            "The files are in /home/user/input and your program runs in that directory.\n"
            "Use the exact filenames and column/table names shown above.\n"
            "Read files from the provided input directory.\n"
            "Write one Python program in a ```python block, then stop.\n"
            "- Look at the data if you need to, then compute the answer.\n"
            "- The final answer must be the last non-empty stdout line.\n"
            "- Keep it under 40 lines.\n"
            "%s"
            "\nOutput code contract (follow exactly):\n"
            "- Return executable Python source only, inside one ```python block.\n"
            "- Use real newline characters between lines; never write a break as a backslash-n pair.\n"
            "- Do not wrap the whole program in a quoted string or return it as one string literal.\n"
            "- Do not add prose before or after the code, and do not escape the source.\n"
            "- Your code must be directly parseable as Python source." % (r["question"], contract_text, scaffold))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def task_validity_for(bucket_prefix):
    bp = (bucket_prefix or "").strip().lower()
    if "california-crime" in bp or "fbi-us__california" in bp:
        return "GOLD_UNREPRODUCIBLE"
    return "VALIDATED"

# ============================================================
# INLINED: sandbox contract functions (from grpo_smoldataenvs.py)
# ============================================================
def stage_bucket(sb, bucket_prefix):
    bp = (bucket_prefix or "").replace("'", "'\\''")
    return sb.run("rm -rf /home/user/input/* && BUCKET_PREFIX='%s' python3 /opt/pull_bucket.py" % bp, check=False)

def inspect_staged_contract(sb):
    files = []
    try:
        res = sb.run("python3 -c \"import os; files = sorted(os.listdir('/home/user/input')); print('\\n'.join(files))\"", check=False)
        flist = [l.strip() for l in (getattr(res, "stdout", "") or "").split("\n") if l.strip()]
    except Exception:
        flist = []
    for fn in flist:
        entry = {"name": fn, "kind": "unknown"}
        if fn.endswith(".csv"):
            entry["kind"] = "csv"
            try:
                res = sb.run("python3 -c \"import csv; r=csv.reader(open('/home/user/input/%s')); print(','.join(next(r,[])))\" 2>/dev/null" % fn, check=False)
                entry["columns"] = [c.strip() for c in (getattr(res, "stdout", "") or "").split(",") if c.strip()]
            except Exception:
                entry["columns"] = []
        elif fn.endswith(".db") or fn.endswith(".sqlite") or fn.endswith(".sqlite3"):
            entry["kind"] = "sqlite"
            try:
                res = sb.run("python3 -c \"import sqlite3; c=sqlite3.connect('/home/user/input/%s'); tables=[r[0] for r in c.execute(\"SELECT name FROM sqlite_master WHERE type='table' ORDER BY name\")]; c.close(); print('\\n'.join(tables))\" 2>/dev/null" % fn, check=False)
                tables = [l.strip() for l in (getattr(res, "stdout", "") or "").split("\n") if l.strip()]
                entry["tables"] = {}
                for t in tables[:10]:
                    try:
                        res2 = sb.run("python3 -c \"import sqlite3; c=sqlite3.connect('/home/user/input/%s'); cols=[r[1] for r in c.execute('PRAGMA table_info(%s)')]; c.close(); print(','.join(cols))\" 2>/dev/null" % (fn, t), check=False)
                        entry["tables"][t] = [c.strip() for c in (getattr(res2, "stdout", "") or "").split(",") if c.strip()]
                    except Exception:
                        entry["tables"][t] = []
            except Exception:
                entry["tables"] = {}
        files.append(entry)
    return {"files": files}

def extract_code(completion):
    import re
    text = completion[-1]["content"] if isinstance(completion, list) else completion
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    return (blocks[-1] if blocks else text or "").strip()

def ast_preflight(code):
    import ast
    try:
        ast.parse(code or "")
        return {"ok": True, "syntax_error_type": None, "syntax_error_line": None, "syntax_error_message": None}
    except SyntaxError as e:
        return {"ok": False, "syntax_error_type": type(e).__name__, "syntax_error_line": e.lineno, "syntax_error_message": str(e)[:200]}

# ============================================================
# STAGE 1: Generate V2 splits
# ============================================================
emit("SPLIT_GEN_START", "")
from datasets import load_dataset
ds = load_dataset("FineEnvs/SmolDataEnvs", split="train")
tierset = {"easy", "medium"}
pool = ds.filter(lambda r: (r.get("difficulty_tier") in tierset))
pool = pool.filter(lambda r: (task_validity_for(r.get("bucket_prefix", "")) == "VALIDATED"))
pool = pool.shuffle(seed=42)
total = len(pool)
emit("POOL_SIZE", str(total))

prior_ids = set()
for i in range(24):
    r = pool[i]
    prior_ids.add(r.get("task_id", r.get("id", "idx_%d" % i)))
emit("PRIOR_EXCLUDED", str(len(prior_ids)))

remaining = pool.select(range(24, total)).shuffle(seed=43)
n_confirm, n_train, n_eval = 32, 24, 32
assert len(remaining) >= n_confirm + n_train + n_eval

def row_info(r, i):
    return {"task_id": r.get("task_id", r.get("id", "idx_%d" % (24+i))),
            "difficulty": r.get("difficulty_tier", "unknown"),
            "bucket": r.get("bucket_prefix", "unknown"), "validity": "VALIDATED"}

confirm = [row_info(remaining[i], i) for i in range(n_confirm)]
train_v2 = [row_info(remaining[i], i) for i in range(n_confirm, n_confirm + n_train)]
eval_v2 = [row_info(remaining[i], i) for i in range(n_confirm + n_train, n_confirm + n_train + n_eval)]

c_ids = set(x["task_id"] for x in confirm)
t_ids = set(x["task_id"] for x in train_v2)
e_ids = set(x["task_id"] for x in eval_v2)
assert not (c_ids & t_ids) and not (c_ids & e_ids) and not (t_ids & e_ids)
assert not (c_ids & prior_ids) and not (t_ids & prior_ids) and not (e_ids & prior_ids)

with open(SPLIT_DIR + "/p1_confirmation_split.json", "w") as f:
    json.dump({"provenance": "p1_confirmation", "seed": 43, "excluded_prior": sorted(prior_ids),
               "confirmation_tasks": confirm, "task_ids": [x["task_id"] for x in confirm]}, f, indent=2)
with open(SPLIT_DIR + "/p1_lora_search_v2_split.json", "w") as f:
    json.dump({"provenance": "p1_lora_search_v2", "seed": 43, "excluded_prior": sorted(prior_ids),
               "train_task_ids": [x["task_id"] for x in train_v2],
               "eval_task_ids": [x["task_id"] for x in eval_v2],
               "train_details": train_v2, "eval_details": eval_v2, "disjoint": True}, f, indent=2)
# Also write to training path
with open(TRAIN_SPLIT_DIR + "/p1_lora_search_v2_split.json", "w") as f:
    json.dump({"train_task_ids": [x["task_id"] for x in train_v2],
               "eval_task_ids": [x["task_id"] for x in eval_v2]}, f, indent=2)
emit("SPLITS_GENERATED", "confirm=%d train=%d eval=%d" % (n_confirm, n_train, n_eval))

# ============================================================
# STAGE 2: Load model
# ============================================================
MODEL = os.environ.get("MODEL", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
SEED = int(os.environ.get("SEED", "42"))
TEMP = float(os.environ.get("TEMPERATURE", "0.8"))
NRS = int(os.environ.get("NUM_RETURN_SEQUENCES", "2"))
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "256"))
SB_IMAGE = os.environ.get("SB_IMAGE", "docker.io/savatar101/env-data-agent-train:base")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
emit("LOADING_MODEL", MODEL)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")
model.eval()
emit("MODEL_LOADED", "%.3f GB" % (torch.cuda.memory_allocated() / 1e9))

# ============================================================
# STAGE 3: HF sandbox
# ============================================================
_tok = os.environ.get("HF_TOKEN") or ""
if not _tok:
    emit("BLOCKED", "HF_AUTH_INJECTION_FAILED")
    sys.exit(0)
from huggingface_hub import HfApi, Sandbox
w = HfApi().whoami(token=_tok)
_wname = w.get("name", "unknown") if isinstance(w, dict) else getattr(w, "name", "unknown")
emit("WHOAMI", _wname)
sb = Sandbox.create(image=SB_IMAGE, flavor="cpu-basic", idle_timeout="30m", forward_hf_token=False)
emit("SANDBOX_CREATE", "PASS")

# ============================================================
# STAGE 4: Confirmation probe (128 gens)
# ============================================================
task_ids = [x["task_id"] for x in confirm]
tid_to_row = {}
for i in range(len(pool)):
    r = pool[i]
    tid = r.get("task_id", r.get("id", "idx_%d" % i))
    tid_to_row[tid] = r
rows = [tid_to_row[tid] for tid in task_ids]
emit("ROWS_LOADED", str(len(rows)))

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

def run_program(code):
    if not code or not code.strip():
        return ""
    try:
        esc = code.replace("'", "'\\''")
        res = sb.run("printf '%%s' '%s' > /tmp/_prog.py && cd /home/user/input && timeout 30 python3 /tmp/_prog.py 2>/tmp/_stderr.txt" % esc, check=False)
        return (getattr(res, "stdout", "") or "").strip()
    except Exception:
        return ""

def prompt_hash(variant, row, contract):
    msgs = build_prompt(row, contract=contract, variant=variant)
    return hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()[:16]

results = {"task_ids": task_ids, "model": MODEL, "temperature": TEMP,
           "num_return_sequences": NRS, "max_new_tokens": MAX_NEW, "seed": SEED,
           "prompt_hashes": {}, "prompt_sigs": {}, "per_task": []}

_c0 = None
try:
    _bp0 = rows[0].get("bucket_prefix", "")
    if _bp0:
        stage_bucket(sb, _bp0)
        _c0 = inspect_staged_contract(sb)
    for v in ["P0", "P1"]:
        results["prompt_hashes"][v] = prompt_hash(v, rows[0], _c0)
        results["prompt_sigs"][v] = prompt_variant_signature(v)
    emit("PROMPT_PROVENANCE", json.dumps({v: results["prompt_hashes"][v] for v in ["P0", "P1"]}))
except Exception as e:
    emit("PROMPT_HASH_WARN", str(e))

for variant in ["P0", "P1"]:
    emit("VARIANT_START", variant)
    torch.manual_seed(SEED)
    for ti, r in enumerate(rows):
        _bp = r.get("bucket_prefix", "")
        _contract = None
        try:
            if _bp:
                stage_bucket(sb, _bp)
                _contract = inspect_staged_contract(sb)
        except Exception as e:
            emit("STAGE_WARN", "%s: %s" % (r.get("task_id", ""), str(e)[:80]))
        _gold = r.get("answer", "")
        _rm = r.get("reward_mode", "flexible")
        _atol = r.get("atol", 0.0)
        _rtol = r.get("rtol", 0.0)
        entry = {"task_id": r.get("task_id", ""), "difficulty": r.get("difficulty_tier", ""),
                 "bucket": _bp, "validity": task_validity_for(_bp), "prompts": {}}
        try:
            msgs = build_prompt(r, contract=_contract, variant=variant)
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True,
                                     temperature=TEMP, num_return_sequences=NRS,
                                     repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
            il = enc["input_ids"].shape[1]
            comps = []
            for gi in range(gen.shape[0]):
                cid = gen[gi][il:]
                ctext = tok.decode(cid, skip_special_tokens=True)
                code = extract_code(ctext)
                ast_res = ast_preflight(code)
                ce = {"gen": gi, "code_len": len(code), "ast_ok": ast_res["ok"],
                      "ast_err": ast_res.get("syntax_error_type"), "tokens": len(cid)}
                stdout = run_program(code) if (code and ast_res["ok"]) else ""
                ce["exec_ok"] = bool(stdout)
                ce["stdout"] = stdout[-200:]
                lines = [l.strip() for l in stdout.split("\n") if l.strip()]
                final = lines[-1] if lines else ""
                g = grade(_gold, final, _rm, _atol, _rtol)
                ce["reward"] = g.reward
                if not code:
                    ce["error_class"] = "NO_CODE"
                elif not ast_res["ok"]:
                    ce["error_class"] = "SYNTAX_ERROR"
                elif not stdout:
                    ce["error_class"] = "RUNTIME_ERROR"
                elif g.reward == 0.0:
                    ce["error_class"] = "VALUE_OR_DATA_ERROR"
                comps.append(ce)
            entry["prompts"][variant] = {"completions": comps}
        except Exception as e:
            entry["prompts"][variant] = {"error": str(e)[:200]}
        results["per_task"].append(entry)
        if (ti + 1) % 8 == 0:
            emit("PROGRESS", "%s %d/%d (%.0fs)" % (variant, ti + 1, len(rows), time.time() - t0))

def agg(task_list):
    all_c = []
    for t in task_list:
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
    err_counts = {}
    for c in all_c:
        ec = c.get("error_class", "OK" if c.get("reward", 0) > 0 else "OTHER")
        err_counts[ec] = err_counts.get(ec, 0) + 1
    task_correct = sum(1 for t in task_list if any(c.get("reward", 0) > 0 for c in t.get("completions", [])))
    nt = len(task_list)
    return {"n": n, "ast_pass": ast_ok / n, "exec_rate": exec_ok / n,
            "nonzero": nonzero, "nonzero_rate": nonzero / n,
            "mean_reward": sum(rewards) / n, "task_correct": task_correct,
            "task_correct_rate": task_correct / nt if nt else 0,
            "mean_tokens": sum(tokens) / n, "cap_hit": cap_hit / n, "errors": err_counts}

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

with open(CONFIRM_DIR + "/results.json", "w") as f:
    json.dump(results, f, indent=2)
emit("CONFIRMATION_DONE", "wall=%.0fs" % results["wall_seconds"])
emit("P0", "reward=%.4f tasks=%d/%d exec=%.3f ast=%.3f" % (
    results["P0"].get("mean_reward", 0), results["P0"].get("task_correct", 0), 32,
    results["P0"].get("exec_rate", 0), results["P0"].get("ast_pass", 0)))
emit("P1", "reward=%.4f tasks=%d/%d exec=%.3f ast=%.3f" % (
    results["P1"].get("mean_reward", 0), results["P1"].get("task_correct", 0), 32,
    results["P1"].get("exec_rate", 0), results["P1"].get("ast_pass", 0)))
emit("DELTA", "reward=%+.4f tasks=%+d exec=%+.3f" % (
    delta["mean_reward"] or 0, results["P1"].get("task_correct", 0) - results["P0"].get("task_correct", 0),
    delta["exec_rate"] or 0.0))
emit("BOOTSTRAP_COMPLETE", "")
