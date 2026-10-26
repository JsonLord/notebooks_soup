#!/usr/bin/env python3
"""Canonical, GIT-TRACKED, REMOTE-ONLY GRPO worker for the OpenTraining pilot.

Single authoritative training source driven through training_jobs/run_remote_grpo.py
(never invoked directly as a production path). This file executes ONLY inside a
colab runtime; it is a faithful parameterization of the ALREADY-QUALIFIED
grpo-smoldataenvs-smoke-001 recipe (T4 GRPO kernel PASS x2). Reward semantics,
model family, precision, generation backend and sandbox behavior are PRESERVED
verbatim; only identity + a bounded set of documented hyperparameters are
env-driven, and a fixed deterministic held-out evaluation slice is added (§7).

Fixed pilot boundaries (§6, enforced, not searched): SmolLM2-360M-Instruct,
fp16=true / bf16=false, use_vllm=false, gradient_checkpointing=true, existing
SmolDataEnvs grader only, HF Sandbox forward_hf_token=False.
"""
import importlib.util
import inspect
import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime, timezone


# ---- byte-identical copy of training_jobs/grpo_eval_split.py (single-file exec)
# A deterministic contract test AST-compares these two definitions to kill drift.
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

# ---- byte-identical copy of training_jobs/grpo_prompt_contract.py
# A deterministic contract test AST-compares these four definitions to kill drift.
def inspect_staged_contract(sb, inp="/home/user/input"):
    """Return a structured contract describing the ACTUAL staged files in `inp`.

    Runs a stdlib-only inspection INSIDE the sandbox `sb` (huggingface_hub
    Sandbox, i.e. sb.run(...) -> object with .stdout) so the result reflects the
    bytes the executed program will actually see. Schema is established with no
    fabrication and never raises: each file is tagged SCHEMA_READ_OK /
    SCHEMA_PARTIAL / SCHEMA_READ_FAILED (csv) or SQLITE_READ_OK /
    SQLITE_READ_FAILED (sqlite). One malformed file never destroys the whole
    contract. Row VALUES are never read; only names/columns/tables. Shape:
      {"dir": ..., "files": [{"name", "kind", ...}, ...]}
    """
    snippet = (
        "import os,sys,json,csv,sqlite3\n"
        "d=sys.argv[1]\n"
        "files=sorted(os.listdir(d)) if os.path.isdir(d) else []\n"
        "out=[]\n"
        "for f in files:\n"
        "    p=os.path.join(d,f)\n"
        "    if not os.path.isfile(p): continue\n"
        "    low=f.lower(); rec={'name':f,'kind':'other'}\n"
        "    if low.endswith('.csv'):\n"
        "        rec['kind']='csv'\n"
        "        try:\n"
        "            with open(p,'rb') as fh: raw=fh.read()\n"
        "            txt=raw.decode('utf-8','replace')\n"
        "            lines=txt.replace('\\r\\n','\\n').replace('\\r','\\n').split('\\n')\n"
        "            hdr=next(csv.reader(lines))\n"
        "            cols=[c.strip() for c in hdr if c.strip()!='']\n"
        "            first=lines[0] if lines else ''\n"
        "            if len(cols)==0:\n"
        "                rec['schema_read']='SCHEMA_READ_FAILED'; rec['columns']=None\n"
        "            elif len(cols)==1 and len(first)>60 and ',' not in first:\n"
        "                rec['schema_read']='SCHEMA_PARTIAL'; rec['columns']=cols\n"
        "            else:\n"
        "                rec['schema_read']='SCHEMA_READ_OK'; rec['columns']=cols\n"
        "        except Exception as e:\n"
        "            rec['schema_read']='SCHEMA_READ_FAILED'; rec['columns']=None; rec['err']=type(e).__name__\n"
        "    elif low.endswith('.sqlite') or low.endswith('.db'):\n"
        "        rec['kind']='sqlite'\n"
        "        try:\n"
        "            con=sqlite3.connect('file:%s?mode=ro'%p,uri=True); cur=con.cursor()\n"
        "            tabs=[r[0] for r in cur.execute(\"select name from sqlite_master where type='table' and name not like 'sqlite_%'\").fetchall()]\n"
        "            sch={}\n"
        "            for t in tabs:\n"
        "                sch[t]=[r[1] for r in cur.execute('PRAGMA table_info(\"%s\")'%t).fetchall()]\n"
        "            rec['tables']=sch; rec['sqlite_read']='SQLITE_READ_OK'; con.close()\n"
        "        except Exception as e:\n"
        "            rec['tables']={}; rec['sqlite_read']='SQLITE_READ_FAILED'; rec['err']=type(e).__name__\n"
        "    out.append(rec)\n"
        "print(json.dumps({'dir':d,'files':out}))\n"
    )
    esc = snippet.replace("'", "'\\''")
    res = sb.run("printf '%%s' '%s' > /tmp/_ic.py && python3 /tmp/_ic.py '%s'" % (esc, inp), check=False)
    raw = (getattr(res, "stdout", "") or "").strip()
    try:
        return json.loads(raw) if raw else {"dir": inp, "files": []}
    except Exception:
        return {"dir": inp, "files": []}


def stage_bucket(sb, bucket_prefix):
    """Wipe the input dir and stage exactly this bucket's files, deterministically.

    Mirrors the worker's proven staging shell. The staged filesystem is then
    authoritative; call inspect_staged_contract immediately after to derive the
    prompt contract, and the task program runs in this same directory.
    """
    bp = (bucket_prefix or "").replace("'", "'\\''")
    return sb.run("rm -rf /home/user/input/* && BUCKET_PREFIX='%s' python3 /opt/pull_bucket.py" % bp, check=False)


def build_prompt(r, contract=None):
    """Build the chat messages for row `r`, disclosing the staged contract.

    When `contract` (from inspect_staged_contract) is provided, every staged
    filename plus its CSV columns / SQLite tables+columns is surfaced, and the
    prompt carries a strict output contract (use exact names; the final answer is
    the last non-empty stdout line). When `contract` is None it degrades to the
    legacy nominal-files prompt, so the trainer's dataset `.map` site -- where no
    bucket is staged yet -- keeps producing the SAME prompt as before with zero
    behavior change. No gold / solution / grader internals are ever emitted.
    """
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
    user = ("%s\n\n%s"
            "The files are in /home/user/input and your program runs in that directory.\n"
            "Use the exact filenames and column/table names shown above.\n"
            "Read files from the provided input directory.\n"
            "Write one Python program in a ```python block, then stop.\n"
            "- Look at the data if you need to, then compute the answer.\n"
            "- The final answer must be the last non-empty stdout line.\n"
            "- Keep it under 40 lines.\n"
            "\n"
            "Output code contract (follow exactly):\n"
            "- Return executable Python source only, inside one ```python block.\n"
            "- Use real newline characters between lines; never write a break as a backslash-n pair.\n"
            "- Do not wrap the whole program in a quoted string or return it as one string literal.\n"
            "- Do not add prose before or after the code, and do not escape the source.\n"
            "- Your code must be directly parseable as Python source." % (r["question"], contract_text))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def task_validity_for(bucket_prefix):
    """Return the experiment eligibility label for a bucket prefix.

    GOLD_UNREPRODUCIBLE is applied only to a task whose gold answer was proven,
    by the known-correct-data discriminator, to be irreproducible from the
    staged data (here: fbi-us__california-crime, whose gold 'Imperial3,
    0.003403' does not equal the literal Imperial3 ratio <latex>5/17338</latex>
    nor the data's true minimum). This is an annotation, not a dataset edit, not
    a deletion, and not grader softening. Everything else is VALIDATED.
    """
    bp = (bucket_prefix or "").strip().lower()
    if "california-crime" in bp or "fbi-us__california" in bp:
        return "GOLD_UNREPRODUCIBLE"
    return "VALIDATED"



# ---- environment / JobSpec parsing (§5: identity has no silent defaults) ----
def now():
    return datetime.now(timezone.utc).isoformat()


def emit(k, v):
    print("%s=%s" % (k, v), flush=True)


def _env(name, default=None):
    v = os.environ.get(name)
    return v if (v is not None and str(v).strip() != "") else default


def _req(name):
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        emit("MISSING_IDENTITY", name)
        emit("GRPO_VERDICT", "MISSING_IDENTITY")
        print("EXECUTION_TARGET=COLAB_REMOTE", flush=True)
        sys.exit(0)  # exit(0) so the artifact producer still bundles evidence
    return v


def _bool(name, default):
    v = _env(name)
    return default if v is None else str(v).strip().lower() in ("1", "true", "yes", "on")


def _num(name, default, cast=float):
    v = _env(name)
    try:
        return default if v is None else cast(v)
    except Exception:
        return default


JOB_ID = _req("JOB_ID")
ATTEMPT_ID = _req("ATTEMPT_ID")
RUN_INSTANCE = _req("RUN_INSTANCE")
MODEL = _env("MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")
DS = _env("DATASET", "FineEnvs/SmolDataEnvs")
SB_IMAGE = _env("SB_IMAGE", "docker.io/savatar101/env-data-agent-train:base")
RUN = _env("REMOTE_RUN_DIR", "/content/opentraining/pilot/%s/%s" % (JOB_ID, ATTEMPT_ID))
ART = RUN + "/artifacts"

difficulty_tiers = [t.strip() for t in _env("DIFFICULTY_TIERS", "easy,medium").split(",") if t.strip()]
SEED = _num("SEED", 42, int)
LEARNING_RATE = _num("LEARNING_RATE", 3e-6, float)
TEMPERATURE = _num("TEMPERATURE", 0.8, float)
REPETITION_PENALTY = _num("REPETITION_PENALTY", 1.05, float)
NUM_GENERATIONS = _num("NUM_GENERATIONS", 2, int)
def resolve_max_new_tokens(env=None, default=128):
    """Resolve the effective generation token budget deterministically.

    Precedence: MAX_NEW_TOKENS, then MAX_COMPLETION_LENGTH (legacy name), then
    `default` (production 128). A missing/blank value falls through; a
    non-integer or non-positive value returns `default`, so a malformed
    experiment setting never silently changes the production default.
    """
    import os as _os
    src = _os.environ if env is None else env
    for key in ("MAX_NEW_TOKENS", "MAX_COMPLETION_LENGTH"):
        raw = src.get(key)
        if raw is None:
            continue
        s = str(raw).strip()
        if not s:
            continue
        try:
            v = int(s)
        except Exception:
            return default
        if v <= 0:
            return default
        return v
    return default


MAX_COMPLETION_LENGTH = resolve_max_new_tokens(os.environ, 128)
MAX_NEW_TOKENS = MAX_COMPLETION_LENGTH
MAX_STEPS = _num("MAX_STEPS", 20, int)
TRAIN_TASK_LIMIT = _num("TRAIN_TASK_LIMIT", 1, int)
EVAL_TASK_LIMIT = _num("EVAL_TASK_LIMIT", 6, int)
FP16 = _bool("FP16", True)
BF16 = _bool("BF16", False)
USE_VLLM = _bool("USE_VLLM", False)
# ---- §3 search: restrict the training/eval population to VALIDATED tasks ----
VALIDITY_FILTER = _bool("VALIDITY_FILTER", False)
# ---- §17/§18 training-method parameterization (full path UNCHANGED when method != lora) ----
TRAINING_METHOD = (_env("TRAINING_METHOD", "full") or "full").strip().lower()
LORA_R = _num("LORA_R", 16, int)
LORA_ALPHA = _num("LORA_ALPHA", 32, int)
LORA_DROPOUT = _num("LORA_DROPOUT", 0.05, float)
LORA_TARGET_MODE = _env("LORA_TARGET_MODE", "all-linear")

print("EXECUTION_TARGET=COLAB_REMOTE", flush=True)
os.makedirs(ART + "/checkpoint", exist_ok=True)
os.makedirs("/content/opentraining", exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(RUN + "/log.txt"), logging.StreamHandler(sys.stdout)])
log = logging.getLogger("grpo_pilot")
log.info("grpo pilot worker start %s job=%s attempt=%s", now(), JOB_ID, ATTEMPT_ID)

HB = RUN + "/heartbeat.json"
PID = os.getpid()
def hb(state, step=0):
    try:
        with open(HB, "w") as fh:
            json.dump({"job_id": JOB_ID, "attempt_id": ATTEMPT_ID, "run_instance": RUN_INSTANCE,
                       "state": state, "last_step": int(step), "updated_at": now(), "pid": PID}, fh)
    except Exception:
        pass
hb("PREPARING", 0)

# ---- §2/§6 safety guard (not a recipe change): never silently 2B/Qwen on T4 ----
_mlow = MODEL.lower()
if (not USE_VLLM) and ("2b" in _mlow or "qwen" in _mlow):
    emit("GRPO_VERDICT", "MODEL_NOT_SUPPORTED_ON_T4")
    hb("FAILED", 0)
    sys.exit(0)

import platform
import torch
from importlib import metadata
def ver(p):
    try:
        return metadata.version(p)
    except Exception:
        return "not-installed"
env = {"captured_at": now(), "hostname": platform.node(),
       "python": sys.version.split()[0],
       "packages": {p: ver(p) for p in ("torch", "transformers", "trl", "peft",
                    "datasets", "accelerate", "numpy", "tokenizers", "huggingface_hub", "torchao")},
       "torch_cuda": torch.version.cuda or "none",
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
       "sm": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
       "bf16_supported": bool(torch.cuda.is_bf16_supported()) if torch.cuda.is_available() else False}
with open(RUN + "/environment.json", "w") as fh:
    json.dump(env, fh, indent=2)
props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
# T4 (cc 7.5) => bf16 unsupported; force fp16 routing per proven SFT/GRPO path.
if props is not None and torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    if cc[0] < 8:
        BF16 = False
        FP16 = True
hardware = {"gpu": props.name if props else "none",
            "compute_capability": list(torch.cuda.get_device_capability(0)) if props else None,
            "total_gpu_mem_mb": round(props.total_memory / 1e6) if props else None,
            "requested_precision": "float16" if FP16 else ("bfloat16" if BF16 else "none"),
            "effective_precision": "fp16=%s,bf16=%s" % (FP16, BF16),
            "use_vllm": USE_VLLM, "generation_backend": "transformers/model.generate",
            "note": "cc[7,5] => bf16 unsupported, fp16 per proven SFT/GRPO routing"}
with open(RUN + "/hardware.json", "w") as fh:
    json.dump(hardware, fh, indent=2)

# ---- intended GRPOConfig kwargs (NO max_prompt_length) ----
CFG = {"output_dir": ART + "/checkpoint", "per_device_train_batch_size": 2,
       "gradient_accumulation_steps": 1, "num_generations": NUM_GENERATIONS, "max_steps": MAX_STEPS,
       "max_completion_length": MAX_COMPLETION_LENGTH, "learning_rate": LEARNING_RATE,
       "temperature": TEMPERATURE, "repetition_penalty": REPETITION_PENALTY,
       "gradient_checkpointing": True, "bf16": BF16, "fp16": FP16, "use_vllm": USE_VLLM,
       "mask_truncated_completions": True, "logging_steps": 1, "report_to": [], "seed": SEED}
REQUESTED_FIELDS = ["fp16", "bf16", "use_vllm", "num_generations", "max_steps",
                    "max_completion_length", "gradient_checkpointing", "logging_steps",
                    "learning_rate", "temperature", "repetition_penalty"]

STATE = {"classification": None, "terminal": None, "gates": {}, "sandbox": {},
         "rewards": [], "gen_count": 0, "completion_lengths": [], "text_sizes_chars": [],
         "boundaries": {}, "observation": {}}
def bnd(name, value, direct):
    STATE["boundaries"][name] = bool(value)
    STATE["observation"][name] = "direct" if direct else "derived"

sb = None
row = None
grader = None
_rewarding_flag = {"done": False}
_lora_group_rewards = []  # §11/§12: per-GRPO-group reward vectors

# ---- notebook reward helpers (UNCHANGED semantics), defined before preflight ----
SYSTEM = "You are a data analyst. You answer questions about CSV files by writing a short Python program and reading what it prints."


def extract_code(completion):
    text = completion[-1]["content"] if isinstance(completion, list) else completion
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    return (blocks[-1] if blocks else text or "").strip()
def run_program(code):
    payload = code.replace("'", "'\\''")
    r = sb.run("cd /home/user/input && printf '%%s' '%s' > /tmp/s.py && timeout 90 python3 /tmp/s.py" % payload, check=False)
    lines = [ln.strip() for ln in (getattr(r, "stdout", "") or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""
def reward_correct(completions, **cols):
    bnd("reward_function_entered", True, True)
    if not _rewarding_flag["done"]:
        _rewarding_flag["done"] = True
    n = len(completions)
    bnd("generation_1_produced", n >= 1, True)
    bnd("generation_2_produced", n >= 2, True)
    out = []
    for i, c in enumerate(completions):
        STATE["gen_count"] += 1
        text = c[-1]["content"] if isinstance(c, list) else (c or "")
        STATE["text_sizes_chars"].append(len(text or ""))
        STATE["completion_lengths"].append(len(text or ""))
        code = extract_code(c)
        if code:
            try:
                _bp = cols.get("bucket_prefix")[i]
            except Exception:
                _bp = None
            if _bp:
                stage_bucket(sb, _bp)
        printed = run_program(code) if code else ""
        try:
            g = grader.grade(cols["answer"][i], printed, reward_mode=cols["reward_mode"][i],
                             abs_tol=cols["atol"][i], rel_tol=cols["rtol"][i]) if printed else None
            val = float(g.reward) if g else 0.0
        except Exception:
            val = 0.0
        out.append(val)
        STATE["rewards"].append(val)
        bnd("reward_%d_computed" % (i + 1), i + 1 == len(out), True)
    # §11/§12 per-GRPO-group reward variance (pure recording, no semantics change)
    if TRAINING_METHOD == "lora":
        try:
            import statistics as _stg
            _g = list(out)
            _bpg = None
            try:
                _bpg = cols.get("bucket_prefix")[0]
            except Exception:
                _bpg = None
            _lora_group_rewards.append({
                "group_index": len(_lora_group_rewards),
                "bucket_prefix": _bpg,
                "rewards": _g,
                "all_identical": (len(set(_g)) <= 1),
                "reward_variance": round(_stg.pvariance(_g), 8) if len(_g) > 1 else 0.0,
                "informative": (len(set(_g)) > 1),
                "nonzero_count": sum(1 for v in _g if v > 0)})
        except Exception:
            pass
    return out

def persist_final(status, err=None):
    STATE["status"] = status
    STATE["error"] = err
    STATE["max_new_tokens"] = MAX_NEW_TOKENS
    try:
        with open(RUN + "/results.json", "w") as fh:
            json.dump(STATE, fh, indent=2)
        with open(RUN + "/sandbox.json", "w") as fh:
            json.dump(STATE["sandbox"], fh, indent=2)
    except Exception:
        pass

def blocked(cls, verdict):
    STATE["classification"] = cls
    emit("GRPO_VERDICT", verdict)
    persist_final("blocked", cls)
    hb("FAILED", 0)
    try:
        if sb is not None:
            sb.kill()
    except Exception:
        pass
    sys.exit(0)

manifest = {"job_id": JOB_ID, "attempt_id": ATTEMPT_ID, "run_instance": RUN_INSTANCE,
            "provider": "colab", "model": MODEL, "dataset": DS, "difficulty_tiers": difficulty_tiers,
            "train_task_limit": TRAIN_TASK_LIMIT, "eval_task_limit": EVAL_TASK_LIMIT,
            "num_generations": NUM_GENERATIONS, "max_steps": MAX_STEPS,
            "max_completion_length": MAX_COMPLETION_LENGTH, "max_new_tokens": MAX_NEW_TOKENS, "seed": SEED,
            "effective_hyperparameters": {"learning_rate": LEARNING_RATE, "temperature": TEMPERATURE,
                                          "repetition_penalty": REPETITION_PENALTY,
                                          "gradient_checkpointing": True},
            "precision": "fp16" if FP16 else ("bf16" if BF16 else "none"), "fp16": FP16, "bf16": BF16,
            "use_vllm": USE_VLLM, "forward_hf_token": False, "remote_run_dir": RUN,
            "git_sha": _env("GIT_SHA", "unknown"), "worker_sha": _env("WORKER_SHA", "unknown"),
            "source_of_truth": "training_jobs/grpo_smoke.py (kernel proven) + §7 eval contract",
            "attempted_at": now()}
with open(RUN + "/manifest.json", "w") as fh:
    json.dump(manifest, fh, indent=2)

# ---- HF auth ----
_tok = os.environ.get("HF_TOKEN") or ""
auth = {"hf_token_present": bool(_tok), "whoami_ok": None, "jobs_permission_ok": None}
with open(RUN + "/auth.json", "w") as fh:
    json.dump(auth, fh, indent=2)
try:
    from huggingface_hub import HfApi
    w = HfApi().whoami(token=_tok or None)
    auth["whoami_ok"] = isinstance(w, dict)
except Exception as e:
    auth["whoami_ok"] = False
    STATE["whoami_err"] = type(e).__name__
try:
    from huggingface_hub import Sandbox
    auth["jobs_permission_ok"] = hasattr(Sandbox, "create")
except Exception as e:
    auth["jobs_permission_ok"] = False
    STATE["sandbox_client_err"] = type(e).__name__
with open(RUN + "/auth.json", "w") as fh:
    json.dump(auth, fh, indent=2)
emit("AUTH_WHOAMI_OK", auth["whoami_ok"])
if not (auth["hf_token_present"] and auth["whoami_ok"]):
    blocked("HF_AUTH_INJECTION_FAILED", "HF_AUTH_INJECTION_FAILED")
hb("AUTH_OK", 0)

# ---- §5a GRPOConfig/GRPOTrainer API preflight ----
import trl
from trl import GRPOConfig, GRPOTrainer
sig = set(inspect.signature(GRPOConfig.__init__).parameters)
try:
    fields = set(GRPOConfig.__dataclass_fields__.keys())
except Exception:
    fields = set()
cfg_fields = sig | fields
accepted = [f for f in REQUESTED_FIELDS if f in cfg_fields]
unsupported = [f for f in REQUESTED_FIELDS if f not in cfg_fields]
construct_ok, construct_err = False, None
try:
    GRPOConfig(**{k: v for k, v in CFG.items() if k != "output_dir"},
               output_dir="/content/opentraining/_pf_tmp")
    construct_ok = True
except Exception as e:
    construct_err = type(e).__name__ + ":" + str(e)[:200]
tp = set(inspect.signature(GRPOTrainer.__init__).parameters)
trainer_sig_ok = all(k in tp for k in ("model", "reward_funcs", "args", "train_dataset"))
api_pf = {"trl_version": trl.__version__, "requested_fields": REQUESTED_FIELDS,
          "accepted_fields": accepted, "unsupported_fields": unsupported,
          "config_construct_ok": construct_ok, "config_construct_err": construct_err,
          "trainer_signature_ok": trainer_sig_ok,
          "trainer_sig_missing": [k for k in ("model", "reward_funcs", "args", "train_dataset") if k not in tp]}
with open(RUN + "/grpo_api_preflight.json", "w") as fh:
    json.dump(api_pf, fh, indent=2)
STATE["gates"]["api_preflight"] = "PASS" if (not unsupported and construct_ok and trainer_sig_ok) else "FAIL"
emit("API_UNEXPECTED_UNSUPPORTED", unsupported)

# ---- §5b callable preflight (reward binding; NO callbacks) ----
rew_sig = inspect.signature(reward_correct)
rew_params = list(rew_sig.parameters.values())
has_positional = any(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in rew_params)
has_var_kw = any(p.kind == p.VAR_KEYWORD for p in rew_params)
reward_callable_ok = bool(callable(reward_correct) and has_positional and has_var_kw)
custom_callbacks = []
callable_pf = {"grpo_config_ok": bool(not unsupported and construct_ok),
               "grpo_trainer_ok": bool(trainer_sig_ok), "reward_callable_ok": reward_callable_ok,
               "reward_signature": str(rew_sig), "custom_callbacks": custom_callbacks,
               "unsupported_fields": unsupported}
callable_pf["verdict"] = "PASS" if (callable_pf["grpo_config_ok"] and callable_pf["grpo_trainer_ok"]
                                   and callable_pf["reward_callable_ok"] and not custom_callbacks
                                   and not unsupported) else "FAIL"
STATE["gates"]["callable_preflight"] = callable_pf["verdict"]
emit("REWARD_SIG", str(rew_sig))
emit("CALLABLE_VERDICT", callable_pf["verdict"])
if callable_pf["verdict"] != "PASS":
    blocked("GRPO_CALLABLE_PREFLIGHT_FAIL", "GRPO_CALLABLE_PREFLIGHT_FAIL")
hb("API_PREFLIGHT_OK", 0)

# ---- dataset load + deterministic disjoint train/eval slice (§7) ----
try:
    from datasets import load_dataset
    ds = load_dataset(DS, split="train")
    tierset = set(difficulty_tiers)
    pool = ds.filter(lambda r: (r.get("difficulty_tier") in tierset))
    _excluded_invalid = 0
    if VALIDITY_FILTER:
        _before_vf = len(pool)
        pool = pool.filter(lambda r: (task_validity_for(r.get("bucket_prefix", "")) == "VALIDATED"))
        _excluded_invalid = _before_vf - len(pool)
    pool = pool.shuffle(seed=SEED)
    total = len(pool)
    STATE["validity_filter"] = {"enabled": VALIDITY_FILTER, "excluded": _excluded_invalid}
    emit("DATASET_GATE", "ok rows=%d pool=%d excluded_invalid=%d" % (len(ds), total, _excluded_invalid))
except Exception as e:
    STATE["gates"]["dataset_load"] = "FAIL"
    emit("DATASET_GATE", "FAIL:" + type(e).__name__ + ":" + str(e)[:120])
    blocked("DATASET", "COLAB_GRPO_CONTROL_PLANE_FAIL")
STATE["gates"]["dataset_load"] = "PASS"
try:
    train_idx, eval_idx = disjoint_split(total, TRAIN_TASK_LIMIT, EVAL_TASK_LIMIT)
    eval_rows = [pool[i] for i in eval_idx]
    manifest["train_subset_indices"] = train_idx
    manifest["eval_subset_indices"] = eval_idx
    with open(RUN + "/manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
except Exception as e:
    STATE["gates"]["eval_split"] = "FAIL"
    emit("EVAL_SPLIT_GATE", "FAIL:" + type(e).__name__ + ":" + str(e)[:120])
    blocked("EVAL_SPLIT", "COLAB_GRPO_CONTROL_PLANE_FAIL")
STATE["gates"]["eval_split"] = "PASS"
emit("EVAL_SPLIT_GATE", "ok train=%d eval=%d" % (len(train_idx), len(eval_idx)))

# ---- grader load ----
import sys as _s
try:
    from huggingface_hub import hf_hub_download
    gpath = hf_hub_download(DS, "grader.py", repo_type="dataset")
    spec = importlib.util.spec_from_file_location("grader", gpath)
    grader = importlib.util.module_from_spec(spec)
    _s.modules["grader"] = grader
    spec.loader.exec_module(grader)
    STATE["gates"]["grader_load"] = "PASS"
    emit("GRADER_GATE", "ok")
except Exception as e:
    STATE["gates"]["grader_load"] = "FAIL"
    emit("GRADER_GATE", "FAIL:" + type(e).__name__ + ":" + str(e)[:120])
    blocked("GRADER", "COLAB_GRPO_CONTROL_PLANE_FAIL")

# ---- §6 sandbox lifecycle + staging (forward_hf_token=False) ----
STATE["sandbox"]["forward_hf_token"] = False
def make_sandbox():
    return Sandbox.create(image=SB_IMAGE, flavor="cpu-basic", idle_timeout="30m", forward_hf_token=False)
try:
    sb = make_sandbox()
    STATE["gates"]["sandbox_create"] = "PASS"
    STATE["sandbox"]["create"] = "PASS"
    emit("HF_SANDBOX_CREATE_PASS", "1")
except Exception as e:
    STATE["gates"]["sandbox_create"] = "FAIL"
    STATE["sandbox"]["create"] = "FAIL"
    STATE["sandbox"]["create_error"] = (type(e).__name__ + ": " + str(e))[:300]
    blocked("HF_SANDBOX_UNAVAILABLE", "HF_SANDBOX_UNAVAILABLE")

# stage the FIRST eval task's bucket for reward round-trip sanity
row = eval_rows[0] if eval_rows else None
try:
    bp = row["bucket_prefix"]
    stage = sb.run("rm -rf /home/user/input/* && BUCKET_PREFIX=%s python3 /opt/pull_bucket.py" % bp, check=False)
    ls = sb.run("ls -la /home/user/input", check=False)
    lsout = (getattr(ls, "stdout", "") or "")
    files = [l.split()[-1] for l in lsout.splitlines() if l.strip() and not l.strip().startswith("total") and not l.strip().endswith(".")]
    files = [f for f in files if f not in (".", "..")]
    staged = len(files) > 0
    STATE["sandbox"]["files"] = files[:20]
    if staged:
        STATE["gates"]["task_staging"] = "PASS"
        STATE["sandbox"]["staging"] = "PASS"
        emit("SANDBOX_DATA_STAGE_PASS", "1 files=%d" % len(files))
    else:
        STATE["gates"]["task_staging"] = "FAIL"
        STATE["sandbox"]["staging"] = "FAIL"
        blocked("SANDBOX_DATA_AUTH_REQUIRED", "SANDBOX_DATA_AUTH_REQUIRED")
except Exception as e:
    STATE["gates"]["task_staging"] = "FAIL"
    STATE["sandbox"]["stage_error"] = (type(e).__name__ + ": " + str(e))[:300]
    blocked("HF_SANDBOX_RUNTIME", "COLAB_GRPO_CONTROL_PLANE_FAIL")
hb("SANDBOX_OK", 0)

# ---- §6 reward round-trip (precheck: gold->1.0, wrong->0.0) ----
try:
    gold = row["answer"]
    g_ok = grader.grade(gold, gold, reward_mode=row["reward_mode"], abs_tol=row["atol"], rel_tol=row["rtol"])
    r_gold = float(g_ok.reward)
    g_wrong = grader.grade(gold, "__definitely_wrong_9x7q__", reward_mode=row["reward_mode"], abs_tol=row["atol"], rel_tol=row["rtol"])
    r_wrong = float(g_wrong.reward)
    STATE["rewards_gold"] = r_gold
    STATE["rewards_wrong"] = r_wrong
    STATE["gates"]["reward_roundtrip"] = "PASS" if (r_gold >= 0.999 and r_wrong < 0.001) else "FAIL"
    emit("REWARD_GOLD", r_gold)
    emit("REWARD_WRONG", r_wrong)
    emit("REWARD_PASS", "1" if STATE["gates"]["reward_roundtrip"] == "PASS" else "0")
except Exception as e:
    STATE["gates"]["reward_roundtrip"] = "FAIL"
    emit("REWARD_PASS", "0 err:" + type(e).__name__)
    blocked("REWARD", "COLAB_GRPO_CONTROL_PLANE_FAIL")
if STATE["gates"]["reward_roundtrip"] != "PASS":
    blocked("REWARD", "COLAB_GRPO_CONTROL_PLANE_FAIL")
hb("REWARD_OK", 0)

# ---- deterministic held-out evaluation (§7): needs model + tokenizer + sandbox ----
def gen_eval_pass(model, tokenizer):
    rewards = []
    valid = 0
    execed = 0
    comp_len = []
    tok = tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        torch.manual_seed(SEED)
    except Exception:
        pass
    was_training = model.training
    try:
        model.eval()
        for r in eval_rows:
            try:
                _bp = r["bucket_prefix"]
                if _bp:
                    stage_bucket(sb, _bp)
                _contract = inspect_staged_contract(sb)
                STATE.setdefault("eval_task_validity", []).append(task_validity_for(_bp))
                msgs = build_prompt(r, contract=_contract)
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                enc = tok(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                                         temperature=TEMPERATURE, repetition_penalty=REPETITION_PENALTY,
                                         top_p=1.0, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
                gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                comp_len.append(len(gen))
                code = extract_code(gen)
                if code:
                    valid += 1
                printed = run_program(code) if code else ""
                if printed:
                    execed += 1
                g = grader.grade(r["answer"], printed, reward_mode=r["reward_mode"], abs_tol=r["atol"], rel_tol=r["rtol"]) if printed else None
                rewards.append(float(g.reward) if g else 0.0)
            except Exception as e:
                STATE.setdefault("eval_err", []).append(type(e).__name__)
                rewards.append(0.0)
    except Exception as e:
        STATE["eval_fatal"] = type(e).__name__ + ": " + str(e)[:200]
        return None
    finally:
        try:
            if was_training:
                model.train()
        except Exception:
            pass
    sc = score_from_rewards(rewards)
    sc["valid_program_rate"] = (valid / len(eval_rows)) if eval_rows else None
    sc["sandbox_execution_rate"] = (execed / len(eval_rows)) if eval_rows else None
    sc["mean_completion_length"] = (sum(comp_len) / len(comp_len)) if comp_len else None
    sc["rewards"] = rewards
    return sc

def build_lora_trainer(train, cfg):
    """Bounded LoRA-GRPO build for a 1.7B on T4 (full path UNCHANGED). Returns (trainer, lora_info).
    Live pinned-stack preflight; MEASURED (not assumed) reference-model behavior and M0-M4 memory
    stages; base-weights frozen proof. Adapter is persisted by the caller (§14). No trainer callbacks
    (forbidden); the memory canary is the strict pre-admission gate + a post-hoc peak-reserve check."""
    emit("TRAINING_METHOD", "lora")
    emit("LEARNING_RATE", LEARNING_RATE)

    def _gpu_free_bytes():
        try:
            return int(torch.cuda.mem_get_info(0)[0])
        except Exception:
            return None

    def _mem_snapshot(tag):
        d = {"tag": tag}
        try:
            d["allocated_bytes"] = int(torch.cuda.memory_allocated())
            d["reserved_bytes"] = int(torch.cuda.max_memory_reserved())
            d["device_free_bytes"] = _gpu_free_bytes()
        except Exception as e:
            d["error"] = type(e).__name__
        try:
            import resource as _res
            d["host_rss_mb"] = round(_res.getrusage(_res.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
        except Exception:
            d["host_rss_mb"] = None
        return d

    # ---- §4/§21 live pinned-stack LoRA+GRPO API preflight (runtime authoritative) ----
    pre = {"training_method": "lora"}
    try:
        import peft as _peft
        from peft import LoraConfig as _LoraConfig, get_peft_model as _get_peft_model
        from peft import TaskType as _TaskType
        pre["peft_version"] = getattr(_peft, "__version__", None)
    except Exception as e:
        STATE["gates"]["lora_api_preflight"] = "FAIL"
        emit("LORA_API", "FAIL: peft import " + type(e).__name__ + ":" + str(e)[:120])
        blocked("PINNED_STACK_LORA_GRPO_UNSUPPORTED", "COLAB_GRPO_CONTROL_PLANE_FAIL")
    tp_trainer = set(inspect.signature(GRPOTrainer.__init__).parameters)
    pre["trl_version"] = trl.__version__
    pre["trainer_accepts_peft_config"] = "peft_config" in tp_trainer
    pre["trainer_accepts_processing_class"] = "processing_class" in tp_trainer
    try:
        lcfg_sig = set(inspect.signature(_LoraConfig.__init__).parameters)
        pre["lora_config_accepts_target_modules"] = "target_modules" in lcfg_sig
        pre["lora_config_accepts_task_type"] = "task_type" in lcfg_sig
    except Exception:
        pre["lora_config_signature_error"] = True
    pre["lora_r"] = LORA_R; pre["lora_alpha"] = LORA_ALPHA; pre["lora_dropout"] = LORA_DROPOUT
    pre["lora_target_mode"] = LORA_TARGET_MODE
    try:
        pre["beta"] = float(getattr(cfg, "beta", None))
    except Exception:
        pre["beta"] = None
    pre["sync_ref_model"] = bool(getattr(cfg, "sync_ref_model", False))
    if not pre["trainer_accepts_processing_class"]:
        with open(RUN + "/lora_grpo_preflight.json", "w") as fh:
            json.dump(pre, fh, indent=2)
        STATE["gates"]["lora_api_preflight"] = "FAIL"
        emit("LORA_API", "FAIL: GRPOTrainer lacks processing_class (TRL %s)" % pre["trl_version"])
        blocked("PINNED_STACK_LORA_GRPO_UNSUPPORTED", "COLAB_GRPO_CONTROL_PLANE_FAIL")
    STATE["gates"]["lora_api_preflight"] = "PASS"
    emit("LORA_API", "PASS trl=%s peft=%s" % (pre["trl_version"], pre.get("peft_version")))

    # ---- §7 memory stages ----
    props = torch.cuda.get_device_properties(0)
    total_bytes = int(props.total_memory)
    torch.cuda.reset_peak_memory_stats()
    mem = {"total_gpu_bytes": total_bytes, "M0_empty": _mem_snapshot("M0")}

    # ---- M1: fp16 base + tokenizer + named-linear inspection ----
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _tok_lora = AutoTokenizer.from_pretrained(MODEL)
    if _tok_lora.pad_token is None:
        _tok_lora.pad_token = _tok_lora.eos_token
    dtype = torch.float16 if FP16 else (torch.bfloat16 if BF16 else torch.float32)
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to("cuda")
    base.eval()
    import torch.nn as _nn
    lin_names = [n for n, m in base.named_modules() if isinstance(m, _nn.Linear)]
    mem["M1_base_loaded"] = _mem_snapshot("M1")
    emit("LORA_M1", "base loaded %d linear modules" % len(lin_names))

    # ---- §2 target modules: all-linear if supported by the pinned PEFT, else derived ----
    target = LORA_TARGET_MODE
    all_linear_supported = False
    if target == "all-linear":
        try:
            _LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
                        task_type=_TaskType.CAUSAL_LM, target_modules="all-linear")
            all_linear_supported = True
        except Exception:
            all_linear_supported = False
    if not all_linear_supported:
        target = [n.rsplit(".", 1)[-1] for n in lin_names
                  if any(k in n.lower() for k in ("q_proj", "k_proj", "v_proj", "o_proj",
                                                  "gate_proj", "up_proj", "down_proj"))]
    lora_cfg = _LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
                           task_type=_TaskType.CAUSAL_LM, target_modules=target)
    with open(RUN + "/lora_config.json", "w") as fh:
        json.dump({"r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT, "bias": "none",
                   "task_type": "CAUSAL_LM", "target_mode": LORA_TARGET_MODE,
                   "all_linear_supported": all_linear_supported, "target_modules": target,
                   "named_linear_count": len(lin_names),
                   "named_linear_sample": sorted(lin_names)[:40]}, fh, indent=2)

    # ---- M2: attach LoRA ----
    peft_model = _get_peft_model(base, lora_cfg)
    mem["M2_lora_attached"] = _mem_snapshot("M2")
    emit("LORA_M2", "lora attached")

    # ---- §6 trainable-parameter proof (base must be frozen) ----
    total_p = 0; train_p = 0; unexpected = []
    for n, p in peft_model.named_parameters():
        nl = n.lower()
        total_p += p.numel()
        if p.requires_grad:
            if ("lora_" in nl) or ("lora" in nl):
                train_p += p.numel()
            else:
                unexpected.append(n)
    if unexpected:
        STATE["gates"]["lora_freeze"] = "FAIL"
        emit("LORA_FREEZE", "FAIL non-lora trainable: " + unexpected[:5])
        blocked("LORA_FREEZE_CONTRACT_FAIL", "COLAB_GRPO_CONTROL_PLANE_FAIL")
    frac = (train_p / total_p) if total_p else 0.0
    ps = {"total_parameters": total_p, "trainable_parameters": train_p,
          "trainable_fraction": round(frac, 8), "base_frozen": True, "lora_trainable": train_p > 0,
          "unexpected_trainable_base_parameters": 0}
    with open(RUN + "/parameter_summary.json", "w") as fh:
        json.dump(ps, fh, indent=2)
    STATE["gates"]["lora_freeze"] = "PASS" if (train_p > 0 and frac < 1.0) else "FAIL"
    emit("LORA_PARAMS", "total=%d train=%d frac=%.6f" % (total_p, train_p, frac))
    if STATE["gates"]["lora_freeze"] != "PASS":
        blocked("LORA_FREEZE_CONTRACT_FAIL", "COLAB_GRPO_CONTROL_PLANE_FAIL")

    # ---- M3: GRPOTrainer on the pre-applied PeftModel (§5 reference-memory gate, measured) ----
    _alloc_before = int(torch.cuda.memory_allocated())
    trainer_l = GRPOTrainer(model=peft_model, processing_class=_tok_lora,
                            reward_funcs=reward_correct, train_dataset=train, args=cfg)
    _alloc_after = int(torch.cuda.memory_allocated())
    ref_delta = max(0, _alloc_after - _alloc_before)
    mem["M3_trainer_constructed"] = _mem_snapshot("M3")
    if ref_delta > 3.0e9:
        mem["reference_delta_bytes"] = ref_delta
        mem["reference_behavior"] = "A_separate_full_model"
        mem["admission"] = "BLOCK"
        with open(RUN + "/memory_admission.json", "w") as fh:
            json.dump(mem, fh, indent=2)
        STATE["gates"]["lora_reference_memory"] = "BLOCK"
        emit("LORA_REF", "BLOCK delta=%d bytes -> separate full ref model resident" % ref_delta)
        blocked("LORA_GRPO_REFERENCE_MEMORY_BLOCK", "COLAB_GRPO_CONTROL_PLANE_FAIL")
    reference_behavior = "B_disable_adapter"
    mem["reference_delta_bytes"] = ref_delta
    mem["reference_behavior"] = reference_behavior
    STATE["gates"]["lora_reference_memory"] = "PASS"
    emit("LORA_REF", "behavior=%s delta=%d bytes" % (reference_behavior, ref_delta))

    # ---- §7 projected reserve before authorizing the first optimizer step ----
    # M4 optimizer state is created inside train() (no callback allowed); project it from M3 +
    # a conservative single-step activation estimate. Not treating uncertainty as admission.
    ACTIVATION_EST_BYTES = 4.0e9
    m3_alloc = mem["M3_trainer_constructed"].get("allocated_bytes", _alloc_after)
    projected_peak = max(m3_alloc, _alloc_after) + ACTIVATION_EST_BYTES
    projected_reserve = total_bytes - projected_peak
    admission_ok = projected_reserve >= 2.5e9
    mem["M4_optimizer_projected"] = {"note": "optimizer state created inside train(); projected, not measured pre-step",
                                     "activation_estimate_bytes": int(ACTIVATION_EST_BYTES),
                                     "projected_peak_bytes": int(projected_peak),
                                     "projected_reserve_bytes": int(projected_reserve),
                                     "admission_reserve_required_bytes": 2500000000,
                                     "admission_ok": bool(admission_ok)}
    with open(RUN + "/memory_admission.json", "w") as fh:
        json.dump(mem, fh, indent=2)
    STATE["gates"]["lora_memory_admission"] = "PASS" if admission_ok else "UNKNOWN"
    emit("LORA_ADMISSION", "projected_reserve=%.2fGB ok=%s" % (projected_reserve / 1e9, admission_ok))
    if not admission_ok:
        blocked("LORA_GRPO_T4_MEMORY_UNKNOWN", "COLAB_GRPO_CONTROL_PLANE_FAIL")

    pre["memory_admission_ok"] = bool(admission_ok)
    with open(RUN + "/lora_grpo_preflight.json", "w") as fh:
        json.dump(pre, fh, indent=2)
    lora_info = {"pre": pre, "mem": mem, "param": ps, "total_bytes": total_bytes}
    return trainer_l, lora_info


# ---- GRPO trainer construct ----
hb("TRAINER_READY", 0)
trainer = None
lora_info = None
try:
    train = pool.select(train_idx).map(lambda r: {"prompt": build_prompt(r)}, load_from_cache_file=False)
    keep = {"prompt", "answer", "reward_mode", "atol", "rtol", "bucket_prefix"}
    train = train.remove_columns([c for c in train.column_names if c not in keep])
    cfg = GRPOConfig(**CFG)
    STATE["gates"]["GRPOConfig_construct"] = "PASS"
    emit("GRPOCONFIG_CONSTRUCT", "ok")
    if TRAINING_METHOD == "lora":
        trainer, lora_info = build_lora_trainer(train, cfg)
    else:
        trainer = GRPOTrainer(model=MODEL, reward_funcs=reward_correct, train_dataset=train, args=cfg)
    STATE["gates"]["GRPOTrainer_construct"] = "PASS"
    emit("GRPOTRAINER_CONSTRUCT", "ok")
except Exception as e:
    STATE["gates"]["GRPOTrainer_construct"] = "FAIL"
    STATE["trainer_err"] = (type(e).__name__ + ": " + str(e))[:300]
    emit("GRPOTRAINER_CONSTRUCT", "FAIL:" + STATE["trainer_err"])
    blocked("TRL_API", "COLAB_GRPO_CONTROL_PLANE_FAIL")
bnd("GRPOTrainer_constructed", True, True)
assert not STATE["sandbox"].get("_has_callback", False)

_model = getattr(trainer, "model", None)
_tok = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)

# ---- PRE-train evaluation on base weights (§7) ----
hb("EVAL_PRE", 0)
pre = gen_eval_pass(_model, _tok) if (_model is not None and _tok is not None) else None
STATE["pre_train_eval"] = pre
emit("PRE_EVAL_MEAN", (pre or {}).get("mean_reward"))

# ---- ONE training execution ----
hb("TRAINING", 0)
bnd("trainer_train_entered", True, True)
torch.cuda.reset_peak_memory_stats()
w0 = time.time()
train_status = "success"
train_err = None
train_cls = None
try:
    trainer.train()
except Exception as e:
    train_status = "failed"
    em = (type(e).__name__ + ": " + str(e))
    train_err = em[:300]
    low = em.lower()
    if "out of memory" in low:
        train_cls = "CUDA_OOM"
    elif "compute capability" in low or "bf16" in low:
        train_cls = "PRECISION"
    else:
        train_cls = "UNKNOWN"
    log.exception("grpo train failed")
wall = time.time() - w0
bnd("trainer_train_returned", train_status == "success", True)
gstep = None
try:
    gstep = int(trainer.state.global_step)
except Exception:
    gstep = None
STATE["global_step"] = gstep
bnd("optimizer_step_1_completed", gstep is not None and gstep >= 1, True)
bnd("backward_entered", train_status == "success", False)
bnd("backward_completed", train_status == "success" and (gstep or 0) >= 1, False)
peak_alloc = int(torch.cuda.max_memory_allocated())
peak_res = int(torch.cuda.max_memory_reserved())
try:
    import resource
    host_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
except Exception:
    host_ram_mb = None
STATE["rewards_eval"] = STATE["rewards"]
STATE["gates"]["generation"] = "PASS" if STATE["gen_count"] >= 2 else "FAIL"
STATE["gates"]["reward_in_loop"] = "PASS" if len(STATE["rewards"]) >= 2 else "FAIL"
STATE["gates"]["backward"] = "PASS" if train_status == "success" else "FAIL"
STATE["gates"]["optimizer_step_1"] = "PASS" if (gstep or 0) >= 1 else "FAIL"
STATE["peak_allocated_bytes"] = peak_alloc
STATE["peak_reserved_bytes"] = peak_res
STATE["host_ram_peak_mb"] = round(host_ram_mb, 1) if host_ram_mb else None
STATE["wall_time_seconds"] = round(wall, 2)
mean_r = (sum(STATE["rewards"]) / len(STATE["rewards"])) if STATE["rewards"] else None
try:
    import statistics
    rstd = statistics.pstdev(STATE["rewards"]) if len(STATE["rewards"]) > 1 else None
except Exception:
    rstd = None
STATE["mean_reward"] = mean_r
STATE["reward_std"] = rstd
try:
    st = trainer.state.log_history
    last = st[-1] if st else {}
    STATE["trainer_loss"] = last.get("loss")
    STATE["trainer_kl"] = last.get("kl")
    STATE["trainer_reward"] = last.get("reward")
    STATE["trainer_reward_std"] = last.get("reward_std")
except Exception:
    pass
with open(RUN + "/metrics.jsonl", "w") as fh:
    fh.write(json.dumps({"step": int(gstep or 0), "rewards": STATE["rewards"], "mean_reward": mean_r,
                         "reward_std": rstd, "gen_count": STATE["gen_count"],
                         "completion_lengths": STATE["completion_lengths"],
                         "loss": STATE.get("trainer_loss"), "kl": STATE.get("trainer_kl"),
                         "global_step": gstep, "step_duration_seconds": round(wall, 4)}) + "\n")
hb("TRAINING", STATE["gen_count"])

# ---- POST-train evaluation on trained weights (§7) ----
hb("EVAL_POST", 0)
post = gen_eval_pass(_model, _tok) if (_model is not None and _tok is not None and train_status == "success") else None
STATE["post_train_eval"] = post
pre_mean = (pre or {}).get("mean_reward")
post_mean = (post or {}).get("mean_reward")
STATE["reward_delta"] = (post_mean - pre_mean) if (pre_mean is not None and post_mean is not None) else None
STATE["pre_train_nonzero_reward_rate"] = (pre or {}).get("nonzero_reward_rate")
STATE["post_train_nonzero_reward_rate"] = (post or {}).get("nonzero_reward_rate")
with open(RUN + "/eval_metrics.json", "w") as fh:
    json.dump({"pre_train_eval": pre, "post_train_eval": post, "reward_delta": STATE["reward_delta"],
               "eval_subset_indices": eval_idx, "eval_difficulty_tiers": difficulty_tiers}, fh, indent=2)
emit("POST_EVAL_MEAN", post_mean)
emit("REWARD_DELTA", STATE["reward_delta"])
emit("GEN_COUNT", STATE["gen_count"])
emit("REWARDS", STATE["rewards"])
emit("MEAN_REWARD", mean_r)
emit("GLOBAL_STEP", gstep)
emit("BACKWARD", "PASS" if train_status == "success" else "FAIL")
emit("OPTIMIZER_STEP", "PASS" if (gstep or 0) >= 1 else "FAIL")
emit("PEAK_ALLOC", peak_alloc)
emit("PEAK_RES", peak_res)
emit("WALL", round(wall, 2))
if train_status != "success":
    emit("TRAIN_ERR", train_err)
    STATE["error"] = train_err
try:
    sb.kill()
    STATE["sandbox"]["final_kill"] = "PASS"
except Exception:
    STATE["sandbox"]["final_kill"] = "FAIL"
hb("COMPLETE" if train_status == "success" else "FAILED", STATE["gen_count"])
STATE["classification"] = None if train_status == "success" else train_cls
verdict = "COLAB_GRPO_CONTROL_PLANE_PASS" if (train_status == "success" and STATE["gen_count"] >= 2 and (gstep or 0) >= 1) else "COLAB_GRPO_CONTROL_PLANE_FAIL"
STATE["terminal"] = verdict
emit("GRPO_DONE", "ok" if train_status == "success" else "fail")
emit("GRPO_VERDICT", verdict)
# ---- §10-§16 LoRA post-train: adapter, groups, classifications (full path untouched) ----
if TRAINING_METHOD == "lora" and lora_info is not None:
    STATE["training_method"] = "lora"
    STATE["lora_pre"] = lora_info["pre"]
    STATE["lora_mem"] = lora_info["mem"]
    STATE["lora_param"] = lora_info["param"]
    # NaN/Inf check
    _loss_val = STATE.get("trainer_loss")
    _kl_val = STATE.get("trainer_kl")
    _nan_inf = False
    for _v in (_loss_val, _kl_val):
        try:
            if _v is not None and (math.isnan(float(_v)) or math.isinf(float(_v))):
                _nan_inf = True
        except Exception:
            pass
    STATE["nan_inf_detected"] = _nan_inf
    emit("NAN_INF", _nan_inf)
    # §14 adapter save (adapter-only, NOT full base model)
    adapter_dir = ART + "/adapter"
    adapter_saved = False
    adapter_size = None
    try:
        import os as _os_a
        _os_a.makedirs(adapter_dir, exist_ok=True)
        _m = trainer.model
        while hasattr(_m, "module"):
            _m = _m.module
        _m.save_pretrained(adapter_dir)
        import glob as _g_a
        _sz = 0
        for _f in _g_a.glob(adapter_dir + "/adapter_model*"):
            _sz += _os_a.path.getsize(_f)
        adapter_saved = _os_a.path.isfile(adapter_dir + "/adapter_config.json")
        adapter_size = _sz
        emit("ADAPTER_SAVE", "ok bytes=%d" % _sz)
    except Exception as _ae:
        emit("ADAPTER_SAVE", "FAIL:" + type(_ae).__name__ + ":" + str(_ae)[:120])
    STATE["adapter_saved"] = adapter_saved
    STATE["adapter_size_bytes"] = adapter_size
    STATE["adapter_dir"] = adapter_dir
    STATE["base_model_id"] = MODEL
    try:
        import json as _j_a
        _ac = _os_a.path.join(adapter_dir, "adapter_config.json") if adapter_saved else None
        if _ac and _os_a.path.isfile(_ac):
            with open(_ac) as _af:
                _acfg = _j_a.load(_af)
            STATE["adapter_config"] = _acfg
            STATE["base_model_revision"] = _acfg.get("base_model_name_or_path", MODEL)
    except Exception:
        pass
    # §11/§12 group rewards
    groups = list(_lora_group_rewards)
    zero_adv = sum(1 for g in groups if not g.get("informative"))
    informative = sum(1 for g in groups if g.get("informative"))
    nonzero_rewards = sum(1 for g in groups for r in g["rewards"] if r > 0)
    signal_present = informative > 0
    STATE["grpo_groups"] = groups
    STATE["zero_advantage_group_count"] = zero_adv
    STATE["informative_group_count"] = informative
    STATE["nonzero_reward_count"] = nonzero_rewards
    STATE["grpo_signal_present"] = signal_present
    with open(RUN + "/group_rewards.jsonl", "w") as fh:
        for g in groups:
            fh.write(json.dumps(g) + "\n")
    emit("GROUPS", "total=%d informative=%d zero_adv=%d" % (len(groups), informative, zero_adv))
    # §16 four independent classifications
    memory_feasible = (not _nan_inf and train_status == "success"
                       and STATE["gates"].get("lora_memory_admission") == "PASS"
                       and STATE["gates"].get("lora_reference_memory") == "PASS")
    training_execution_valid = (train_status == "success" and (gstep or 0) >= 1
                                 and STATE["gates"].get("GRPOTrainer_construct") == "PASS")
    pre_mr = (pre or {}).get("mean_reward")
    post_mr = (post or {}).get("mean_reward")
    quality_improved = (pre_mr is not None and post_mr is not None and post_mr > pre_mr)
    STATE["memory_feasible"] = memory_feasible
    STATE["training_execution_valid"] = training_execution_valid
    STATE["grpo_signal_present"] = signal_present
    STATE["quality_improved"] = quality_improved
    # §13/§26 LoRA verdict
    if not memory_feasible:
        lora_verdict = "1P7B_LORA_GRPO_T4_MEMORY_BLOCKED"
    elif not training_execution_valid:
        lora_verdict = "1P7B_LORA_GRPO_T4_FAIL"
    elif signal_present:
        lora_verdict = "1P7B_LORA_GRPO_T4_PASS"
    else:
        lora_verdict = "1P7B_LORA_GRPO_T4_EXECUTION_PASS"
    STATE["lora_verdict"] = lora_verdict
    emit("LORA_VERDICT", lora_verdict)
    # post-hoc reserve verification (§9: observed reserve >= 2.0 GB)
    try:
        _obs_peak = int(torch.cuda.max_memory_reserved())
        _obs_res = lora_info["total_bytes"] - _obs_peak
        STATE["observed_peak_reserved_bytes"] = _obs_peak
        STATE["observed_reserve_bytes"] = _obs_res
        STATE["observed_reserve_gb"] = round(_obs_res / 1e9, 3)
        if _obs_res < 2.0e9:
            STATE["observed_reserve_margin_fail"] = True
            emit("RESERVE_MARGIN", "FAIL observed=%.2fGB < 2.0GB" % (_obs_res / 1e9))
        else:
            STATE["observed_reserve_margin_fail"] = False
            emit("RESERVE_MARGIN", "PASS observed=%.2fGB" % (_obs_res / 1e9))
    except Exception:
        pass
persist_final("success" if train_status == "success" else "failed", train_err)
sys.exit(0)
