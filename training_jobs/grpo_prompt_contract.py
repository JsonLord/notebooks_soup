#!/usr/bin/env python3
"""Dependency-free prompt/staged-contract + task-validity helpers for the
OpenTraining GRPO worker and the matched inference probe.

Pure stdlib ONLY: importable and unit-testable WITHOUT torch / trl / GPU / HF /
Colab. The training worker keeps a BYTE-IDENTICAL inline copy of these four
definitions so it stays a single self-contained file for `colab exec -f`; a
contract test (tests/test_grpo_worker_contract.py) AST-compares the four
definitions between this module and the worker to kill drift -- exactly the
precedent already set by grpo_eval_split.py.

Why this module exists (PROMPT_TASK_REDESIGN).

Two independent, model-independent defects made the earlier 360M and 1.7B probes
record reward 0 for every task, which was misread as a capability ceiling:

  (A) PROMPT_STAGING_CONTRACT_DEFECT -- the old prompt surfaced only the dataset
      row's nominal `files` field with no CSV columns and no SQLite
      table/column names, so a model (or a naive program) had to GUESS
      environmental facts the runtime already knew (observed model errors:
      `sqlite3.OperationalError: no such table: Matches` and
      `FileNotFoundError: ... crime.csv`). The model's task is to reason about
      the data, not to guess filenames.

  (B) STAGE-ONCE-NEVER-RESTAGE -- the worker staged only the FIRST eval bucket,
      then ran every other task's program in that same directory, so every
      non-first task hit FileNotFoundError, printed nothing and scored 0. Both
      defects are task-independent and model-independent; the staged
      filesystem is authoritative and must be inspected per task.

The invariant enforced here is

    PROMPT_FILE_CONTRACT == ACTUAL_STAGED_FILE_CONTRACT

The contract is derived from the ACTUAL staged bytes (never a second manually
curated list), and the prompt also carries a strict output contract. No gold
answer, no reference solution, and no grader internals are ever emitted.

task_validity is an EXPERIMENT eligibility annotation (NOT a dataset edit, NOT a
deletion, NOT grader softening): a task whose gold cannot be reproduced from
the staged data by the known-correct computation is quarantined as
GOLD_UNREPRODUCIBLE, kept in diagnostic logs, but excluded from the reward
denominator so it is counted as neither model success nor model failure.
"""
import ast
import json
import re


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


def build_prompt(r, contract=None, variant="P0"):
    """Build the chat messages for row `r`, disclosing the staged contract.

    When `contract` (from inspect_staged_contract) is provided, every staged
    filename plus its CSV columns / SQLite tables+columns is surfaced, and the
    prompt carries a strict output contract (use exact names; the final answer is
    the last non-empty stdout line). When `contract` is None it degrades to the
    legacy nominal-files prompt, so the trainer's dataset `.map` site -- where no
    bucket is staged yet -- keeps producing the SAME prompt as before with zero
    behavior change. No gold / solution / grader internals are ever emitted.

    `variant` selects the reasoning scaffolding (P0-P3). P0 is the original
    prompt with no scaffolding. P1-P3 add progressively richer internal
    task-solving checklists without providing any answer, solution, or grader
    detail. The output code contract is identical across all variants.
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
    _scaffolds = {
        "P0": "",
        "P1": ("\nBefore writing the program, internally determine:\n"
               "1. Which supplied file/database contains the needed information.\n"
               "2. Which exact columns/tables correspond to the question.\n"
               "3. What operation is required: filter / min / max / count / sum / average / ratio / join / SQL query.\n"
               "4. What exact answer format the question requests.\n"
               "Then output only the executable Python program.\n"),
        "P2": ("\nBefore writing the program, internally determine:\n"
               "1. Which supplied file/database contains the needed information.\n"
               "2. Which exact columns/tables correspond to the question.\n"
               "3. What operation is required: filter / min / max / count / sum / average / ratio / join / SQL query.\n"
               "4. What exact answer format the question requests.\n"
               "Then output only the executable Python program.\n"
               "When working with tabular data:\n"
               "- inspect the supplied column names exactly\n"
               "- convert numeric-looking strings when arithmetic is required\n"
               "- account for commas, percent signs, missing values, or other formatting when necessary\n"
               "- do not assume a field is numeric merely because the question sounds numeric\n"
               "- use the actual supplied schema rather than guessed field names\n"),
        "P3": ("\nBefore writing the program, internally determine:\n"
               "1. Which supplied file/database contains the needed information.\n"
               "2. Which exact columns/tables correspond to the question.\n"
               "3. What operation is required: filter / min / max / count / sum / average / ratio / join / SQL query.\n"
               "4. What exact answer format the question requests.\n"
               "Then output only the executable Python program.\n"
               "When working with tabular data:\n"
               "- inspect the supplied column names exactly\n"
               "- convert numeric-looking strings when arithmetic is required\n"
               "- account for commas, percent signs, missing values, or other formatting when necessary\n"
               "- do not assume a field is numeric merely because the question sounds numeric\n"
               "- use the actual supplied schema rather than guessed field names\n"
               "Internally:\n"
               "A. choose the relevant supplied data source\n"
               "B. load/query it using its exact filename/schema\n"
               "C. derive the requested value\n"
               "D. sanity-check that the result matches the requested quantity/type\n"
               "E. print only the requested final answer\n"
               "Prefer the simplest correct program.\n"
               "Avoid unnecessary helper functions or unrelated processing.\n"),
    }
    scaffold = _scaffolds.get(variant, "")
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


def extract_code(completion):
    text = completion[-1]["content"] if isinstance(completion, list) else completion
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    return (blocks[-1] if blocks else text or "").strip()


def ast_preflight(code):
    """Pure AST gate. Parse `code` and report parseability WITHOUT altering it.

    Returns {"ok": bool, "syntax_error_type", "syntax_error_line",
    "syntax_error_message"}. No repair, no replace, no decode, no regeneration
    (HARDEN_CODE_OUTPUT_CONTRACT §3/§4): an unparseable program is merely
    detected earlier than sandbox execution; the same terminal zero-reward
    outcome is unchanged. Empty/blank code is a parse failure too.
    """
    src = code if isinstance(code, str) else (code or "")
    if src.strip() == "":
        return {"ok": False, "syntax_error_type": "EMPTY_SOURCE",
                "syntax_error_line": None, "syntax_error_message": "no source"}
    try:
        ast.parse(src)
        return {"ok": True, "syntax_error_type": None,
                "syntax_error_line": None, "syntax_error_message": None}
    except SyntaxError as e:
        return {"ok": False, "syntax_error_type": type(e).__name__,
                "syntax_error_line": e.lineno, "syntax_error_message": e.msg}


def classify_form_error(code, parse):
    """Deterministic FORM taxonomy for one extracted program (no repair).

    `parse` is ast_preflight(code). Assign ONE primary terminal classification
    plus overlapping flags where the evidence supports them (§7). Flags:
      contains_literal_backslash_n (two-char backslash + n present) + count
    Primary order: if parse ok -> PARSE_OK; else the most specific of
    LITERAL_ESCAPED_NEWLINES / UNTERMINATED_STRING / INDENTATION_ERROR /
    UNMATCHED_DELIMITER / INVALID_SYNTAX / PARSE_OTHER.
    """
    src = code if isinstance(code, str) else (code or "")
    lit = src.count("\\n")
    flags = {"contains_literal_backslash_n": lit > 0, "literal_backslash_n_count": lit}
    if parse.get("ok"):
        return dict(flags, primary="PARSE_OK", flags=["PARSE_OK"])
    t = (parse.get("syntax_error_type") or "")
    m = (parse.get("syntax_error_message") or "").lower()
    if "invalid non-printable" in m or "null bytes" in m:
        primary = "PARSE_OTHER"
    elif lit > 0 and ("unterminated string" in m):
        primary = "UNTERMINATED_STRING"      # §7 example: lit flag + unterminated
    elif lit > 0:
        primary = "LITERAL_ESCAPED_NEWLINES"
    elif "unterminated string" in m:
        primary = "UNTERMINATED_STRING"
    elif "expected an indented block" in m or "unexpected indent" in m or t == "IndentationError":
        primary = "INDENTATION_ERROR"
    elif "was never closed" in m or "unmatched" in m or "parentheses" in m or "bracket" in m:
        primary = "UNMATCHED_DELIMITER"
    elif "invalid syntax" in m:
        primary = "INVALID_SYNTAX"
    else:
        primary = "PARSE_OTHER"
    flags_list = [primary]
    if lit > 0 and primary != "LITERAL_ESCAPED_NEWLINES":
        flags_list.append("LITERAL_ESCAPED_NEWLINES")
    return dict(flags, primary=primary, flags=flags_list)


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


def classify_truncation(code, parse, hit_cap):
    """Classify one AST failure as token-cap truncation vs genuine breakage.

    TOKEN_CAP_TRUNCATION requires BOTH evidence that the completion consumed the
    full budget (`hit_cap`, from the ACTUAL generated-token count) AND that it
    terminates syntactically incomplete (unterminated string / never-closed
    delimiter / EOF-while-scanning / trailing open token). A parse failure that
    finished before the cap, or that fails for a deterministic reason unrelated
    to a cut, is GENUINE_SYNTAX_FAILURE; empty/undecodable is UNKNOWN_PARSE_FAILURE.
    """
    if parse.get("ok"):
        return "PARSE_OK"
    m = (parse.get("syntax_error_message") or "").lower()
    if (not m) or (parse.get("syntax_error_type") or "") == "EMPTY_SOURCE":
        return "UNKNOWN_PARSE_FAILURE"
    incomplete = ("never closed" in m or "unterminated" in m or "eof while" in m
                  or "eol while" in m or "unexpected eof" in m)
    if not incomplete:
        s = (code or "").rstrip()
        if s.endswith((",", ":", "(", "[", "{", ".", "+", "-", "*", "/", "%", "=", "<", ">", "|", "&")):
            incomplete = True
        else:
            for a, b in (("(", ")"), ("[", "]"), ("{", "}")):
                if s.count(a) != s.count(b):
                    incomplete = True
                    break
        if (not incomplete) and (s.count("'") % 2 == 1 or s.count('"') % 2 == 1):
            incomplete = True
    if hit_cap and incomplete:
        return "TOKEN_CAP_TRUNCATION"
    return "GENUINE_SYNTAX_FAILURE"
