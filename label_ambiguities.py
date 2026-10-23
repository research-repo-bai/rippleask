"""Reference implementation of the annotation-labeling procedure (Algorithm 1 in the paper).

For one annotated ambiguity it decides whether, on the case's own input, the decision is
CONSEQUENTIAL, MOOT, or UNRESOLVED; and for a MOOT one, whether a minimal realistic input
edit makes it LATENT or leaves it MOOT-AFTER-EDIT.

Prerequisites:
  * a PrepBench checkout (https://github.com/TsinghuaDatabaseGroup/prepbench): per-case
    inputs, reference solution, gold tables, and amb_kb.json annotations.
  * two OpenAI-compatible endpoints (a generator and a judge), configured via environment:
      GEN_BASE_URL, GEN_MODEL, GEN_API_KEY         (alternative-reading generator + perturber)
      JUDGE_BASE_URL, JUDGE_MODEL, JUDGE_API_KEY   (isolation/plausibility judge)
    API keys are read from the environment only; none are stored in this file or its output.

This is a reference implementation meant to be read alongside the paper, not a turnkey script.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from openai import OpenAI

EXEC_TIMEOUT = 120
SANDBOX = "(version 1)(allow default)(deny network*)"          # macOS seatbelt; adapt per OS
PY = sys.executable
TOL = 0.02                                                      # benchmark comparator: 2% relative


# ----------------------------------------------------------------- LLM access (keys from env only)
def _client(prefix: str) -> tuple[OpenAI, str]:
    base, model, key = (os.environ[f"{prefix}_BASE_URL"], os.environ[f"{prefix}_MODEL"],
                        os.environ[f"{prefix}_API_KEY"])
    return OpenAI(base_url=base, api_key=key, timeout=600, max_retries=4), model


def ask(prefix: str, system: str, user: str) -> str:
    client, model = _client(prefix)
    r = client.chat.completions.create(model=model,
                                       messages=[{"role": "system", "content": system},
                                                 {"role": "user", "content": user}])
    return r.choices[0].message.content or ""


def code_block(text: str) -> str | None:
    m = re.findall(r"```(?:python)?\s*\n([\s\S]*?)```", text)
    return m[-1] if m else None


def json_block(text: str) -> dict | None:
    for cand in ([code_block(text)] if "```" in text else []) + re.findall(r"\{[\s\S]*\}", text):
        try:
            return json.loads(cand)
        except Exception:
            pass
    return None


# ----------------------------------------------------------------- execution + comparator
def run_solve(code: str, inputs_dir: Path, out_dir: Path) -> bool:
    """Run a legacy zero-argument or solve(inputs_dir) program with the shared adapter."""
    inputs_dir, out_dir = Path(inputs_dir).resolve(), Path(out_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    (out_dir / "solution.py").write_text(code, encoding="utf-8")
    (out_dir / "driver.py").write_text(
        "import inspect,sys;from pathlib import Path\n"
        "inputs=Path(sys.argv[2]);o=Path(sys.argv[3])\n"
        "g={'__name__':'solution','__file__':str(inputs.parent/'solution.py')}\n"
        "exec(compile(Path(sys.argv[1]).read_text(),'s','exec'),g)\n"
        "solve=g['solve'];required=[p for p in inspect.signature(solve).parameters.values() "
        "if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_ONLY,p.POSITIONAL_OR_KEYWORD)]\n"
        "r=solve() if not required else solve(inputs)\n"
        "[df.to_csv(o/Path(n).name,index=False) for n,df in r.items()]\n", encoding="utf-8")
    try:
        proc = subprocess.run(["sandbox-exec", "-p", SANDBOX, PY, "driver.py", "solution.py",
                               str(inputs_dir), str(out_dir)], cwd=out_dir,
                              capture_output=True, text=True, timeout=EXEC_TIMEOUT)
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode == 0


def cell_diff(a_dir: Path, b_dir: Path) -> float:
    """Mean per-table cell disagreement after canonical row sorting and numeric tolerance.

    This auxiliary score uses different row alignment from the official comparator.
    Paired missing cells agree; a missing cell paired with a value disagrees.
    A missing or differently shaped table contributes 1.0.
    """
    names = {p.name for p in a_dir.glob("*.csv")} | {p.name for p in b_dir.glob("*.csv")}
    scores = []
    for n in sorted(names):
        fa, fb = a_dir / n, b_dir / n
        if not (fa.exists() and fb.exists()):
            scores.append(1.0); continue
        da, db = pd.read_csv(fa, dtype=str), pd.read_csv(fb, dtype=str)
        if da.shape != db.shape or set(da.columns) != set(db.columns):
            scores.append(1.0); continue
        if da.empty:
            scores.append(0.0); continue
        db = db[list(da.columns)]

        def num(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        def order(df):
            key = df.apply(lambda col: col.map(lambda v: f"{num(v):.2g}" if num(v) is not None
                                               else ("" if v != v else str(v).strip())))
            return df.iloc[key.assign(_i=range(len(df))).sort_values(list(df.columns) + ["_i"])["_i"]].values
        ra, rb = order(da), order(db)
        diff = total = 0
        for row_a, row_b in zip(ra, rb):
            for x, y in zip(row_a, row_b):
                total += 1
                nx, ny = num(x), num(y)
                if pd.isna(x) or pd.isna(y):
                    same = bool(pd.isna(x) and pd.isna(y))
                elif nx is not None and ny is not None:
                    same = abs(nx - ny) < 0.02 if (nx == 0 or ny == 0) else abs(nx - ny) / max(abs(nx), abs(ny)) < TOL
                else:
                    same = ("" if x != x else str(x).strip()) == ("" if y != y else str(y).strip())
                diff += not same
        scores.append(diff / max(total, 1))
    return sum(scores) / len(scores) if scores else 0.0


def official_equal(expected: Path, candidate: Path, config: Path) -> bool:
    """Use PrepBench's key-based comparator; put its src/ directory on PYTHONPATH."""
    from evaluate.core import evaluate
    passed, error = evaluate(str(expected), str(candidate), str(config))
    if error and error["error_type"] in {"CONFIG_MISSING", "EXCEPTION", "READ_ERROR"}:
        raise RuntimeError(f"Official comparison could not be completed: {error}")
    return passed


def valid_edit(original: Path, edited: Path) -> bool:
    """Require matching files/parsed columns and 1--3 changed rows per affected file."""
    files = {p.name for p in original.glob("*.csv")}
    if files != {p.name for p in edited.glob("*.csv")}:
        return False
    total = 0
    for name in sorted(files):
        before = (original / name).read_text(encoding="utf-8").splitlines()
        after = (edited / name).read_text(encoding="utf-8").splitlines()
        if not before or not after:
            return False
        # Compare the schema used by the CSV parser, permitting equivalent BOM
        # and empty-header serialization while rejecting renamed/added columns.
        try:
            original_columns = list(pd.read_csv(original / name, nrows=0).columns)
            edited_columns = list(pd.read_csv(edited / name, nrows=0).columns)
        except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
            return False
        if original_columns != edited_columns:
            return False
        a, b = Counter(before[1:]), Counter(after[1:])
        changed = max(sum((a - b).values()), sum((b - a).values()))
        if changed > 3:
            return False
        total += changed
    return total > 0


# ----------------------------------------------------------------- prompts (verbatim, as used in the paper)
GEN_SYS = """You are auditing a data-preparation benchmark. You get a natural-language request, the benchmark's reference solution, and one annotated ambiguity: a decision the request leaves open, together with the code the reference uses to resolve it.

Propose up to 2 distinct ALTERNATIVE resolutions of THIS decision only, each one a reasonable analyst could have chosen from the request's wording. An alternative must differ from the reference's choice, must not contradict anything the request states explicitly, and must leave every other decision exactly as the reference makes it.

For each alternative, return the COMPLETE modified solution (same `solve(inputs_dir)` signature and output file names), changing only the code that implements this decision. Format:

### Alternative 1
Description: <one sentence>
```python
<full code>
```

### Alternative 2
Description: <one sentence>
```python
<full code>
```

If only one reasonable alternative exists, return one."""

JUDGE_SYS = """You check one edit to a data-preparation solution. You get the original request, one annotated ambiguous decision, and a unified diff from the reference solution to a modified solution.

Answer two questions:
- isolated: does the diff change ONLY how this decision is resolved (incidental refactoring needed for that change is fine), and leave all other behaviour as it was?
- plausible: is the modified resolution a reasonable reading of the original request, one that does not contradict anything the request states explicitly?

Return only a ```json block: {"isolated": true|false, "plausible": true|false, "reason": "<one sentence>"}"""

PERT_SYS = """You are testing whether two readings of an ambiguous data-preparation decision can produce different results on realistic data. You get the request, previews of the input CSV files, the reference program, an alternative program that resolves ONE annotated decision differently, and that decision.

On the current inputs both programs produce the same output. Design the SMALLEST REALISTIC change to the input data under which the two readings would plausibly produce different outputs: add, duplicate or edit at most 3 rows in total per file, using values that fit the observed data (e.g. a tie, a value exactly on a boundary, a duplicate row, a missing value, a name with a middle name). Do not add columns, rename files, or change the data's format. The change must be something that could realistically occur in this kind of data.

If no realistic change of that size could make the two readings differ, reply exactly NO_REALISTIC_CHANGE and one sentence why.

Otherwise reply with one sentence describing the change, then one ```python block defining:
    def perturb(src_dir: Path, dst_dir: Path) -> None
which reads every CSV in src_dir, applies the change, and writes ALL files (changed or not) with the same names to dst_dir. Use only the standard library and pandas. Preserve the files' original columns and formatting as far as possible (e.g. read with dtype=str and keep_default_na=False, write with index=False)."""


def label_annotation(case_dir: Path, gold_dir: Path, reference_code: str, annotation: dict, work: Path) -> dict:
    """Return the original-input label and a separate perturbation_label (Algorithm 1)."""
    inputs = case_dir / "inputs"
    config = gold_dir / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"Official comparator configuration required: {config}")
    reference_out = work / "reference_original"
    if not run_solve(reference_code, inputs, reference_out) or not official_equal(gold_dir, reference_out, config):
        return {"label": "unresolved", "note": "reference program does not pass on the original input"}
    request = (case_dir / "query.md").read_text(encoding="utf-8")
    entry = "\n".join(f"{k}: {annotation[k]}" for k in ("id", "kind", "op") if annotation.get(k)) \
        + f"\nrequest text: {annotation.get('source_text') or annotation.get('intent')}\nreference code:\n{annotation['ref']}"

    reply = ask("GEN", GEN_SYS, f"# Request\n{request}\n\n# Reference\n```python\n{reference_code}\n```\n\n# Annotation\n{entry}")
    valid = []
    for k, chunk in enumerate(re.split(r"^###\s*Alternative\s*\d+", reply, flags=re.M)[1:3]):
        code = code_block(chunk)
        if not code or code.strip() == reference_code.strip():
            continue
        if not run_solve(code, inputs, work / f"alt{k}"):
            continue
        import difflib
        diff = "".join(difflib.unified_diff(reference_code.splitlines(True), code.splitlines(True), n=2))
        j = json_block(ask("JUDGE", JUDGE_SYS, f"# Request\n{request}\n\n# Decision\n{entry}\n\n# Diff\n```diff\n{diff}\n```")) or {}
        if j.get("isolated") and j.get("plausible"):
            valid.append({"code": code, "out": work / f"alt{k}", "desc": chunk.strip().splitlines()[0][:200]})

    if not valid:
        return {"label": "unresolved"}
    if any(not official_equal(gold_dir, a["out"], config) for a in valid):
        return {"label": "consequential"}

    # moot on this input: try a minimal realistic edit that could separate a reading from the reference
    alt = valid[0]
    prev = [f.name + ": " + "\n".join(f.read_text(encoding="utf-8", errors="replace").splitlines()[:6])
            for f in sorted(inputs.glob("*.csv"))]
    pr = ask("GEN", PERT_SYS, f"# Request\n{request}\n\n# Inputs\n" + "\n".join(prev)
             + f"\n\n# Reference\n```python\n{reference_code}\n```\n\n# Alternative\n```python\n{alt['code']}\n```")
    code = code_block(pr)
    if "NO_REALISTIC_CHANGE" in pr:
        return {"label": "moot", "perturbation_label": "no-edit-found"}
    if not code:
        return {"label": "moot", "perturbation_label": "unresolved", "note": "no perturbation program returned"}
    pert_in = work / "perturbed" / "inputs"
    pert_in.mkdir(parents=True, exist_ok=True)
    (work / "perturb.py").write_text(code, encoding="utf-8")
    (work / "pdriver.py").write_text(
        "import sys;from pathlib import Path\n"
        "g={'__name__':'p'};exec(compile(Path(sys.argv[1]).read_text(),'p','exec'),g)\n"
        "g['perturb'](Path(sys.argv[2]),Path(sys.argv[3]))\n", encoding="utf-8")
    try:
        result = subprocess.run(["sandbox-exec", "-p", SANDBOX, PY, "pdriver.py", "perturb.py", str(inputs), str(pert_in)],
                                cwd=work, capture_output=True, text=True, timeout=EXEC_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"label": "moot", "perturbation_label": "unresolved", "note": "perturbation timed out"}
    if result.returncode != 0 or not valid_edit(inputs, pert_in):
        return {"label": "moot", "perturbation_label": "unresolved", "note": "perturbation failed or violated the edit limit"}
    ok_r = run_solve(reference_code, pert_in, work / "ref_p")
    ok_a = run_solve(alt["code"], pert_in, work / "alt_p")
    if not (ok_r and ok_a):
        return {"label": "moot", "perturbation_label": "unresolved", "note": "a program failed on the edited input"}
    before = cell_diff(reference_out, alt["out"])
    after = cell_diff(work / "ref_p", work / "alt_p")
    latent = after > before and not official_equal(work / "ref_p", work / "alt_p", config)
    return {"label": "moot", "perturbation_label": "latent" if latent else "moot-after-edit",
            "divergence_on_input": round(before, 4), "divergence_after_edit": round(after, 4)}


if __name__ == "__main__":
    print(__doc__)
