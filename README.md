# Which clarification questions matter? — code and labels

Supplementary material for the paper *Moot Here, Material Later: How Single-Input
Evaluation Misjudges Clarification Questions for Data-Preparation Agents*.

The paper studies, on [PrepBench](https://github.com/TsinghuaDatabaseGroup/prepbench), how
much an annotated ambiguity in a data-preparation task actually matters: whether it can change
the expected output on the case's own input, and whether a minimal realistic edit to that input
makes it matter.

PrepBench commit used: `993a3139da932e144d588f7b775d092827fc89c8` (2026-07-14).

## Contents

| File | What it is |
| --- | --- |
| `labels.csv`, `labels.json` | The labeling result for all **183** annotations in **33** PrepBench cases: each annotation's kind, the open decision, its label on the given input (`consequential` / `moot` / `unresolved`), and, for moot annotations drawn into the perturbation study, whether a minimal input edit makes it `latent` or leaves it `moot-after-edit`. |
| `reproduce.py` | Recomputes the audit and perturbation numbers in the paper from `labels.csv` (no API calls). |
| `label_ambiguities.py` | Reference implementation of the labeling procedure (Algorithm 1): generate alternative readings, execute them against the gold output, judge them, apply a minimal input edit, and classify. Reads API keys from the environment only. |

## `labels.csv` columns

`case`, `annotation_id`, `kind`, `decision`; `input_label` ∈ {`consequential`, `moot`, `unresolved`};
`perturbation_tested` (whether this annotation was in the perturbation study); `perturbation_label`
∈ {`latent`, `moot-after-edit`, `na`}; `divergence_on_input`, `divergence_after_edit` (fraction of
output cells the two readings differ in, under the benchmark comparator); `reviewer_votes`
(how many of three LLM reviewers accepted the perturbation item, for moot items only).

Annotations with `input_label = consequential` and `perturbation_tested = yes` are the
perturbation study's controls.

Notes on the two `divergence_*` columns:
* They are measured for the **single alternative the perturbation study used** (the first
  valid alternative), not necessarily the alternative that set `input_label`. An annotation can
  have several valid alternatives; it is labelled `consequential` if **any** of them differs from
  the gold output on the original input. So a `consequential` control can show
  `divergence_on_input = 0.0` here (the perturbation study happened to use a non-differing
  alternative) — e.g. `case_200 / 1_O2_filename_timestamp_parse_format`. This is provenance, not
  an error.
* Only the **40** sampled moot annotations and **12** controls that produced a runnable
  perturbation appear with `perturbation_tested = yes`; `labels.csv` contains the **29** moot and
  **9** control rows that ran. The other 11 moot samples are omitted: 6 admitted no realistic
  edit and 5 failed to run or exceeded the three-row limit.

## Reproduce

```bash
python reproduce.py
```

prints the moot share among resolved annotations (62%, 95% CI [54, 70]), the breakdown by kind,
and the latent share among sampled moot annotations.

## Re-running the labeling from scratch

`label_ambiguities.py` is a reference implementation to read alongside the paper. It needs a
PrepBench checkout (per-case `inputs/`, `query.md`, `amb_kb.json`, reference solutions, and gold
tables) and two OpenAI-compatible endpoints, configured through environment variables
(`GEN_BASE_URL`/`GEN_MODEL`/`GEN_API_KEY`, `JUDGE_BASE_URL`/`JUDGE_MODEL`/`JUDGE_API_KEY`). No
API keys are stored in this release. The generation, judging, and perturbation prompts are
included verbatim in the script.

It sandboxes candidate programs with macOS `sandbox-exec` (see the `SANDBOX` constant); on Linux
or Windows, replace that call with another sandbox (e.g. `firejail`, a container, or `nsjail`).

## Scope and limits

Labels are defined with respect to PrepBench's comparator (2% numeric tolerance, order-insensitive)
and the alternative readings and input edits our procedure tried; "moot" means moot under those,
not under every possible reading or input. The perturbation reviewers are LLMs. The agent-study
model outputs are not included in this minimal release.
