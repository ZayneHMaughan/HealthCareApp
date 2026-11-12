# healthcoach

Desktop, on-device health guidance tool. Not a diagnostic device. Rules and thresholds are drafts for clinician review.

## Layout
```
src/healthcoach/
  schema.py      Intake, Vitals, Tier
  rules.py       hard-coded red-flag layer (runs first, can only raise urgency)
  risk.py        loads models/risk_model/ bundle, returns score + tier
  safety.py      post-checks on LLM output
  pipeline.py    rules -> classifier -> retrieval + LLM -> safety check
  resources.py   region-specific emergency/crisis contacts
  intake/
    flow.py      data-driven question flow + IntakeSession (UI-agnostic)
    cli.py       terminal demo: python -m healthcoach.intake.cli
    data/        questions.json (screener, follow-ups, context, readings),
                 symptoms.json (patient-facing labels/synonyms), cc_symptom_mapping.csv
evals/
  cases/*.jsonl  test cases (seed set; grow to 100+, clinician-reviewed)
  run_evals.py   harness, exits 1 on any under-triage
tests/           unit tests
```
Planned (not included): `training/` (feature build + classifier), `finetune/` (synthetic data + LoRA),
`retrieval/` (vetted-source index), `app/` (desktop UI), `models/` (git-ignored artifacts).

## Quickstart
```
pip install -e ".[dev]"
python evals/run_evals.py
pytest
```

## Intake flow
Screener (emergencies first) -> age/sex -> symptoms -> severity/duration -> red-flag follow-ups ->
context -> optional readings -> optional free text. Rules re-run after every answer; once they
reach EMERGENCY the flow stops asking. "Not sure" answers are treated conservatively on
high-stakes questions and recorded in `IntakeResult.assumptions`.
Edit `questions.json` / `symptoms.json` (no code changes needed) and run the tests: they check that
every modifier the rules read can be collected, that no question is dead, and that every seed eval case
can be expressed through the flow.

## Rules of the road
- Final tier = max(rules, classifier). The LLM can explain a tier but never lower it.
- Emergency, crisis, and out-of-scope responses never wait on the LLM.
- Every rule change must keep `python evals/run_evals.py` passing; add a case for each new rule.
- Never commit the ER dataset or patient-level derivatives.
