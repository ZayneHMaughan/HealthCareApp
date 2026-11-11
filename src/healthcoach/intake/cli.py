"""Terminal demo of the intake flow + rules/classifier pipeline:  python -m healthcoach.intake.cli"""
from __future__ import annotations
from ..pipeline import run
from .flow import Flow, IntakeSession, Step


def _pick(step: Step, multi: bool):
    for n, o in enumerate(step.options, 1):
        print(f"  {n}. {o['label']}")
    raw = input("> ").strip()
    if not raw and step.optional:
        return []
    idx = [int(x) for x in raw.replace(",", " ").split()]
    vals = [step.options[i - 1]["value"] for i in idx]
    return vals if multi else vals[0]


def _symptoms(flow: Flow):
    chosen: list[str] = []
    while True:
        q = input("Describe a symptom (or press Enter when done, 'none' if nothing fits): ").strip()
        if not q:
            return chosen or ["none"]
        if q == "none":
            return ["none"]
        hits = flow.search_symptoms(q)
        if not hits:
            print("  No match, try other words.")
            continue
        for n, g in enumerate(hits, 1):
            print(f"  {n}. {flow.sym['groups'][g]['label']}")
        pick = input("Pick a number (or Enter to skip): ").strip()
        if pick.isdigit() and 1 <= int(pick) <= len(hits) and hits[int(pick) - 1] not in chosen:
            chosen.append(hits[int(pick) - 1])


def ask(step: Step, flow: Flow):
    print(f"\n{step.text}")
    if step.why:
        print(f"  (why we ask: {step.why})")
    if step.id == "symptoms":
        return _symptoms(flow)
    if step.kind == "multi_select":
        print("Enter numbers separated by commas" + (" (blank to skip)" if step.optional else ""))
        return _pick(step, True)
    if step.kind == "single_select":
        return _pick(step, False)
    if step.kind == "yes_no_unsure":
        return {"y": "yes", "n": "no", "u": "unsure"}.get(input("y / n / u (not sure): ").strip().lower()[:1], "")
    if step.kind == "scale":
        return int(input("0-10: "))
    if step.kind == "number":
        raw = input("number (blank to skip): " if step.optional else "number: ").strip()
        if not raw:
            return None
        val = float(raw)
        if step.units:
            return {"value": val, "unit": input(f"unit {step.units}: ").strip().upper() or step.units[0]}
        return val
    return input("> ")


def main() -> None:
    flow, s = Flow(), IntakeSession()
    while (step := s.current) is not None:
        try:
            s.answer(ask(step, flow))
        except (ValueError, IndexError, KeyError) as e:
            print(f"  ! {e or 'Invalid input'}")
    res = s.result()
    resp = run(res.intake)
    print("\n--- result ---")
    print("tier:", resp.tier.name, "| crisis:", resp.crisis, "| ended early:", res.halted_reason)
    for r in resp.reasons:
        print("  ", r)
    for a in res.assumptions:
        print("  assumed:", a)
    print(resp.message)


if __name__ == "__main__":
    main()
