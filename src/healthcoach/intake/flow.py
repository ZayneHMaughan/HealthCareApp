"""Data-driven intake flow. UI-agnostic: the UI asks `session.current`, shows it, calls `session.answer(value)`.

The flow is a pure function of the answers given so far (the next step is recomputed each time),
so going back is just deleting the last answer. Order of the flow:

  1 emergency screener (before anything else; halting items stop the flow immediately)
  2 age, sex, pregnancy (if relevant)
  3 symptoms (up to 5)
  4 severity, duration
  5 red-flag follow-ups for the chosen symptoms (highest priority first)
  6 context (blood thinners, immune status), optional readings, optional free text

After every answer the rules are re-run; once they reach EMERGENCY nothing can lower the result,
so the flow stops asking questions.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from ..rules import evaluate
from ..schema import Intake, Vitals

DATA = Path(__file__).parent / "data"
MAX_SYMPTOMS = 5
PLACEHOLDER_AGE = 30.0  # only used if an emergency screener item halts the flow before age is asked


@dataclass
class Step:
    id: str
    kind: str  # multi_select | single_select | yes_no_unsure | number | scale | text
    text: str
    options: list[dict] = field(default_factory=list)  # {"value", "label"}
    why: str | None = None
    optional: bool = False
    min: float | None = None
    max: float | None = None
    units: list[str] | None = None


@dataclass
class Draft:
    age: float | None = None
    sex: str = "unknown"
    symptoms: set[str] = field(default_factory=set)
    modifiers: set[str] = field(default_factory=set)
    severity: int | None = None
    duration_hours: float | None = None
    pregnant: bool = False
    immunocompromised: bool = False
    on_anticoagulants: bool = False
    vitals: dict = field(default_factory=dict)
    vitals_picked: list[str] = field(default_factory=list)
    free_text: str = ""
    assumptions: list[str] = field(default_factory=list)
    halted: bool = False
    halt_reason: str | None = None

    def to_intake(self) -> Intake:
        return Intake(age=self.age if self.age is not None else PLACEHOLDER_AGE, sex=self.sex,
                      symptoms=set(self.symptoms), modifiers=set(self.modifiers), severity=self.severity,
                      duration_hours=self.duration_hours, pregnant=self.pregnant,
                      immunocompromised=self.immunocompromised, on_anticoagulants=self.on_anticoagulants,
                      vitals=Vitals(**self.vitals), free_text=self.free_text)


@dataclass
class IntakeResult:
    intake: Intake
    complete: bool                 # True when finished or halted
    halted_reason: str | None
    assumptions: list[str]         # things we assumed ("not sure" treated as yes, placeholder age)


def _to_number(step: Step, value) -> float:
    unit = None
    if isinstance(value, dict):
        unit, value = value.get("unit"), value.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Please enter a number.")
    if unit == "F":
        value = (value - 32) * 5 / 9
    elif step.units and unit not in (None, *step.units):
        raise ValueError("Unknown unit.")
    return float(value)


def validate(step: Step, value) -> None:
    k = step.kind
    if k == "multi_select":
        if not isinstance(value, list):
            raise ValueError("Choose from the list.")
        allowed = {o["value"] for o in step.options}
        bad = [v for v in value if v not in allowed]
        if bad:
            raise ValueError(f"Unknown option: {bad[0]}")
        if "none" in value and len(value) > 1:
            raise ValueError("'None' can't be combined with other choices.")
        if not value and not step.optional:
            raise ValueError("Pick at least one, or choose 'None'.")
        if step.id == "symptoms" and len(value) > MAX_SYMPTOMS:
            raise ValueError(f"Please choose up to {MAX_SYMPTOMS}.")
    elif k == "single_select":
        if value not in {o["value"] for o in step.options}:
            raise ValueError("Choose one of the options.")
    elif k == "yes_no_unsure":
        if value not in ("yes", "no", "unsure"):
            raise ValueError("Answer yes, no, or not sure.")
    elif k == "scale":
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10:
            raise ValueError("Pick a number from 0 to 10.")
    elif k == "number":
        if value is None and step.optional:
            return
        n = _to_number(step, value)
        if not (step.min <= n <= step.max):
            if step.id == "age":
                raise ValueError(f"Please enter an age between {step.min:.0f} and {step.max:.0f}.")
            raise ValueError("That reading looks unusual. Please re-check it, or skip it.")
    elif k == "text":
        if not isinstance(value, str) or len(value) > 2000:
            raise ValueError("Please keep this under 2000 characters.")


class Flow:
    def __init__(self, data_dir: Path = DATA):
        self.q = json.loads((data_dir / "questions.json").read_text())
        self.sym = json.loads((data_dir / "symptoms.json").read_text())
        self.screener = {i["id"]: i for i in self.q["screener"]["items"]}
        self.durations = {d["id"]: d for d in self.q["durations"]}
        self.mod_q = {q["modifier"]: q for q in self.q["modifier_questions"]}
        self.ctx_q = {q["field"]: q for q in self.q["context_questions"]}
        self.vit = {v["name"]: v for v in self.q["vitals"]}

    # ---- symptom picker helpers -------------------------------------------------
    def symptom_options(self) -> list[dict]:
        cats = self.sym["categories"]
        items = sorted(self.sym["groups"].items(), key=lambda kv: (cats[kv[1]["category"]], kv[1]["label"]))
        return [{"value": g, "label": s["label"], "category": cats[s["category"]]} for g, s in items] + \
               [{"value": "none", "label": "None of these match"}]

    def search_symptoms(self, query: str, limit: int = 8) -> list[str]:
        words = [w for w in query.lower().split() if len(w) > 1]
        scored = []
        for g, s in self.sym["groups"].items():
            hay = " ".join([s["label"].lower(), *[x.lower() for x in s["synonyms"]]])
            score = sum(w in hay for w in words)
            if score:
                scored.append((score, g))
        return [g for _, g in sorted(scored, key=lambda t: -t[0])[:limit]]

    # ---- sequence ---------------------------------------------------------------------
    def _sequence(self, d: Draft):
        sc = self.q["screener"]
        yield Step("screener", "multi_select", sc["text"],
                   options=[{"value": i["id"], "label": i["label"]} for i in sc["items"]] +
                           [{"value": "none", "label": "None of these"}],
                   why="Some situations need emergency care before any questions.")
        yield Step("age", "number", "How old are you?", min=0, max=120)
        yield Step("sex", "single_select", "What is your sex?",
                   options=[{"value": "female", "label": "Female"}, {"value": "male", "label": "Male"},
                            {"value": "unknown", "label": "Other / prefer not to say"}],
                   why="Some conditions and risks differ by sex.")
        if d.sex != "male" and 12 <= d.age <= 55:
            yield Step("pregnant", "yes_no_unsure", "Are you pregnant?",
                       why="Pregnancy changes which symptoms need urgent care.")
        yield Step("symptoms", "multi_select", f"What are you experiencing? Choose up to {MAX_SYMPTOMS}.",
                   options=self.symptom_options())
        if not d.symptoms:
            yield Step("free_text", "text", "Describe what's going on in your own words.", optional=True)
            return
        yield Step("severity", "scale", "How bad is it at its worst? (0 = barely noticeable, 10 = worst imaginable)",
                   min=0, max=10)
        yield Step("duration", "single_select", "How long has this been going on?",
                   options=[{"value": x["id"], "label": x["label"]} for x in self.q["durations"]])
        for q in sorted(self.q["modifier_questions"], key=lambda q: q["priority"]):
            if d.symptoms & set(q["symptoms_any"]) and q["modifier"] not in d.modifiers:
                yield Step(f"mod:{q['modifier']}", "yes_no_unsure", q["text"], why=q["why"])
        for q in self.q["context_questions"]:
            if d.symptoms & set(q["symptoms_any"]):
                yield Step(f"ctx:{q['field']}", "yes_no_unsure", q["text"], why=q["why"])
        avail = [v for v in self.q["vitals"] if d.symptoms & set(v["symptoms_any"])]
        if avail:
            yield Step("vitals_pick", "multi_select", "Do you have any of these readings handy? (Optional)",
                       options=[{"value": v["name"], "label": v["label"]} for v in avail], optional=True,
                       why="Readings can only raise concern, never lower it. A normal reading doesn't mean you're fine.")
            for name in d.vitals_picked:
                v = self.vit[name]
                yield Step(f"vital:{name}", "number", v["label"], optional=True, min=v["min"], max=v["max"],
                           units=v["units"])
        yield Step("free_text", "text", "Anything else you'd like to add? (Optional)", optional=True)

    def next_step(self, answers: dict) -> tuple[Step | None, Draft]:
        d = Draft()
        for step in self._sequence(d):
            if step.id not in answers:
                return step, d
            self._apply(step, answers[step.id], d)
            self._check_halt(d)
            if d.halted:
                break
        return None, d

    def _check_halt(self, d: Draft) -> None:
        if d.halted or d.age is None:
            return
        ev = evaluate(d.to_intake())
        if ev.emergency or ev.crisis:
            d.halted, d.halt_reason = True, "rules: " + "; ".join(ev.reasons[-2:])

    def _apply(self, step: Step, value, d: Draft) -> None:
        sid = step.id
        if sid == "screener":
            for iid in value:
                if iid == "none":
                    continue
                item = self.screener[iid]
                d.symptoms |= set(item["symptoms"])
                d.modifiers |= set(item["modifiers"])
                if item["halt"]:
                    d.halted, d.halt_reason = True, f"screener: {iid}"
        elif sid == "age":
            d.age = _to_number(step, value)
        elif sid == "sex":
            d.sex = value
        elif sid == "pregnant":
            d.pregnant = value in ("yes", "unsure")
            if value == "unsure":
                d.assumptions.append("Treated as possibly pregnant because you weren't sure.")
        elif sid == "symptoms":
            d.symptoms |= {v for v in value if v != "none"}
        elif sid == "severity":
            d.severity = value
        elif sid == "duration":
            d.duration_hours = self.durations[value]["hours"]
        elif sid.startswith("mod:"):
            q = self.mod_q[sid[4:]]
            if value == "yes" or (value == "unsure" and q["unsure"] == "yes"):
                d.modifiers.add(q["modifier"])
                if value == "unsure":
                    d.assumptions.append(f"Treated as 'yes' because you weren't sure: {q['text']}")
        elif sid.startswith("ctx:"):
            q = self.ctx_q[sid[4:]]
            if value == "yes" or (value == "unsure" and q["unsure"] == "yes"):
                setattr(d, q["field"], True)
                if value == "unsure":
                    d.assumptions.append(f"Treated as 'yes' because you weren't sure: {q['text']}")
        elif sid == "vitals_pick":
            d.vitals_picked = [v for v in value if v != "none"]
        elif sid.startswith("vital:"):
            if value is not None:
                d.vitals[sid[6:]] = _to_number(step, value)
        elif sid == "free_text":
            d.free_text = value


class IntakeSession:
    def __init__(self, flow: Flow | None = None):
        self.flow = flow or Flow()
        self.answers: dict = {}
        self._order: list[str] = []

    @property
    def current(self) -> Step | None:
        return self.flow.next_step(self.answers)[0]

    def answer(self, value) -> None:
        step = self.current
        if step is None:
            raise RuntimeError("Intake is already complete.")
        validate(step, value)
        self.answers[step.id] = value
        self._order.append(step.id)

    def back(self) -> None:
        if self._order:
            del self.answers[self._order.pop()]

    def result(self) -> IntakeResult:
        step, d = self.flow.next_step(self.answers)
        notes = list(d.assumptions)
        if d.age is None:
            notes.append("Age was not collected because an emergency was flagged first.")
        return IntakeResult(d.to_intake(), complete=step is None, halted_reason=d.halt_reason, assumptions=notes)
