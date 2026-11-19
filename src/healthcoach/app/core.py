"""App logic, independent of HTTP: sessions, intake steps as JSON, and the final response.

Privacy: sessions live in memory only, expire after `ttl` seconds idle, and are never written anywhere."""
from __future__ import annotations
import secrets
import threading
import time
from ..intake.flow import Flow, IntakeSession, Step
from ..pipeline import Response, run
from ..resources import RESOURCES
from ..schema import Tier

TIER_LABEL = {Tier.EMERGENCY: "Call emergency services now", Tier.URGENT: "Get checked today",
              Tier.SEE_DOCTOR_SOON: "See a clinician in the next few days", Tier.SELF_CARE: "May be manageable at home"}


class AppError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


class SessionStore:
    def __init__(self, ttl: float = 1800.0, max_sessions: int = 50, clock=time.monotonic):
        self.ttl, self.max, self.clock = ttl, max_sessions, clock
        self._s: dict[str, tuple[IntakeSession, float]] = {}
        self._lock = threading.Lock()

    def _purge(self) -> None:
        now = self.clock()
        for k in [k for k, (_, t) in self._s.items() if now - t > self.ttl]:
            del self._s[k]
        while len(self._s) >= self.max:                       # evict the least recently used
            del self._s[min(self._s, key=lambda k: self._s[k][1])]

    def create(self, session: IntakeSession) -> str:
        with self._lock:
            self._purge()
            sid = secrets.token_urlsafe(24)
            self._s[sid] = (session, self.clock())
            return sid

    def get(self, sid) -> IntakeSession:
        with self._lock:
            item = self._s.get(sid) if isinstance(sid, str) else None
            if item is None or self.clock() - item[1] > self.ttl:
                self._s.pop(sid, None) if isinstance(sid, str) else None
                raise AppError(404, "Your session expired. Please start again.")
            self._s[sid] = (item[0], self.clock())
            return item[0]

    def delete(self, sid) -> None:
        with self._lock:
            if isinstance(sid, str):
                self._s.pop(sid, None)

    def __len__(self) -> int:
        return len(self._s)


class HealthCoach:
    def __init__(self, flow: Flow | None = None, engine=None, llm=None, retriever=None, store: SessionStore | None = None,
                 region: str = "US", info: dict | None = None):
        self.flow, self.engine, self.llm, self.retriever = flow or Flow(), engine, llm, retriever
        self.store, self.region, self.info = store or SessionStore(), region, info or {}
        self._run_lock = threading.Lock()     # the local LLM is not thread-safe

    # ---- steps ----------------------------------------------------------------------------------------------------
    def _step_dict(self, session: IntakeSession) -> dict | None:
        step: Step | None = session.current
        if step is None:
            return None
        d = {"id": step.id, "kind": step.kind, "text": step.text, "why": step.why, "optional": step.optional,
             "min": step.min, "max": step.max, "units": step.units, "index": len(session.answers) + 1,
             "options": [dict(o) for o in step.options]}
        if step.id == "symptoms":     # let the browser search synonyms too
            groups = self.flow.sym["groups"]
            for o in d["options"]:
                g = groups.get(o["value"])
                o["search"] = " ".join([g["label"], *g["synonyms"]]).lower() if g else ""
        return d

    def start(self) -> dict:
        session = IntakeSession(self.flow)
        return {"sid": self.store.create(session), "step": self._step_dict(session)}

    def answer(self, sid, value) -> dict:
        session = self.store.get(sid)
        if session.current is None:
            raise AppError(409, "This questionnaire is already complete.")
        try:
            session.answer(value)
        except ValueError as e:
            raise AppError(400, str(e))
        step = self._step_dict(session)
        return {"done": step is None, "step": step}

    def back(self, sid) -> dict:
        session = self.store.get(sid)
        session.back()
        return {"step": self._step_dict(session)}

    def end(self, sid) -> dict:
        self.store.delete(sid)
        return {}

    # ---- result ---------------------------------------------------------------------------------------------------
    def result(self, sid) -> dict:
        session = self.store.get(sid)
        ir = session.result()
        if not ir.complete:
            raise AppError(409, "Please finish the questions first.")
        with self._run_lock:
            resp = run(ir.intake, engine=self.engine, llm=self.llm, retriever=self.retriever, region=self.region)
        return self.response_to_dict(resp, ir)

    def response_to_dict(self, resp: Response, ir) -> dict:
        labels = self.flow.sym["groups"]
        d = {"tier": resp.tier.name.lower(), "tier_label": TIER_LABEL[resp.tier], "crisis": resp.crisis,
             "out_of_scope": resp.out_of_scope, "ended_early": bool(ir.halted_reason), "assumptions": ir.assumptions,
             "reasons": resp.reasons, "used_fallback": resp.used_fallback,
             "fallback_reason": resp.violations[:3] if resp.used_fallback else [],
             "uncovered": [labels.get(g, {}).get("label", g.replace("_", " ")) for g in resp.uncovered],
             "sources": resp.sources, "attribution": resp.attribution, "explanation": None, "message": None,
             "extra_resources": []}
        if resp.llm_output and not resp.used_fallback:
            d["explanation"] = {k: resp.llm_output.get(k) for k in ("considerations", "follow_up_questions",
                                                                    "when_to_seek_care")}
        else:
            d["message"] = resp.message
        if "poisoning_overdose" in ir.intake.symptoms:
            d["extra_resources"].append(RESOURCES[self.region]["poison"])
        return d

    def health(self) -> dict:
        kb = None
        if self.retriever is not None:
            idx = self.retriever.index
            kb = {"chunks": len(idx.chunks),
                  "sources": {k: v.get("source_date") for k, v in idx.manifest.get("sources", {}).items()},
                  "age_days": idx.age_days(), "stale": idx.stale()}
        return {"rules": True, "knowledge_base": kb, "language_model": bool(self.llm), "risk_model": self.engine is not None,
                "retrieval_budget": getattr(self.retriever, "budget", None), **self.info}
