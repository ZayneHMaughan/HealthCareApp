import http.client
import json
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from healthcoach.app.core import AppError, HealthCoach, SessionStore
from healthcoach.app.server import MAX_BODY, STATIC, make_server
from healthcoach.intake.flow import IntakeSession
from healthcoach.llm import schema_for_prompt
from healthcoach.retrieval import Retriever
from test_retrieval import index

ROOT = Path(__file__).parent.parent


class Served:
    def __init__(self, coach):
        self.srv = make_server(coach, "127.0.0.1", 0)
        self.port = self.srv.server_address[1]
        self.t = threading.Thread(target=self.srv.serve_forever, daemon=True)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.srv.shutdown(); self.srv.server_close()

    def call(self, method, path, body=None, headers=None, raw=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        h = dict(headers or {})
        data = raw
        if body is not None:
            data = json.dumps(body); h.setdefault("Content-Type", "application/json")
        c.request(method, path, data, h)
        r = c.getresponse()
        text = r.read()
        try:
            return r.status, json.loads(text), r
        except ValueError:
            return r.status, text, r


DEFAULTS = {"screener": ["none"], "age": 30, "sex": "male", "symptoms": ["cough"], "severity": 2,
            "duration": "about_a_day", "vitals_pick": [], "free_text": ""}


def drive(s: Served, answers=None, default_yn="no"):
    answers = {**DEFAULTS, **(answers or {})}
    _, r, _ = s.call("POST", "/api/start", {})
    sid, step = r["sid"], r["step"]
    for _ in range(40):
        if step is None:
            break
        v = answers.get(step["id"], default_yn if step["kind"] == "yes_no_unsure" else None)
        assert v is not None, f"unscripted step {step['id']}"
        st, r, _ = s.call("POST", "/api/answer", {"sid": sid, "value": v})
        assert st == 200, r
        step = r["step"]
    return sid


def test_emergency_screener_ends_immediately_with_emergency_result():
    with Served(HealthCoach()) as s:
        _, r, _ = s.call("POST", "/api/start", {})
        st, r2, _ = s.call("POST", "/api/answer", {"sid": r["sid"], "value": ["chest"]})
        assert r2["done"] is True and r2["step"] is None
        _, res, _ = s.call("POST", "/api/result", {"sid": r["sid"]})
        assert res["tier"] == "emergency" and "911" in res["message"] and res["ended_early"]


def test_overdose_shows_poison_control_and_self_harm_shows_988():
    with Served(HealthCoach()) as s:
        for item, needle in (("overdose", "1-800-222-1222"), ("self_harm", "988")):
            sid = drive(s, {"screener": [item]})
            _, res, _ = s.call("POST", "/api/result", {"sid": sid})
            text = json.dumps(res)
            assert res["tier"] == "emergency" and needle in text, item


def test_rules_only_mode_returns_a_standard_message():
    with Served(HealthCoach()) as s:
        sid = drive(s)
        _, res, _ = s.call("POST", "/api/result", {"sid": sid})
        assert res["tier"] == "self_care" and res["message"] and res["explanation"] is None


def test_grounded_explanation_with_sources_and_attribution():
    def stub(prompt):
        sch = schema_for_prompt(prompt)
        return json.dumps({"urgency": sch["properties"]["urgency"]["enum"][0], "considerations": ["Common causes vary."],
                           "follow_up_questions": [], "when_to_seek_care": "Get checked if it gets worse.",
                           "sources": [sch["properties"]["sources"]["items"]["enum"][0]]})
    with Served(HealthCoach(llm=stub, retriever=Retriever(index()))) as s:
        sid = drive(s, {"symptoms": ["abdominal_pain"]})
        _, res, _ = s.call("POST", "/api/result", {"sid": sid})
        assert res["explanation"]["when_to_seek_care"] and res["sources"][0]["title"] == "Abdominal Pain"
        assert res["attribution"] and not res["used_fallback"]
        _, h, _ = s.call("GET", "/api/health")
        assert h["knowledge_base"]["chunks"] > 0 and h["language_model"] is True


def test_validation_errors_back_and_unit_conversion():
    with Served(HealthCoach()) as s:
        _, r, _ = s.call("POST", "/api/start", {})
        sid = r["sid"]
        s.call("POST", "/api/answer", {"sid": sid, "value": ["none"]})
        st, err, _ = s.call("POST", "/api/answer", {"sid": sid, "value": 200})
        assert st == 400 and "between" in err["error"]
        s.call("POST", "/api/answer", {"sid": sid, "value": 40})
        _, b, _ = s.call("POST", "/api/back", {"sid": sid})
        assert b["step"]["id"] == "age"
        sid2 = drive(s, {"symptoms": ["fever_chills"], "vitals_pick": ["temp_c"], "vital:temp_c": {"value": 101.5, "unit": "F"}})
        _, res, _ = s.call("POST", "/api/result", {"sid": sid2})
        assert res["tier"] in ("self_care", "see_doctor_soon", "urgent")


def test_result_requires_finished_questions_and_unknown_session_is_404():
    with Served(HealthCoach()) as s:
        _, r, _ = s.call("POST", "/api/start", {})
        assert s.call("POST", "/api/result", {"sid": r["sid"]})[0] == 409
        assert s.call("POST", "/api/answer", {"sid": "nope", "value": 1})[0] == 404
        assert s.call("POST", "/api/answer", {"sid": None, "value": 1})[0] == 404


def test_local_only_protections():
    with Served(HealthCoach()) as s:
        assert s.call("GET", "/api/health", headers={"Host": "evil.example"})[0] == 403                    # DNS rebinding
        assert s.call("GET", "/api/health", headers={"Host": f"127.0.0.1:{s.port + 1}"})[0] == 403
        assert s.call("POST", "/api/start", {}, headers={"Origin": "http://evil.example"})[0] == 403        # cross-site
        assert s.call("POST", "/api/start", raw="{}", headers={"Content-Type": "text/plain"})[0] == 415
        assert s.call("POST", "/api/start", raw="x" * (MAX_BODY + 1), headers={"Content-Type": "application/json"})[0] == 413
        assert s.call("POST", "/api/start", raw="not json", headers={"Content-Type": "application/json"})[0] == 400
        assert s.call("POST", "/api/start", raw="[1]", headers={"Content-Type": "application/json"})[0] == 400
        assert s.call("GET", "/static/../../etc/passwd")[0] == 404 and s.call("GET", "/nope")[0] == 404
        st, _, resp = s.call("GET", "/")
        h = {k.lower(): v for k, v in resp.getheaders()}
        assert st == 200 and "default-src 'none'" in h["content-security-policy"] and h["x-content-type-options"] == "nosniff"
        assert s.call("GET", "/api/health")[2].getheader("Cache-Control") == "no-store"


def test_server_binds_to_localhost_by_default():
    assert make_server(HealthCoach(), port=0).server_address[0] == "127.0.0.1"


def test_session_store_expiry_and_cap():
    now = [0.0]
    st = SessionStore(ttl=10, max_sessions=3, clock=lambda: now[0])
    a = st.create(IntakeSession())
    now[0] = 11
    try:
        st.get(a)
    except AppError as e:
        assert e.status == 404
    else:
        raise AssertionError("expired session should be gone")
    ids = []
    for _ in range(5):
        now[0] += 1
        ids.append(st.create(IntakeSession()))
    assert len(st) <= 3 and ids[-1] in st._s and ids[0] not in st._s       # least recently used evicted


def test_static_assets_are_self_contained_and_never_use_innerhtml():
    html = (STATIC / "index.html").read_text()
    js = (STATIC / "app.js").read_text()
    css = (STATIC / "app.css").read_text()
    assert not re.search(r'(src|href)="https?://', html) and "@import" not in css and not re.search(r"url\(\s*['\"]?https?:", css)
    assert not re.search(r"innerHTML|insertAdjacentHTML|document\.write|eval\(|new Function", js)
    assert "<script>" not in html                                  # no inline script: the CSP forbids it


def test_cli_help_needs_no_heavy_dependencies():
    r = subprocess.run([sys.executable, "-m", "healthcoach.app", "--help"], capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert r.returncode == 0 and "--gguf" in r.stdout and "--port" in r.stdout


def test_front_end_runs_end_to_end_in_a_fake_dom():
    node = shutil.which("node")
    if not node:
        return                                                      # skipped: node not installed
    with Served(HealthCoach()) as s:
        r = subprocess.run([node, str(ROOT / "tests/js/driver.js"), f"http://127.0.0.1:{s.port}", str(STATIC / "app.js")],
                           capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr + r.stdout
