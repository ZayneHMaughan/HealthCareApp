import json
from datetime import date
from pathlib import Path

from healthcoach.pipeline import run
from healthcoach.retrieval import Index, Retriever
from healthcoach.retrieval.embed import HashingEmbedder
from healthcoach.retrieval.ingest import (MPLUS_ATTRIBUTION, classify_kind, load_topic_map, parse_curated_dir,
                                          parse_medlineplus)
from healthcoach.retrieval.retriever import coverage_report
from healthcoach.retrieval.textproc import chunk_blocks, html_to_blocks
from healthcoach.schema import Intake, Tier

FIX = Path(__file__).parent / "fixtures"
TMAP = {"chest pain": ["chest_pain"], "heart attack": ["chest_pain"], "abdominal pain": ["abdominal_pain"],
        "headache": ["headache", "new_onset_headache"], "dehydration": ["dehydration"],
        "not in the file": ["cough"]}


def corpus():
    return parse_medlineplus(FIX / "mini_medlineplus.xml", TMAP)


def index(embedder=None):
    return Index.build(corpus()[0], embedder)


def mk(**kw):
    return Intake.from_dict({"age": 40, **kw})


def raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except ValueError:
        return True
    return False


# ---- ingestion -------------------------------------------------------------------------------------
def test_parse_filters_and_reports():
    chunks, rep = corpus()
    titles = {c.title for c in chunks}
    assert "Dolor abdominal" not in titles and rep["skipped_non_english"] == 1          # Spanish skipped
    assert "Encyclopedia Style Topic" not in titles and rep["skipped_copyright_marker"] == ["Encyclopedia Style Topic"]
    assert not any("THIRD PARTY" in c.text or "Some Org" in c.text for c in chunks)      # <site> never read
    assert rep["source_date"] == "2026-11-11"
    assert rep["map_titles_unmatched"] == ["not in the file"]                             # topic_map typos get reported
    assert all(c.attribution == MPLUS_ATTRIBUTION and c.source_date == "2026-11-11" for c in chunks)


def test_both_summary_encodings_parse_and_aliases_kept():
    by = {c.title: c for c in corpus()[0]}
    assert "Bellyache" in by["Abdominal Pain"].aliases           # child-element encoding
    assert "heartburn" in by["Chest Pain"].text                  # escaped-HTML encoding
    assert by["Chest Pain"].topics == ["chest_pain"]


def test_leadin_stays_with_its_list():
    blocks = html_to_blocks("<p>Short intro.</p><p>Symptoms may include</p><ul><li>One</li><li>Two</li></ul>")
    chunks = chunk_blocks(blocks, max_words=5)
    joined = next(c for c in chunks if "Symptoms may include" in c)
    assert "- One" in joined


def test_chunks_respect_max_words():
    long = "<p>" + " ".join(f"Sentence number {i} is here." for i in range(80)) + "</p>"
    assert all(len(c.split()) <= 170 for c in chunk_blocks(html_to_blocks(long), 160))


def test_kind_classification():
    assert classify_kind("Call 911 if this happens.") == "seek_care"
    assert classify_kind("See your health care provider if it persists.") == "seek_care"
    assert classify_kind("Most cases are mild.") == "overview"


def test_curated_ingest():
    cs = {c.text.split("\n")[0]: c for c in parse_curated_dir(FIX / "curated")}
    assert cs["Overview"].kind == "overview"
    assert cs["When to get emergency care"].kind == "seek_care"           # decided by the section heading
    assert cs["Overview"].topics == ["chest_pain"] and cs["Overview"].source_date == "2026-01-01"
    assert "- Sign one" in cs["When to get emergency care"].text          # list survives chunking


def test_curated_requires_license(tmp_path):
    (tmp_path / "x.md").write_text("---\ntitle: T\nsource: S\n---\nbody")
    assert raises(parse_curated_dir, tmp_path)


def test_default_topic_map_loads():
    m = load_topic_map(Path(__file__).parent.parent / "src/healthcoach/retrieval/data/topic_map.csv")
    assert "chest pain" in m and "chest_pain" in m["heart attack"]


# ---- index -----------------------------------------------------------------------------------------
def test_bm25_finds_by_alias_and_stem():
    hits = index().search("stomachache bellyache", k=3)
    assert hits[0].chunk.title == "Abdominal Pain"


def test_roundtrip_integrity_and_embedder_checks(tmp_path):
    idx = index(HashingEmbedder())
    idx.save(tmp_path)
    assert Index.load(tmp_path, HashingEmbedder()).vectors is not None
    assert Index.load(tmp_path).vectors is None                                   # BM25-only when no embedder given
    assert raises(Index.load, tmp_path, HashingEmbedder(dim=256))                 # embedder mismatch refused
    (tmp_path / "chunks.jsonl").write_text((tmp_path / "chunks.jsonl").read_text().replace("Chest", "Chesty", 1))
    assert raises(Index.load, tmp_path)                                           # tampering detected


def test_hybrid_search_uses_both_signals():
    hits = index(HashingEmbedder()).search("arm jaw sweating chest", k=3)
    assert hits[0].dense is not None and hits[0].bm25 > 0 and hits[0].chunk.title == "Chest Pain"


def test_staleness():
    idx = index()
    idx.save(Path("/tmp/kb_stale_test"))
    assert idx.stale(180, today=date(2027, 11, 11)) and not idx.stale(180, today=date(2026, 12, 1))


# ---- retriever -------------------------------------------------------------------------------------
def test_tagged_chunk_first_and_tier_changes_order():
    r = Retriever(index())
    calm = r.retrieve(mk(symptoms=["chest_pain"], modifiers=["sweating"]), Tier.SEE_DOCTOR_SOON)
    urgent = r.retrieve(mk(symptoms=["chest_pain"], modifiers=["sweating"]), Tier.URGENT)
    assert calm.passages[0].chunk.title == "Chest Pain"
    assert urgent.passages[0].chunk.kind == "seek_care"          # "when to get care" leads for urgent tiers


def test_round_robin_gives_each_symptom_a_passage():
    res = Retriever(index(), per_symptom=1).retrieve(mk(symptoms=["chest_pain", "abdominal_pain"]))
    assert {p.matched_symptom for p in res.passages} == {"chest_pain", "abdominal_pain"}


def test_weak_lexical_match_is_rejected_and_reported_uncovered():
    # "Knee Pain" and others contain the word "pain", but nothing is about testicle pain
    res = Retriever(index()).retrieve(mk(symptoms=["testicular_pain"]))
    assert res.passages == [] and res.uncovered == ["testicular_pain"]


def test_token_budget_is_respected():
    res = Retriever(index(), per_symptom=3, budget_tokens=60).retrieve(mk(symptoms=["chest_pain"]))
    assert 1 <= len(res.passages) <= 2


def test_coverage_report_lists_gaps():
    cov = coverage_report(Retriever(index()), ["chest_pain", "testicular_pain"])
    assert cov["chest_pain"] and cov["testicular_pain"] == []


def test_prompt_text_has_citable_ids():
    res = Retriever(index()).retrieve(mk(symptoms=["abdominal_pain"]))
    assert res.prompt_text().startswith("[S1] Abdominal Pain") and res.ids[0] == "S1"


# ---- pipeline grounding ----------------------------------------------------------------------------
def llm_reply(urgency="self_care", sources=("S1",), extra=""):
    return lambda prompt: json.dumps({"urgency": urgency, "considerations": [f"Common causes vary. {extra}"],
                                      "follow_up_questions": ["How long has it lasted?"],
                                      "when_to_seek_care": "If it gets worse.", "sources": list(sources)})


def test_grounded_answer_passes_and_carries_attribution():
    resp = run(mk(symptoms=["abdominal_pain"]), llm=llm_reply(), retriever=Retriever(index()))
    assert not resp.used_fallback and resp.sources[0]["title"] == "Abdominal Pain"
    assert resp.attribution == [MPLUS_ATTRIBUTION]


def test_no_retriever_means_no_llm_answer():
    resp = run(mk(symptoms=["abdominal_pain"]), llm=llm_reply())
    assert resp.used_fallback and "no grounding" in resp.violations[0]


def test_llm_never_called_when_nothing_relevant_retrieved():
    called = []
    resp = run(mk(symptoms=["testicular_pain"]), llm=lambda p: called.append(p) or "{}", retriever=Retriever(index()))
    assert not called and resp.used_fallback and resp.uncovered == ["testicular_pain"]


def test_invented_or_missing_citations_are_rejected():
    r = Retriever(index())
    bad = run(mk(symptoms=["abdominal_pain"]), llm=llm_reply(sources=("S9",)), retriever=r)
    none = run(mk(symptoms=["abdominal_pain"]), llm=llm_reply(sources=()), retriever=r)
    assert bad.used_fallback and "not given" in bad.violations[0]
    assert none.used_fallback and "no sources cited" in none.violations[0]


def test_llm_cannot_soften_tier_even_with_good_citations():
    resp = run(mk(symptoms=["abdominal_pain"], severity=8), llm=llm_reply("self_care"), retriever=Retriever(index()))
    assert resp.tier == Tier.URGENT and resp.used_fallback and any("softened" in v for v in resp.violations)


def test_malformed_llm_output_falls_back():
    resp = run(mk(symptoms=["abdominal_pain"]), llm=lambda p: "not json", retriever=Retriever(index()))
    assert resp.used_fallback and resp.message


def test_prompt_contains_references_and_reasons():
    seen = []
    run(mk(symptoms=["abdominal_pain"], severity=8), llm=lambda p: seen.append(p) or "{}", retriever=Retriever(index()))
    assert "[S1] Abdominal Pain" in seen[0] and "<assessment tier=\"urgent\">" in seen[0] and "Abdominal pain with severe" in seen[0]


def test_retrieval_harness_passes_good_cases_and_flags_bad_ones():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rre", Path(__file__).parent.parent / "evals/run_retrieval_evals.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    cases = [json.loads(l) for l in (FIX / "retrieval_cases.jsonl").read_text().splitlines()]
    rows = {r["id"]: r for r in mod.run_cases(Retriever(index()), cases)}
    assert not rows["chest"]["problems"] and not rows["urgent_first_kind"]["problems"] and not rows["gap_is_reported"]["problems"]
    assert rows["deliberately_wrong_expectation"]["problems"]


def test_build_resolves_directories_and_explains_missing_files(tmp_path):
    from healthcoach.retrieval.build import resolve_source
    (tmp_path / "mplus_topics_2026-01-05.xml").write_text("<x/>")
    (tmp_path / "mplus_topics_2026-11-11.xml").write_text("<x/>")
    assert resolve_source(str(tmp_path)).name == "mplus_topics_2026-11-11.xml"       # newest wins
    assert resolve_source(str(tmp_path / "mplus_topics_2026-01-05.xml")).name.endswith("01-05.xml")
    try:
        resolve_source(str(tmp_path / "nope.xml"))
    except SystemExit as e:
        assert "current directory" in str(e) and "data/medlineplus" in str(e)
    else:
        raise AssertionError("should have exited")
