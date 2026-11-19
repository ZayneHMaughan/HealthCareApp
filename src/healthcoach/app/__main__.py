"""Run the local web app:

  python -m healthcoach.app --index models/kb --gguf models/gguf/tuned-q4_k_m.gguf          # http://localhost:8888

Works without --gguf/--index (rules + standard messages only). The language model is only used when a knowledge base
is also loaded, because every explanation must be grounded in retrieved passages."""
from __future__ import annotations
import argparse
import json
import sys
import webbrowser
from pathlib import Path
from .core import HealthCoach
from .server import make_server


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m healthcoach.app")
    ap.add_argument("--host", default="127.0.0.1", help="keep this on localhost: the app handles health information")
    ap.add_argument("--port", type=int, default=8888, help="8888 is also Jupyter's default; change it if that is running")
    ap.add_argument("--index", help="knowledge-base directory built by healthcoach.retrieval.build")
    ap.add_argument("--embedder", default="none")
    ap.add_argument("--gguf", help="local language model (.gguf)")
    ap.add_argument("--risk-model", help="risk classifier bundle directory")
    ap.add_argument("--budget-tokens", type=int, help="retrieval budget; must match the one used to build training data")
    ap.add_argument("--train-stats", default="data/ft/stats.json")
    ap.add_argument("--open", action="store_true", help="open the page in your browser")
    a = ap.parse_args(argv)

    if a.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: binding to {a.host} exposes this app to your network. It has no login and handles health data.",
              file=sys.stderr)

    retriever = engine = llm = None
    info = {}
    budget = a.budget_tokens
    if budget is None and Path(a.train_stats).exists():
        budget = json.loads(Path(a.train_stats).read_text()).get("retrieval_budget_tokens")
        if budget:
            print(f"Retrieval budget {budget} tokens (from {a.train_stats})")
    if a.index:
        from ..retrieval import Index, Retriever
        from ..retrieval.embed import build_embedder
        retriever = Retriever(Index.load(a.index, build_embedder(a.embedder)), **({"budget_tokens": budget} if budget else {}))
        if retriever.index.stale():
            print("NOTE: the knowledge base is more than 180 days old.", file=sys.stderr)
    if a.risk_model:
        from ..risk import RiskEngine
        engine = RiskEngine(a.risk_model)       # fail loudly: a missing classifier would silently lower urgency estimates
    if a.gguf:
        from ..llm import LlamaCppLLM, default_gpu_layers
        llm = LlamaCppLLM(a.gguf, n_gpu_layers=default_gpu_layers())
        info["model_file"] = Path(a.gguf).name
        if retriever is None:
            print("WARNING: --gguf without --index: every answer will use the standard message (grounding is required).",
                  file=sys.stderr)

    coach = HealthCoach(engine=engine, llm=llm, retriever=retriever, info=info)
    srv = make_server(coach, a.host, a.port)
    url = f"http://localhost:{srv.server_address[1]}"
    print(f"Health Coach running at {url}   (Ctrl+C to stop)")
    if a.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
