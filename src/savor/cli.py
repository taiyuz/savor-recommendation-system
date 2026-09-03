"""CLI: recommend, evaluate, serve."""

from __future__ import annotations

import argparse
import json
import sys

from savor.config import DEFAULT_K
from savor.data.loader import load_catalog
from savor.evaluate import run_offline_eval
from savor.pipeline import Recommender


def _print_recs(user_id: str, k: int) -> int:
    catalog = load_catalog()
    if user_id not in set(catalog.user_ids()):
        print(f"unknown user_id: {user_id}", file=sys.stderr)
        return 1
    rec = Recommender().fit(catalog)
    strategy = "cold_start_popularity" if rec.generator.is_cold(user_id) else "two_stage"
    items = rec.recommend(user_id, k=k)
    payload = {
        "user_id": user_id,
        "k": k,
        "strategy": strategy,
        "items": [
            {
                "item_id": row.item_id,
                "name": row.name,
                "cuisine": row.cuisine,
                "neighborhood": row.neighborhood,
                "price_tier": row.price_tier,
                "score": round(row.score, 6),
                "sources": list(row.sources),
            }
            for row in items
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0


def _print_eval(k: int) -> int:
    report = run_offline_eval(k=k)
    payload = {
        "dataset": report.dataset,
        "k": report.k,
        "n_eval_users": report.n_eval_users,
        "two_stage": {
            "recall": round(report.recall, 4),
            "ndcg": round(report.ndcg, 4),
            "mrr": round(report.mrr, 4),
            "coverage": round(report.coverage, 4),
        },
        "popularity": {
            "recall": round(report.popularity_recall, 4),
            "ndcg": round(report.popularity_ndcg, 4),
            "mrr": round(report.popularity_mrr, 4),
            "coverage": round(report.popularity_coverage, 4),
        },
        "item_cold_start": {
            "n_cold_items": report.n_cold_items,
            "label_fraction": round(report.cold_label_fraction, 4),
            "two_stage_shown_coverage": round(report.cold_item_coverage, 4),
            "popularity_shown_coverage": round(report.popularity_cold_item_coverage, 4),
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="savor", description="Savor recommendation backend")
    sub = parser.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("recommend", help="print a ranked restaurant list for one user")
    rec.add_argument("user_id")
    rec.add_argument("-k", type=int, default=DEFAULT_K)

    ev = sub.add_parser("evaluate", help="run temporal-split metrics on the synthetic sample")
    ev.add_argument("-k", type=int, default=DEFAULT_K)

    serve = sub.add_parser("serve", help="run the FastAPI app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    if args.cmd == "recommend":
        return _print_recs(args.user_id, args.k)
    if args.cmd == "evaluate":
        return _print_eval(args.k)
    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("savor.api.app:app", host=args.host, port=args.port, reload=False)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
