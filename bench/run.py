"""Benchmark app ranking: accuracy (top-1, top-3, MRR) and latency over bench/cases.json.

    uv run python bench/run.py [--configs tournament:4,choice] [--styles name,short,full] [--per-option 24,48]
                               [--shuffles 0] [--baseline] [--misses]

The model comes from the same DECISION_SEARCHBAR_* settings as the app. The cases name the author's installed
apps; edit them for yours.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import statistics
import time
from pathlib import Path

from decision_searchbar import open_model
from decision_searchbar.apps import installed_apps
from decision_searchbar.rank import rank

CASES = json.loads((Path(__file__).parent / "cases.json").read_text())


def keyword_rank(query, apps):
    """Baseline: count query words found in the app's name, generic name, description and keywords."""
    words = set(re.findall(r"\w+", query.lower()))
    def score(app):
        text = set(re.findall(r"\w+", " ".join([app.name, app.generic, app.description, *app.keywords]).lower()))
        return len(words & text)
    return sorted(((a, score(a)) for a in apps), key=lambda p: p[1], reverse=True)


def evaluate(ranker, apps, label):
    top1 = top3 = rr = 0.0
    times, misses = [], []
    for case in CASES:
        start = time.perf_counter()
        ranked = ranker(case["query"], apps)
        times.append((time.perf_counter() - start) * 1000)
        ids = [a.id for a, _ in ranked]
        pos = min(ids.index(i) for i in case["ok"])
        top1 += pos == 0
        top3 += pos < 3
        rr += 1 / (pos + 1)
        if pos > 0:
            misses.append(f'    {case["query"]!r}: got {ranked[0][0].name!r}, wanted at #{pos + 1}')
    n = len(CASES)
    times.sort()
    print(f"{label:<34} top1 {top1 / n:5.0%}  top3 {top3 / n:5.0%}  MRR {rr / n:.3f}  "
          f"median {statistics.median(times):6.1f} ms  p95 {times[int(0.95 * (n - 1))]:6.1f} ms")
    return misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--styles", default="full")
    ap.add_argument("--configs", default="tournament:4", help="comma list of mode[:orders], e.g. choice:4,tournament,noul")
    ap.add_argument("--per-option", default="24")
    ap.add_argument("--group", default="8")
    ap.add_argument("--advance", default="1")
    ap.add_argument("--shuffles", type=int, default=0, help="extra runs with the app order shuffled")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--misses", action="store_true")
    args = ap.parse_args()

    apps = installed_apps()
    missing = sorted({i for case in CASES for i in case["ok"]} - {a.id for a in apps})
    if missing:
        raise SystemExit(f"bench/cases.json expects apps you don't have: {', '.join(missing)}")
    if args.baseline:
        evaluate(keyword_rank, apps, "keyword overlap baseline")
    model = open_model()
    model.ready()
    rank(model, "warm up", apps)
    print(f"{len(apps)} apps, {len(CASES)} cases, on {model.device}")
    for config in args.configs.split(","):
      mode, _, orders = config.partition(":")
      orders = int(orders or 1)
      for style in args.styles.split(","):
        for per, group, advance in itertools.product(*(map(int, s.split(",")) for s in
                                                       (args.per_option, args.group, args.advance))):
            app_orders = [apps] + [random.Random(s).sample(apps, len(apps)) for s in range(1, args.shuffles + 1)]
            for i, order in enumerate(app_orders):
                label = f"{mode}x{orders} {style} g{group}a{advance}" + (f" shuffle={i}" if i else "")
                misses = evaluate(lambda q, a: rank(model, q, a, style, mode, orders, group, advance, per), order, label)
                if args.misses:
                    print("\n".join(misses))


if __name__ == "__main__":
    main()
