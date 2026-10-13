"""Rank apps for a free-text request with d1.

Modes:
- `choice`: one choice question whose options are all the apps, averaged over `orders` shuffled option orders
  (d1 is sensitive to option order with ~90 options).
- `tournament`: choice questions over groups of `group` apps, then one choice over each group's top `advance`.
- `noul`: one yes/no question per app, ranked by P(yes).
"""

from __future__ import annotations

import random

from .apps import App
from .model import D1

INSTRUCTIONS = "Which application should be opened to do what the user asks?"


def keys(apps: list[App]) -> list[str]:
    out, seen = [], set()
    for app in apps:
        key = app.name if app.name not in seen else f"{app.name} ({app.id.removesuffix('.desktop')})"
        seen.add(key)
        out.append(key)
    return out


def choice(apps: list[App], style: str = "full") -> dict:
    return {"type": "choice", "instructions": INSTRUCTIONS,
            "criteria": {k: a.describe(style) for k, a in zip(keys(apps), apps)}}


def _choice_scores(model, query, apps, style, orders, per_option):
    """Mean probability per app over `orders` option orders (the first is the given order)."""
    perms = [list(range(len(apps)))] + [random.Random(s).sample(range(len(apps)), len(apps))
                                        for s in range(1, orders)]
    probs = model.probabilities(query, [choice([apps[i] for i in p], style) for p in perms], per_option=per_option)
    scores = [0.0] * len(apps)
    for p, row in zip(perms, probs):
        for i, prob in zip(p, row):
            scores[i] += prob / len(perms)
    return scores


def rank(model: D1, query: str, apps: list[App], style: str = "full", mode: str = "tournament", orders: int = 4,
         group: int = 8, advance: int = 1, per_option: int = 24) -> list[tuple[App, float]]:
    """Apps best first, each with its probability in the round that placed it."""
    if mode == "choice":
        scores = _choice_scores(model, query, apps, style, orders, per_option)
    elif mode == "tournament":
        n = -(-len(apps) // group)  # groups of at most `group`, sizes differing by at most one
        groups = [list(range(len(apps)))[i::n] for i in range(n)]
        rows = model.probabilities(query, [choice([apps[i] for i in g], style) for g in groups],
                                   per_option=per_option)
        first = {}
        for g, row in zip(groups, rows):
            first.update(zip(g, row))
        finalists = [i for g, row in zip(groups, rows)
                     for _, i in sorted(zip(row, g), reverse=True)[:advance]]
        final = dict(zip(finalists, _choice_scores(model, query, [apps[i] for i in finalists], style, orders,
                                                   per_option)))
        # finalists by their final-round probability, then everyone else by their group probability
        order = sorted(range(len(apps)), key=lambda i: (i in final, final.get(i, first[i])), reverse=True)
        return [(apps[i], final.get(i, first[i])) for i in order]
    elif mode == "noul":
        questions = [{"type": "noul", "instructions": f"Is the application {k} ({a.describe(style)}) the one to "
                      "open to do what the user asks?"} for k, a in zip(keys(apps), apps)]
        scores = [p[0] for p in model.probabilities(query, questions, per_option=per_option)]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return sorted(zip(apps, scores), key=lambda pair: pair[1], reverse=True)
