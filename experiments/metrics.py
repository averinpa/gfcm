"""Evaluation metrics via bnmetrics. Skeleton-level scores compare an undirected
true skeleton against a learned one (symmetric adjacency), so bnmetrics' shd/recall/
precision/f1 act as skeleton metrics; false positives and per-edge-type recall are
derived for the discovery tables."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import bnmetrics


def _edge_pair(e):
    return tuple(e)


def skeleton_adjacency(edges, nodes):
    """Symmetric 0/1 adjacency for an undirected skeleton over ``nodes``."""
    idx = {n: i for i, n in enumerate(nodes)}
    p = len(nodes)
    A = np.zeros((p, p), dtype=int)
    for e in edges:
        a, b = _edge_pair(e)
        i, j = idx[a], idx[b]
        A[i, j] = A[j, i] = 1
    return A


def skeleton_scores(true_edges, learned_edges, nodes):
    """SHD / recall / precision / F1 (via bnmetrics) + false-positive edge count."""
    T = skeleton_adjacency(true_edges, nodes)
    L = skeleton_adjacency(learned_edges, nodes)
    fp = int(((L == 1) & (T == 0)).sum() // 2)
    return dict(
        shd=int(bnmetrics.shd(T, L)),
        recall=float(bnmetrics.recall(T, L)),
        precision=float(bnmetrics.precision(T, L)),
        f1=float(bnmetrics.f1(T, L)),
        fp_edges=fp,
    )


def per_type_recall(etype, learned_edges):
    """Recall split by the edge-type map ``etype`` (frozenset edge -> 'linear'/'z2'/'scale')."""
    learned = set(learned_edges)
    by = defaultdict(list)
    for e, t in etype.items():
        by[t].append(e)
    return {t: (float(np.mean([1.0 if e in learned else 0.0 for e in es])) if es else float("nan"))
            for t, es in by.items()}
