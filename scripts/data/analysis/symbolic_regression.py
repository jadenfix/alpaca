"""Tiny genetic-programming symbolic regression engine.

Searches the space of small expression trees for formulas mapping
multivariate features X (T, F) to a target y (T,). Useful for
discovering alpha formulas that a human wouldn't have hand-coded.

Pure numpy + stdlib. Deterministic given a seed.

Tree node representation
------------------------
A `Node` is a tagged record:

    kind = "var"   -> payload is int (column index into X)
    kind = "const" -> payload is float
    kind = "op"    -> payload is str (op name); children is [c1] or [c1, c2]

Supported ops:
    binary: "+", "-", "*", "/", "max", "min"
    unary : "abs", "tanh", "sign", "log_abs"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import math

import numpy as np


# ───────────────────────────── data structures ──────────────────────────────


@dataclass
class Node:
    """A single node in an expression tree."""

    kind: str            # "var", "const", or "op"
    payload: object      # int (var idx), float (const), or str (op name)
    children: list = field(default_factory=list)  # 0, 1, or 2 child Node entries


# Op tables.
BINARY_OPS = ("+", "-", "*", "/", "max", "min")
UNARY_OPS = ("abs", "tanh", "sign", "log_abs")
ALL_OPS = BINARY_OPS + UNARY_OPS

_EPS = 1e-9
_CLIP = 1e6  # clamp evaluation outputs to avoid overflow blow-ups


# ───────────────────────────── evaluation ──────────────────────────────


def _safe(arr: np.ndarray) -> np.ndarray:
    """Replace non-finite entries with 0 and clip extreme magnitudes."""
    out = np.where(np.isfinite(arr), arr, 0.0)
    return np.clip(out, -_CLIP, _CLIP)


def evaluate(node: Node, X: np.ndarray) -> np.ndarray:
    """Evaluate the tree on data ``X`` of shape (T, F). Returns shape (T,).

    Uses safe ops (division-by-near-zero -> 0, log on |.| with eps,
    tanh saturates). On overflow / exception returns zeros.
    """
    T = X.shape[0]
    try:
        with np.errstate(all="ignore"):
            return _eval(node, X)
    except Exception:
        return np.zeros(T, dtype=np.float64)


def _eval(node: Node, X: np.ndarray) -> np.ndarray:
    T = X.shape[0]
    if node.kind == "var":
        idx = int(node.payload)
        # Guard against malformed trees.
        if idx < 0 or idx >= X.shape[1]:
            return np.zeros(T, dtype=np.float64)
        return _safe(X[:, idx].astype(np.float64, copy=False))
    if node.kind == "const":
        return np.full(T, float(node.payload), dtype=np.float64)
    if node.kind == "op":
        name = str(node.payload)
        if name in BINARY_OPS:
            a = _eval(node.children[0], X)
            b = _eval(node.children[1], X)
            if name == "+":
                out = a + b
            elif name == "-":
                out = a - b
            elif name == "*":
                out = a * b
            elif name == "/":
                # Safe division: denominators close to zero -> 0.
                denom = np.where(np.abs(b) < _EPS, 1.0, b)
                out = np.where(np.abs(b) < _EPS, 0.0, a / denom)
            elif name == "max":
                out = np.maximum(a, b)
            elif name == "min":
                out = np.minimum(a, b)
            else:
                out = np.zeros(T, dtype=np.float64)
            return _safe(out)
        if name in UNARY_OPS:
            a = _eval(node.children[0], X)
            if name == "abs":
                out = np.abs(a)
            elif name == "tanh":
                out = np.tanh(a)
            elif name == "sign":
                out = np.sign(a)
            elif name == "log_abs":
                out = np.log(np.abs(a) + _EPS)
            else:
                out = np.zeros(T, dtype=np.float64)
            return _safe(out)
    return np.zeros(T, dtype=np.float64)


# ───────────────────────────── tree utilities ──────────────────────────────


def tree_depth(node: Node) -> int:
    """Maximum depth of the tree (a single leaf has depth 1)."""
    if node.kind != "op" or not node.children:
        return 1
    return 1 + max(tree_depth(c) for c in node.children)


def tree_size(node: Node) -> int:
    """Number of nodes in the tree."""
    if node.kind != "op" or not node.children:
        return 1
    return 1 + sum(tree_size(c) for c in node.children)


def _copy(node: Node) -> Node:
    """Deep copy of a tree."""
    return Node(
        kind=node.kind,
        payload=node.payload,
        children=[_copy(c) for c in node.children],
    )


def _collect_nodes(node: Node) -> List[Node]:
    """List every node in the tree (post-order, includes the root)."""
    out: List[Node] = [node]
    for c in node.children:
        out.extend(_collect_nodes(c))
    return out


# ───────────────────────────── random generation ──────────────────────────────


def _random_terminal(rng: np.random.Generator, n_features: int) -> Node:
    # 70% var, 30% small constant.
    if rng.random() < 0.7 and n_features > 0:
        idx = int(rng.integers(0, n_features))
        return Node(kind="var", payload=idx, children=[])
    val = float(rng.uniform(-2.0, 2.0))
    return Node(kind="const", payload=val, children=[])


def random_tree(
    rng: np.random.Generator,
    n_features: int,
    max_depth: int = 3,
    p_terminal: float = 0.3,
) -> Node:
    """Recursively grow a random tree.

    At each step a terminal is generated with probability ``p_terminal``
    (or when ``max_depth`` is exhausted). Otherwise a random op is
    selected. ``max_depth`` is the maximum permitted depth from this
    node (a depth of 1 forces a terminal).
    """
    if max_depth <= 1 or rng.random() < p_terminal:
        return _random_terminal(rng, n_features)

    op_name = ALL_OPS[int(rng.integers(0, len(ALL_OPS)))]
    if op_name in BINARY_OPS:
        c1 = random_tree(rng, n_features, max_depth - 1, p_terminal)
        c2 = random_tree(rng, n_features, max_depth - 1, p_terminal)
        return Node(kind="op", payload=op_name, children=[c1, c2])
    # unary
    c1 = random_tree(rng, n_features, max_depth - 1, p_terminal)
    return Node(kind="op", payload=op_name, children=[c1])


# ───────────────────────────── crossover / mutation ──────────────────────────


def _replace_subtree(root: Node, target: Node, replacement: Node) -> Node:
    """Return a copy of ``root`` in which ``target`` (by identity) is
    swapped for a deep copy of ``replacement``."""
    if root is target:
        return _copy(replacement)
    new_children = [_replace_subtree(c, target, replacement) for c in root.children]
    return Node(kind=root.kind, payload=root.payload, children=new_children)


def crossover(rng: np.random.Generator, a: Node, b: Node) -> Node:
    """Subtree crossover. Returns a NEW tree that is ``a`` with a randomly
    chosen subtree replaced by a randomly chosen subtree of ``b``."""
    a_nodes = _collect_nodes(a)
    b_nodes = _collect_nodes(b)
    a_pick = a_nodes[int(rng.integers(0, len(a_nodes)))]
    b_pick = b_nodes[int(rng.integers(0, len(b_nodes)))]
    return _replace_subtree(a, a_pick, b_pick)


def mutate(
    rng: np.random.Generator,
    tree: Node,
    n_features: int,
    p_mutate: float = 0.1,
) -> Node:
    """With probability ``p_mutate`` replace a random subtree with a
    fresh random tree of comparable depth. Returns a NEW tree."""
    if rng.random() >= p_mutate:
        return _copy(tree)
    nodes = _collect_nodes(tree)
    pick = nodes[int(rng.integers(0, len(nodes)))]
    # Roughly match the depth of the subtree we're replacing.
    depth = max(1, tree_depth(pick))
    fresh = random_tree(rng, n_features, max_depth=depth, p_terminal=0.3)
    return _replace_subtree(tree, pick, fresh)


# ───────────────────────────── Spearman fitness ──────────────────────────────


def _ranks(x: np.ndarray) -> np.ndarray:
    """Simple ranks via double argsort (ties broken by original index)."""
    return np.argsort(np.argsort(x)).astype(np.float64)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation. Returns NaN on degenerate input."""
    if x.size != y.size or x.size < 2:
        return float("nan")
    rx = _ranks(x)
    ry = _ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float(np.dot(rx, rx)) * float(np.dot(ry, ry)))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(rx, ry) / denom)


def fitness(tree: Node, X: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation between tree predictions and y.

    Trees that produce NaN/inf or constant output return -1e9. Trees
    deeper than 5 are penalized by a 0.9**(depth-5) factor (parsimony).
    The sign of the correlation is preserved (we want positive corr).
    """
    pred = evaluate(tree, X)
    if not np.all(np.isfinite(pred)):
        return -1e9
    # Constant predictions have undefined Spearman -> treat as worst.
    if np.allclose(pred, pred[0]):
        return -1e9
    rho = _spearman(pred, y)
    if not math.isfinite(rho):
        return -1e9
    depth = tree_depth(tree)
    if depth > 5:
        rho *= (0.9 ** (depth - 5))
    return float(rho)


# ───────────────────────────── pretty printing ──────────────────────────────


def tree_to_str(node: Node) -> str:
    """Pretty-print, e.g. ``((X[2] * 0.5) + tanh(X[0]))``."""
    if node.kind == "var":
        return f"X[{int(node.payload)}]"
    if node.kind == "const":
        return f"{float(node.payload):.4g}"
    if node.kind == "op":
        name = str(node.payload)
        if name in BINARY_OPS:
            l = tree_to_str(node.children[0])
            r = tree_to_str(node.children[1])
            return f"({l} {name} {r})"
        if name in UNARY_OPS:
            c = tree_to_str(node.children[0])
            return f"{name}({c})"
    return "?"


# ───────────────────────────── evolution loop ──────────────────────────────


def _tournament_select(
    rng: np.random.Generator,
    pop: List[Node],
    fits: List[float],
    k: int = 3,
) -> Node:
    """Tournament selection of size k; returns a COPY of the winner."""
    n = len(pop)
    idxs = rng.integers(0, n, size=k)
    best_i = int(idxs[0])
    best_f = fits[best_i]
    for ii in idxs[1:]:
        i = int(ii)
        if fits[i] > best_f:
            best_f = fits[i]
            best_i = i
    return _copy(pop[best_i])


def evolve(
    X: np.ndarray,
    y: np.ndarray,
    n_features: int,
    population_size: int = 200,
    generations: int = 30,
    seed: int = 0,
) -> dict:
    """Run genetic programming and return best tree, fitness, formula,
    and per-generation best-fitness history.

    The reported ``history`` is monotone non-decreasing because we apply
    elitism (the best individual carries over) and report the running
    max across generations.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    # Initial population: random trees of varied depth.
    pop: List[Node] = []
    while len(pop) < population_size:
        d = int(rng.integers(2, 5))  # initial depths 2..4
        pop.append(random_tree(rng, n_features, max_depth=d, p_terminal=0.3))

    fits = [fitness(t, X, y) for t in pop]

    history: List[float] = []
    best_overall_fit = max(fits)
    best_overall_tree = _copy(pop[int(np.argmax(fits))])
    history.append(float(best_overall_fit))

    for _gen in range(1, generations):
        # Elitism: carry the best individual unchanged.
        new_pop: List[Node] = [_copy(best_overall_tree)]
        seen = {tree_to_str(new_pop[0])}

        while len(new_pop) < population_size:
            p1 = _tournament_select(rng, pop, fits, k=3)
            p2 = _tournament_select(rng, pop, fits, k=3)
            child = crossover(rng, p1, p2)
            child = mutate(rng, child, n_features, p_mutate=0.2)
            key = tree_to_str(child)
            if key in seen:
                # Diversity: backfill with a fresh random tree.
                d = int(rng.integers(2, 5))
                child = random_tree(rng, n_features, max_depth=d, p_terminal=0.3)
                key = tree_to_str(child)
                if key in seen:
                    continue
            seen.add(key)
            new_pop.append(child)

        pop = new_pop
        fits = [fitness(t, X, y) for t in pop]

        gen_best = max(fits)
        if gen_best > best_overall_fit:
            best_overall_fit = gen_best
            best_overall_tree = _copy(pop[int(np.argmax(fits))])
        # Report running max so history is monotone non-decreasing.
        history.append(float(best_overall_fit))

    return {
        "best_tree": best_overall_tree,
        "best_fitness": float(best_overall_fit),
        "best_formula": tree_to_str(best_overall_tree),
        "history": history,
    }
