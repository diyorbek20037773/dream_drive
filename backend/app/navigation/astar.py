"""A* route search over the static road graph (edge length weights, Euclidean heuristic)."""

from __future__ import annotations

import heapq
import math

from app.simulators.base import RoadGraph


class RouteUnreachable(ValueError):  # noqa: N818 - name fixed by docs/IMPLEMENTATION_SPEC.md 4.1
    """No drivable route exists. ``code`` is a stable machine code for API error mapping."""

    def __init__(self, message: str, code: str = "destination_unreachable") -> None:
        super().__init__(message)
        self.code = code


def adjacency(graph: RoadGraph) -> dict[str, list[tuple[str, float]]]:
    """Two-way adjacency with edge polyline lengths as weights (sorted for deterministic expansion)."""
    adj: dict[str, list[tuple[str, float]]] = {n.id: [] for n in graph.nodes}
    for e in graph.edges:
        adj[e.a].append((e.b, e.length_m))
        adj[e.b].append((e.a, e.length_m))
    for k in adj:
        adj[k].sort()
    return adj


def astar(graph: RoadGraph, start: str, goal: str) -> list[str]:
    """Shortest node sequence from ``start`` to ``goal``.

    Raises RouteUnreachable(code=unknown_node | same_start_destination | destination_unreachable).
    Ties are broken by node id so the result is deterministic.
    """
    ids = {n.id for n in graph.nodes}
    for nid in (start, goal):
        if nid not in ids:
            raise RouteUnreachable(f"unknown node {nid!r}", code="unknown_node")
    if start == goal:
        raise RouteUnreachable("start and destination are the same node", code="same_start_destination")
    pos = {n.id: (n.x, n.y) for n in graph.nodes}
    gx, gy = pos[goal]

    def h(nid: str) -> float:
        x, y = pos[nid]
        return math.hypot(gx - x, gy - y)

    adj = adjacency(graph)
    g_cost: dict[str, float] = {start: 0.0}
    parent: dict[str, str] = {}
    open_heap: list[tuple[float, float, str]] = [(h(start), 0.0, start)]
    closed: set[str] = set()
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return path[::-1]
        closed.add(cur)
        for nxt, w in adj[cur]:
            ng = g + w
            if nxt not in closed and ng < g_cost.get(nxt, math.inf) - 1e-9:
                g_cost[nxt] = ng
                parent[nxt] = cur
                heapq.heappush(open_heap, (ng + h(nxt), ng, nxt))
    raise RouteUnreachable(f"no road connects {start!r} to {goal!r}", code="destination_unreachable")
