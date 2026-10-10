"""LiteRoad static town road graph (allowed runtime navigation input).

Frame: ENU metres, x east, y north. Edges are two-way roads 7 m wide (2 x 3.5 m lanes, right-hand traffic).
Curved edges are polylines sampled about every 2 m so that their markings and lane-centre offsets stay smooth.
Every degree-2 node joins its edges tangentially; corners only occur inside junction boxes (degree >= 3).
Node ``Q`` (closed depot) is deliberately not connected to anything: routes to it are rejected.
"""

from __future__ import annotations

import math
from functools import lru_cache

from app.simulators.base import GraphEdge, GraphNode, RoadGraph

LANE_WIDTH_M = 3.5
ROAD_HALF_WIDTH_M = 2 * LANE_WIDTH_M / 2  # 3.5 m: white edge lines sit here
JUNCTION_HALF_SIZE_M = 9.0  # square asphalt box without markings around nodes of degree >= 3
DEFAULT_START = "A"

_NODES: tuple[GraphNode, ...] = (
    GraphNode("A", 0.0, 0.0, "Bosh darvoza", "Main gate"),
    GraphNode("B", 60.0, 0.0, "Bozor chorrahasi", "Market junction"),
    GraphNode("C", 130.0, 0.0, "Maktab chorrahasi", "School junction"),
    GraphNode("D", 60.0, 70.0, "Park chorrahasi", "Park junction"),
    GraphNode("E", 130.0, 70.0, "Shifoxona chorrahasi", "Hospital junction"),
    GraphNode("F", 130.0, 130.0, "Stadion", "Stadium"),
    GraphNode("G", 185.0, 55.0, "Sharqiy ombor", "East warehouse"),
    GraphNode("H", 10.0, 120.0, "Ko‘l bo‘yi", "Lakeside"),
    GraphNode("I", -20.0, 150.0, "Bog‘", "Garden"),
    GraphNode("Q", 215.0, 140.0, "Yopiq depo (yo‘l yo‘q)", "Closed depot (no road)"),
)


def _arc(cx: float, cy: float, r: float, a0: float, a1: float, step_m: float = 2.0) -> list[tuple[float, float]]:
    """Points on a circle from polar angle a0 to a1 (radians), inclusive, about every ``step_m`` metres."""
    n = max(2, math.ceil(abs(a1 - a0) * r / step_m) + 1)
    return [
        (cx + r * math.cos(a0 + (a1 - a0) * i / (n - 1)), cy + r * math.sin(a0 + (a1 - a0) * i / (n - 1)))
        for i in range(n)
    ]


def _dedupe(points: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    out: list[tuple[float, float]] = []
    for p in points:
        q = (round(p[0], 4), round(p[1], 4))
        if not out or math.hypot(q[0] - out[-1][0], q[1] - out[-1][1]) > 1e-6:
            out.append(q)
    return tuple(out)


def _straight(a: GraphNode, b: GraphNode) -> tuple[tuple[float, float], ...]:
    return ((a.x, a.y), (b.x, b.y))


def _edges(nodes: dict[str, GraphNode]) -> tuple[GraphEdge, ...]:
    n = nodes
    # C -> G: 15 m east, 90 deg left arc of radius 40 m (centre (145, 40)), 15 m north.
    c_to_g = [(130.0, 0.0), (145.0, 0.0), *_arc(145.0, 40.0, 40.0, -math.pi / 2, 0.0), (185.0, 55.0)]
    # D -> H: 10 m north, 90 deg left arc of radius 40 m (centre (20, 80)), 10 m west.
    d_to_h = [(60.0, 70.0), (60.0, 80.0), *_arc(20.0, 80.0, 40.0, 0.0, math.pi / 2), (10.0, 120.0)]
    # H -> I: 90 deg right arc of radius 30 m (centre (10, 150)) from heading west to heading north.
    h_to_i = _arc(10.0, 150.0, 30.0, -math.pi / 2, -math.pi)
    return (
        GraphEdge("AB", "A", "B", _straight(n["A"], n["B"])),
        GraphEdge("BC", "B", "C", _straight(n["B"], n["C"])),
        GraphEdge("BD", "B", "D", _straight(n["B"], n["D"])),
        GraphEdge("CE", "C", "E", _straight(n["C"], n["E"])),
        GraphEdge("DE", "D", "E", _straight(n["D"], n["E"])),
        GraphEdge("EF", "E", "F", _straight(n["E"], n["F"])),
        GraphEdge("CG", "C", "G", _dedupe(c_to_g)),
        GraphEdge("DH", "D", "H", _dedupe(d_to_h)),
        GraphEdge("HI", "H", "I", _dedupe(h_to_i)),
    )


@lru_cache(maxsize=1)
def build_town_graph() -> RoadGraph:
    """The static LiteRoad town. Cached: the topology never changes at runtime."""
    nodes = {node.id: node for node in _NODES}
    edges = _edges(nodes)
    for e in edges:  # polylines must start/end exactly on their nodes
        assert e.polyline[0] == (nodes[e.a].x, nodes[e.a].y), e.id
        assert e.polyline[-1] == (nodes[e.b].x, nodes[e.b].y), e.id
    return RoadGraph(nodes=_NODES, edges=edges, lane_width_m=LANE_WIDTH_M)


def node_degrees(graph: RoadGraph) -> dict[str, int]:
    deg = {n.id: 0 for n in graph.nodes}
    for e in graph.edges:
        deg[e.a] += 1
        deg[e.b] += 1
    return deg


def junction_nodes(graph: RoadGraph) -> list[str]:
    """Nodes rendered as unmarked junction boxes (degree >= 3)."""
    return [nid for nid, d in node_degrees(graph).items() if d >= 3]


def reachable_from(graph: RoadGraph, start: str) -> frozenset[str]:
    """Connected component of ``start`` (breadth-first over two-way edges)."""
    adj: dict[str, set[str]] = {n.id: set() for n in graph.nodes}
    for e in graph.edges:
        adj[e.a].add(e.b)
        adj[e.b].add(e.a)
    if start not in adj:
        return frozenset()
    seen = {start}
    frontier = [start]
    while frontier:
        cur = frontier.pop()
        for nxt in sorted(adj[cur]):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return frozenset(seen)
