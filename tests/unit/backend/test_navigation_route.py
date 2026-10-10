"""A* routing and lane-centre route geometry (right-hand traffic, ENU)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.navigation.astar import RouteUnreachable, astar
from app.navigation.route import LANE_OFFSET_M, RoadSurface, build_route, plan_route
from app.simulators.lite.graph import JUNCTION_HALF_SIZE_M, build_town_graph, junction_nodes


@pytest.fixture(scope="module")
def graph():  # type: ignore[no-untyped-def]
    return build_town_graph()


def test_astar_shortest_paths(graph) -> None:  # type: ignore[no-untyped-def]
    assert astar(graph, "A", "C") == ["A", "B", "C"]
    assert astar(graph, "A", "D") == ["A", "B", "D"]
    assert astar(graph, "G", "I") in (["G", "C", "B", "D", "H", "I"], ["G", "C", "E", "D", "H", "I"])
    path = astar(graph, "F", "A")
    assert path[0] == "F" and path[-1] == "A"
    # deterministic tie-breaking
    assert astar(graph, "A", "E") == astar(graph, "A", "E")


def test_astar_matches_bruteforce_lengths(graph) -> None:  # type: ignore[no-untyped-def]
    import itertools

    ids = [n.id for n in graph.nodes if n.id != "Q"]
    edges = {(e.a, e.b): e.length_m for e in graph.edges} | {(e.b, e.a): e.length_m for e in graph.edges}
    dist = {(a, b): (0.0 if a == b else edges.get((a, b), math.inf)) for a in ids for b in ids}
    for k, i, j in itertools.product(ids, ids, ids):  # Floyd-Warshall reference
        dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
    for a, b in itertools.permutations(ids, 2):
        p = astar(graph, a, b)
        length = sum(edges[(u, v)] for u, v in zip(p[:-1], p[1:], strict=True))
        assert length == pytest.approx(dist[a, b])


@pytest.mark.parametrize(
    ("start", "dest", "code"),
    [
        ("A", "Q", "destination_unreachable"),
        ("Q", "A", "destination_unreachable"),
        ("A", "Z", "unknown_node"),
        ("A", "A", "same_start_destination"),
    ],
)
def test_route_rejections(graph, start: str, dest: str, code: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(RouteUnreachable) as exc:
        plan_route(graph, start, dest)
    assert exc.value.code == code
    assert isinstance(exc.value, ValueError)


def test_lane_centre_is_right_of_travel_direction(graph) -> None:  # type: ignore[no-untyped-def]
    east = build_route(graph, ["A", "B"])  # heading east: right is -y
    assert np.allclose(east.path[:, 1], -LANE_OFFSET_M, atol=1e-6)
    assert east.path[0, 0] == pytest.approx(0.0) and east.path[-1, 0] == pytest.approx(60.0)
    west = build_route(graph, ["B", "A"])  # heading west: right is +y
    assert np.allclose(west.path[:, 1], LANE_OFFSET_M, atol=1e-6)
    north = build_route(graph, ["B", "D"])  # heading north: right is +x
    assert np.allclose(north.path[:, 0], 60.0 + LANE_OFFSET_M, atol=1e-6)


def test_waypoint_spacing_and_curvature_sign(graph) -> None:  # type: ignore[no-untyped-def]
    left_turn = plan_route(graph, "A", "D")  # east then north at B: left (CCW) turn
    ds = np.diff(left_turn.s)
    assert ds.max() < 0.55 and ds.min() > 0.3
    assert left_turn.curvature.max() > 0.05 and left_turn.curvature.min() > -0.01
    right_turn = plan_route(graph, "E", "I")  # west then north at D: right (CW) turn
    assert right_turn.curvature.min() < -0.05
    # right-turn lane radius stays above the vehicle's minimum turning radius (2.6 / tan 0.5 = 4.76 m)
    assert 1.0 / abs(right_turn.curvature.min()) > 4.9
    # the lane path is continuous (no gaps at junction fillets)
    for r in (left_turn, right_turn):
        assert np.linalg.norm(np.diff(r.path, axis=0), axis=1).max() < 0.6


def test_projection_and_progress(graph) -> None:  # type: ignore[no-untyped-def]
    r = plan_route(graph, "A", "C")
    p = r.project(30.0, -1.75 + 0.4)
    assert p.s_m == pytest.approx(30.0, abs=0.05)
    assert p.lateral_m == pytest.approx(0.4, abs=1e-6), "left of the travel direction is positive"
    assert r.progress(p.s_m) == pytest.approx(30.0 / r.length_m, abs=1e-3)
    assert r.project(30.0, -1.75 - 0.3, hint_index=p.index).lateral_m == pytest.approx(-0.3, abs=1e-6)
    assert r.to_dict()["node_ids"] == ["A", "B", "C"]
    assert r.to_dict()["length_m"] == pytest.approx(130.0, abs=0.01)


def test_road_surface_distance(graph) -> None:  # type: ignore[no-untyped-def]
    road = RoadSurface(graph, JUNCTION_HALF_SIZE_M, junction_nodes(graph))
    assert road.distance_to_centreline(30.0, -1.75) == pytest.approx(1.75)
    assert road.distance_to_centreline(30.0, -20.0) == pytest.approx(20.0)
    assert road.distance_to_centreline(66.0, 6.0) == 0.0  # inside junction box B
