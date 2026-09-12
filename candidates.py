"""Phase 5: deterministic K-shortest simple candidate routes."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import networkx as nx

from algorithms.baseline import disrupted_edge_set, load_environment, shortest_path_for_trip
from env import DEFAULT_NUM_TRIPS, Environment
from flow import FREE_FLOW_ATTR, route_edges

OUTPUT_PATH = Path("output") / "candidates.json"
ALGORITHM_NAME = "k_shortest_candidates"
DEFAULT_K = 5
WEIGHT_ATTR = FREE_FLOW_ATTR


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def free_flow_cost(graph: nx.DiGraph, route: list[int]) -> float:
    return float(sum(graph[u][v][WEIGHT_ATTR] for u, v in route_edges(route)))


def yen_k_shortest_paths(
    graph: nx.DiGraph,
    origin: int,
    destination: int,
    k: int = DEFAULT_K,
) -> list[list[int]]:
    """Yen's algorithm for K simple shortest paths using free-flow weights."""
    first = shortest_path_for_trip(graph, origin, destination)
    accepted: list[list[int]] = [first]
    potentials: list[tuple[float, list[int]]] = []

    def already_known(route: list[int]) -> bool:
        return any(route == path for path in accepted) or any(
            route == path for _, path in potentials
        )

    for _ in range(1, k):
        previous = accepted[-1]
        for spur_index in range(len(previous) - 1):
            spur_node = previous[spur_index]
            root_path = previous[: spur_index + 1]
            work = graph.copy()
            for path in accepted:
                if path[: spur_index + 1] == root_path and len(path) > spur_index + 1:
                    u, v = path[spur_index], path[spur_index + 1]
                    if work.has_edge(u, v):
                        work.remove_edge(u, v)
            for node in root_path[:-1]:
                if work.has_node(node):
                    work.remove_node(node)
            if not work.has_node(spur_node) or not work.has_node(destination):
                continue
            try:
                spur_path = nx.dijkstra_path(
                    work, spur_node, destination, weight=WEIGHT_ATTR
                )
            except nx.NetworkXNoPath:
                continue
            candidate = root_path[:-1] + spur_path
            if len(candidate) != len(set(candidate)):
                continue
            if already_known(candidate):
                continue
            potentials.append((free_flow_cost(graph, candidate), candidate))

        if not potentials:
            break
        potentials.sort(key=lambda item: (item[0], item[1]))
        _, best = potentials.pop(0)
        accepted.append(best)

    return accepted


def _unique_sorted_candidates(
    graph: nx.DiGraph,
    baseline_route: list[int],
    generated: list[list[int]],
    k: int,
) -> list[list[int]]:
    unique: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for route in [baseline_route, *generated]:
        key = tuple(route)
        if key in seen:
            continue
        seen.add(key)
        unique.append(route)

    rest = [route for route in unique if route != baseline_route]
    rest.sort(key=lambda route: (free_flow_cost(graph, route), route))
    ordered = [baseline_route, *rest]
    return ordered[:k]


def generate_candidates_for_trip(
    graph: nx.DiGraph,
    origin: int,
    destination: int,
    k: int = DEFAULT_K,
) -> tuple[list[list[int]], list[float]]:
    baseline_route = shortest_path_for_trip(graph, origin, destination)
    generated = yen_k_shortest_paths(graph, origin, destination, k=k)
    routes = _unique_sorted_candidates(graph, baseline_route, generated, k)
    costs = [free_flow_cost(graph, route) for route in routes]
    return routes, costs


def validate_candidates_for_trip(
    graph: nx.DiGraph,
    origin: int,
    destination: int,
    routes: list[list[int]],
    costs: list[float],
    k: int,
) -> None:
    blocked = disrupted_edge_set()
    _require(1 <= len(routes) <= k, f"Candidate count {len(routes)} is outside [1, {k}]")
    _require(len(routes) == len(costs), "Candidate routes and costs have different lengths")
    _require(routes[0][0] == origin, "First candidate does not start at origin")
    _require(routes[0][-1] == destination, "First candidate does not end at destination")

    expected_first = shortest_path_for_trip(graph, origin, destination)
    _require(
        routes[0] == expected_first,
        f"Candidate 0 {routes[0]} does not match free-flow shortest path {expected_first}",
    )

    seen: set[tuple[int, ...]] = set()
    for index, (route, cost) in enumerate(zip(routes, costs)):
        _require(len(route) >= 2, f"Candidate {index} is too short: {route}")
        _require(route[0] == origin, f"Candidate {index} starts at {route[0]}, expected {origin}")
        _require(
            route[-1] == destination,
            f"Candidate {index} ends at {route[-1]}, expected {destination}",
        )
        _require(len(route) == len(set(route)), f"Candidate {index} is not simple: {route}")
        key = tuple(route)
        _require(key not in seen, f"Duplicate candidate route: {route}")
        seen.add(key)
        for u, v in route_edges(route):
            _require(graph.has_edge(u, v), f"Candidate {index} uses missing edge {u}->{v}")
            _require((u, v) not in blocked, f"Candidate {index} uses disrupted edge {u}->{v}")
        expected_cost = free_flow_cost(graph, route)
        _require(cost == expected_cost, f"Candidate {index} cost {cost} != {expected_cost}")

    for left, right in zip(costs, costs[1:]):
        _require(left <= right, f"Candidates are not sorted by free-flow cost: {costs}")

    rest = routes[1:]
    rest_costs = costs[1:]
    for (left_cost, left_route), (right_cost, right_route) in zip(
        zip(rest_costs, rest), zip(rest_costs[1:], rest[1:])
    ):
        if left_cost == right_cost:
            _require(
                left_route <= right_route,
                "Equal-cost alternatives are not in deterministic node-sequence order",
            )


def generate_candidates(env: Environment, k: int = DEFAULT_K) -> list[dict]:
    _require(len(env.trips) == DEFAULT_NUM_TRIPS, f"Expected {DEFAULT_NUM_TRIPS} trips")
    records: list[dict] = []
    for trip_index, (origin, destination) in enumerate(env.trips):
        routes, costs = generate_candidates_for_trip(env.graph, origin, destination, k=k)
        validate_candidates_for_trip(env.graph, origin, destination, routes, costs, k)
        records.append(
            {
                "trip_index": trip_index,
                "origin": int(origin),
                "destination": int(destination),
                "candidates": [[int(node) for node in route] for route in routes],
                "candidate_costs": [float(cost) for cost in costs],
            }
        )
    return records


def build_output(env: Environment, records: list[dict], k: int = DEFAULT_K) -> dict:
    return {
        "algorithm": ALGORITHM_NAME,
        "seed": env.seed,
        "k": k,
        "number_of_nodes": env.graph.number_of_nodes(),
        "number_of_directed_edges": env.graph.number_of_edges(),
        "number_of_trips": len(env.trips),
        "candidate_routes": records,
    }


def write_candidates_json(payload: dict, path: Path = OUTPUT_PATH) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        path.unlink(missing_ok=True)
        raise ValueError("output/candidates.json is not a JSON object")
    return loaded


def compare_candidate0_to_baseline(records: list[dict], baseline_path: Path = Path("output") / "baseline.json") -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_routes = baseline["routes"]
    _require(len(records) == len(baseline_routes), "Candidate trip count does not match baseline")
    for record, baseline_route in zip(records, baseline_routes):
        first = record["candidates"][0]
        _require(
            first == baseline_route,
            (
                f"Trip {record['trip_index']} candidate 0 {first} "
                f"does not match baseline route {baseline_route}"
            ),
        )


def print_summary(payload: dict) -> None:
    counts = [len(record["candidates"]) for record in payload["candidate_routes"]]
    histogram = Counter(counts)
    print("algorithm:", payload["algorithm"])
    print("seed:", payload["seed"])
    print("k:", payload["k"])
    print("trips:", payload["number_of_trips"])
    print("total_candidates:", sum(counts))
    print("minimum_candidates_per_trip:", min(counts))
    print("maximum_candidates_per_trip:", max(counts))
    print("average_candidates_per_trip:", sum(counts) / len(counts))
    for n in range(5, 0, -1):
        print(f"trips_with_{n}_candidates:", histogram.get(n, 0))
    print("wrote:", OUTPUT_PATH.as_posix())


def main() -> None:
    env = load_environment()
    records = generate_candidates(env, k=DEFAULT_K)
    compare_candidate0_to_baseline(records)
    payload = build_output(env, records, k=DEFAULT_K)
    write_candidates_json(payload)
    print_summary(payload)


if __name__ == "__main__":
    main()
