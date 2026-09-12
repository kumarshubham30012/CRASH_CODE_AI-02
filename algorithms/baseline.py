"""Phase 2 baseline: independent free-flow Dijkstra for every trip."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import networkx as nx
import numpy as np

from env import (
    DEFAULT_NUM_TRIPS,
    REMOVED_COORD_PAIRS,
    Environment,
    create_environment,
    node_id,
)
from flow import (
    FREE_FLOW_ATTR,
    compute_congested_edge_times,
    compute_edge_flows,
    compute_trip_times,
    max_congestion_ratio,
    route_edges,
    serialize_edge_map,
)

OUTPUT_PATH = Path("output") / "baseline.json"
ALGORITHM_NAME = "baseline_independent_shortest_path"
WEIGHT_ATTR = FREE_FLOW_ATTR


def load_environment() -> Environment:
    return create_environment()


def shortest_path_for_trip(
    graph: nx.DiGraph,
    origin: int,
    destination: int,
) -> list[int]:
    return nx.dijkstra_path(graph, origin, destination, weight=WEIGHT_ATTR)


def compute_baseline_routes(
    graph: nx.DiGraph,
    trips: list[tuple[int, int]],
) -> list[list[int]]:
    return [shortest_path_for_trip(graph, origin, dest) for origin, dest in trips]


def disrupted_edge_set() -> set[tuple[int, int]]:
    return {
        (node_id(x1, y1), node_id(x2, y2))
        for (x1, y1), (x2, y2) in REMOVED_COORD_PAIRS
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_routes(
    graph: nx.DiGraph,
    trips: list[tuple[int, int]],
    routes: list[list[int]],
) -> None:
    _require(len(trips) == DEFAULT_NUM_TRIPS, f"Expected {DEFAULT_NUM_TRIPS} trips, got {len(trips)}")
    _require(len(routes) == DEFAULT_NUM_TRIPS, f"Expected {DEFAULT_NUM_TRIPS} routes, got {len(routes)}")
    _require(len(trips) == len(routes), "Trip count and route count differ")

    blocked = disrupted_edge_set()
    for index, ((origin, destination), route) in enumerate(zip(trips, routes)):
        _require(len(route) >= 2, f"Trip {index} route is too short: {route}")
        _require(route[0] == origin, f"Trip {index} route does not start at origin {origin}: {route}")
        _require(route[-1] == destination, f"Trip {index} route does not end at destination {destination}: {route}")
        _require(len(route) == len(set(route)), f"Trip {index} route is not simple: {route}")

        edges = route_edges(route)
        for u, v in edges:
            _require(graph.has_edge(u, v), f"Trip {index} uses missing directed edge {u}->{v}")
            _require((u, v) not in blocked, f"Trip {index} uses disrupted edge {u}->{v}")

        route_free_flow = sum(graph[u][v][WEIGHT_ATTR] for u, v in edges)
        shortest_free_flow = nx.dijkstra_path_length(
            graph, origin, destination, weight=WEIGHT_ATTR
        )
        _require(
            math.isfinite(route_free_flow) and route_free_flow > 0,
            f"Trip {index} has invalid free-flow distance {route_free_flow}",
        )
        _require(
            math.isclose(route_free_flow, shortest_free_flow, rel_tol=0.0, abs_tol=1e-12),
            (
                f"Trip {index} free-flow distance {route_free_flow} "
                f"is not a shortest-path length {shortest_free_flow}"
            ),
        )


def validate_flows_and_metrics(
    graph: nx.DiGraph,
    routes: list[list[int]],
    flows: dict[tuple[int, int], int],
    trip_times: list[float],
    mean_travel_time: float,
    p95_travel_time: float,
    max_ratio: float,
) -> None:
    recomputed = compute_edge_flows(graph, routes)
    _require(recomputed == flows, "Edge flows do not match route traversal counts")

    traversal_sum = sum(len(route) - 1 for route in routes)
    flow_sum = sum(flows.values())
    _require(
        traversal_sum == flow_sum,
        f"Sum of route edge traversals ({traversal_sum}) != sum of edge flows ({flow_sum})",
    )

    _require(len(trip_times) == len(routes), "Each route must have a travel time")
    for index, travel_time in enumerate(trip_times):
        _require(
            math.isfinite(travel_time) and travel_time > 0,
            f"Trip {index} has non-finite or non-positive travel time {travel_time}",
        )

    for name, value in (
        ("mean_travel_time", mean_travel_time),
        ("p95_travel_time", p95_travel_time),
        ("max_congestion_ratio", max_ratio),
    ):
        _require(math.isfinite(value), f"Metric {name} is not finite: {value}")


def compute_metrics(trip_times: list[float]) -> tuple[float, float]:
    times = np.asarray(trip_times, dtype=float)
    mean_travel_time = float(np.mean(times))
    p95_travel_time = float(np.percentile(times, 95, method="linear"))
    return mean_travel_time, p95_travel_time


def run_baseline(env: Environment) -> dict:
    start = time.perf_counter()
    routes = compute_baseline_routes(env.graph, env.trips)
    flows = compute_edge_flows(env.graph, routes)
    congested_times = compute_congested_edge_times(env.graph, flows)
    trip_times = compute_trip_times(routes, congested_times)
    mean_travel_time, p95_travel_time = compute_metrics(trip_times)
    max_ratio = max_congestion_ratio(env.graph, flows)
    runtime_seconds = time.perf_counter() - start

    validate_routes(env.graph, env.trips, routes)
    validate_flows_and_metrics(
        env.graph,
        routes,
        flows,
        trip_times,
        mean_travel_time,
        p95_travel_time,
        max_ratio,
    )

    return {
        "algorithm": ALGORITHM_NAME,
        "seed": env.seed,
        "number_of_nodes": env.graph.number_of_nodes(),
        "number_of_directed_edges": env.graph.number_of_edges(),
        "number_of_trips": len(env.trips),
        "trips": [[int(origin), int(destination)] for origin, destination in env.trips],
        "routes": [[int(node) for node in route] for route in routes],
        "edge_flows": serialize_edge_map(flows),
        "trip_travel_times": [float(t) for t in trip_times],
        "mean_travel_time": mean_travel_time,
        "p95_travel_time": p95_travel_time,
        "max_congestion_ratio": max_ratio,
        "runtime_seconds": runtime_seconds,
    }


def write_baseline_json(result: dict, path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        path.unlink(missing_ok=True)
        raise ValueError("output/baseline.json is not a JSON object")


def comparable_result(result: dict) -> dict:
    payload = dict(result)
    payload.pop("runtime_seconds", None)
    return payload


def main() -> None:
    env = load_environment()
    result = run_baseline(env)
    write_baseline_json(result)
    print("algorithm:", result["algorithm"])
    print("seed:", result["seed"])
    print("number_of_nodes:", result["number_of_nodes"])
    print("number_of_directed_edges:", result["number_of_directed_edges"])
    print("number_of_trips:", result["number_of_trips"])
    print("number_of_routes:", len(result["routes"]))
    print("mean_travel_time:", result["mean_travel_time"])
    print("p95_travel_time:", result["p95_travel_time"])
    print("max_congestion_ratio:", result["max_congestion_ratio"])
    print("runtime_seconds:", result["runtime_seconds"])
    print("wrote:", OUTPUT_PATH.as_posix())


if __name__ == "__main__":
    main()
