"""Phase 4: sequential congestion-aware greedy routing."""

from __future__ import annotations

import heapq
import json
import math
import time
from pathlib import Path

import networkx as nx

from algorithms.baseline import (
    P95_METHOD,
    RUNTIME_FIELD,
    compute_metrics,
    disrupted_edge_set,
    load_environment,
    validate_flows_and_metrics,
    validate_output_schema,
)
from env import DEFAULT_NUM_TRIPS, Environment
from flow import (
    CAPACITY_ATTR,
    FREE_FLOW_ATTR,
    compute_congested_edge_times,
    compute_edge_flows,
    compute_trip_times,
    congested_travel_time,
    max_congestion_ratio,
    route_edges,
    serialize_edge_map,
)

OUTPUT_PATH = Path("output") / "greedy.json"
ALGORITHM_NAME = "congestion_aware_greedy"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def live_edge_cost(
    graph: nx.DiGraph,
    flows: dict[tuple[int, int], int],
    u: int,
    v: int,
) -> float:
    data = graph[u][v]
    return congested_travel_time(
        data[FREE_FLOW_ATTR],
        flows[(u, v)],
        data[CAPACITY_ATTR],
    )


def dijkstra_live(
    graph: nx.DiGraph,
    origin: int,
    destination: int,
    flows: dict[tuple[int, int], int],
) -> list[int]:
    """Deterministic Dijkstra with live congested costs and node-ID tie-breaks."""
    dist: dict[int, float] = {node: math.inf for node in graph.nodes()}
    parent: dict[int, int | None] = {node: None for node in graph.nodes()}
    dist[origin] = 0.0
    heap: list[tuple[float, int]] = [(0.0, origin)]
    finalized: set[int] = set()

    while heap:
        cost, node = heapq.heappop(heap)
        if node in finalized:
            continue
        if cost != dist[node]:
            continue
        finalized.add(node)
        if node == destination:
            break
        for nxt in sorted(graph.successors(node)):
            if nxt in finalized:
                continue
            candidate = cost + live_edge_cost(graph, flows, node, nxt)
            current = dist[nxt]
            if candidate < current:
                dist[nxt] = candidate
                parent[nxt] = node
                heapq.heappush(heap, (candidate, nxt))
            elif candidate == current and (parent[nxt] is None or node < parent[nxt]):
                parent[nxt] = node

    if not math.isfinite(dist[destination]):
        raise ValueError(f"No path from {origin} to {destination} on the disrupted graph")

    route: list[int] = []
    node: int | None = destination
    seen: set[int] = set()
    while node is not None:
        _require(node not in seen, f"Cycle while reconstructing path to {destination}")
        seen.add(node)
        route.append(node)
        if node == origin:
            break
        node = parent[node]
    route.reverse()
    _require(route[0] == origin, f"Reconstructed route does not start at {origin}: {route}")
    return route


def validate_selected_route(
    graph: nx.DiGraph,
    origin: int,
    destination: int,
    route: list[int],
    trip_index: int,
) -> None:
    blocked = disrupted_edge_set()
    _require(len(route) >= 2, f"Trip {trip_index} route has fewer than two nodes: {route}")
    _require(route[0] == origin, f"Trip {trip_index} starts at {route[0]}, expected {origin}")
    _require(route[-1] == destination, f"Trip {trip_index} ends at {route[-1]}, expected {destination}")
    _require(len(route) == len(set(route)), f"Trip {trip_index} route is not simple: {route}")
    for u, v in route_edges(route):
        _require(graph.has_edge(u, v), f"Trip {trip_index} uses missing directed edge {u}->{v}")
        _require((u, v) not in blocked, f"Trip {trip_index} uses disrupted edge {u}->{v}")


def validate_greedy_routes(
    graph: nx.DiGraph,
    trips: list[tuple[int, int]],
    routes: list[list[int]],
) -> None:
    _require(len(trips) == DEFAULT_NUM_TRIPS, f"Expected {DEFAULT_NUM_TRIPS} trips, got {len(trips)}")
    _require(len(routes) == DEFAULT_NUM_TRIPS, f"Expected {DEFAULT_NUM_TRIPS} routes, got {len(routes)}")
    _require(len(trips) == len(routes), "Trip count and route count differ")
    for index, ((origin, destination), route) in enumerate(zip(trips, routes)):
        validate_selected_route(graph, origin, destination, route, index)


def compute_greedy_routes(
    graph: nx.DiGraph,
    trips: list[tuple[int, int]],
) -> tuple[list[list[int]], dict[tuple[int, int], int]]:
    live_flows = {(u, v): 0 for u, v in graph.edges()}
    routes: list[list[int]] = []
    for index, (origin, destination) in enumerate(trips):
        route = dijkstra_live(graph, origin, destination, live_flows)
        validate_selected_route(graph, origin, destination, route, index)
        for u, v in route_edges(route):
            live_flows[(u, v)] += 1
        routes.append(route)
    return routes, live_flows


def run_greedy(env: Environment) -> dict:
    start = time.perf_counter()
    routes, live_flows = compute_greedy_routes(env.graph, env.trips)
    reconstructed = compute_edge_flows(env.graph, routes)
    _require(reconstructed == live_flows, "Live greedy flows do not match route traversal counts")
    _require(all(isinstance(value, int) and value >= 0 for value in live_flows.values()), "Flows must be non-negative integers")

    congested_times = compute_congested_edge_times(env.graph, reconstructed)
    trip_times = compute_trip_times(routes, congested_times)
    mean_travel_time, p95_travel_time = compute_metrics(trip_times)
    max_ratio = max_congestion_ratio(env.graph, reconstructed)
    runtime_seconds = time.perf_counter() - start

    validate_greedy_routes(env.graph, env.trips, routes)
    validate_flows_and_metrics(
        env.graph,
        routes,
        reconstructed,
        trip_times,
        mean_travel_time,
        p95_travel_time,
        max_ratio,
    )

    result = {
        "algorithm": ALGORITHM_NAME,
        "seed": env.seed,
        "number_of_nodes": env.graph.number_of_nodes(),
        "number_of_directed_edges": env.graph.number_of_edges(),
        "number_of_trips": len(env.trips),
        "number_of_routes": len(routes),
        "trips": [[int(origin), int(destination)] for origin, destination in env.trips],
        "routes": [[int(node) for node in route] for route in routes],
        "edge_flows": serialize_edge_map(reconstructed),
        "trip_travel_times": [float(t) for t in trip_times],
        "mean_travel_time": mean_travel_time,
        "p95_travel_time": p95_travel_time,
        "max_congestion_ratio": max_ratio,
        RUNTIME_FIELD: runtime_seconds,
        "metadata": {
            "runtime_seconds": runtime_seconds,
            "runtime_is_deterministic": False,
            "p95_method": P95_METHOD,
        },
    }
    validate_output_schema(result)
    return result


def write_greedy_json(result: dict, path: Path = OUTPUT_PATH) -> dict:
    validate_output_schema(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        path.unlink(missing_ok=True)
        raise ValueError("output/greedy.json is not a JSON object")
    validate_output_schema(loaded)
    return loaded


def main() -> None:
    env = load_environment()
    result = run_greedy(env)
    write_greedy_json(result)
    print("algorithm:", result["algorithm"])
    print("seed:", result["seed"])
    print("number_of_nodes:", result["number_of_nodes"])
    print("directed_edges:", result["number_of_directed_edges"])
    print("trips:", result["number_of_trips"])
    print("routes:", result["number_of_routes"])
    print("mean_travel_time:", result["mean_travel_time"])
    print("p95_travel_time:", result["p95_travel_time"])
    print("max_congestion_ratio:", result["max_congestion_ratio"])
    print("runtime_seconds:", result["runtime_seconds"])
    print("wrote:", OUTPUT_PATH.as_posix())


if __name__ == "__main__":
    main()
