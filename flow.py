"""Flow aggregation and official BPR congestion evaluation."""

from __future__ import annotations

import networkx as nx

BPR_ALPHA = 0.15
BPR_BETA = 4.0
FREE_FLOW_ATTR = "free_flow_time"
CAPACITY_ATTR = "capacity"


def edge_key(u: int, v: int) -> str:
    return f"{u}->{v}"


def parse_edge_key(key: str) -> tuple[int, int]:
    left, right = key.split("->", 1)
    return int(left), int(right)


def route_edges(route: list[int]) -> list[tuple[int, int]]:
    return list(zip(route, route[1:]))


def compute_edge_flows(
    graph: nx.DiGraph,
    routes: list[list[int]],
) -> dict[tuple[int, int], int]:
    """Count how many routes traverse each directed edge in the graph."""
    flows = {(u, v): 0 for u, v in graph.edges()}
    for route in routes:
        for u, v in route_edges(route):
            if (u, v) not in flows:
                raise ValueError(f"Route uses edge {u}->{v} that is not in the graph")
            flows[(u, v)] += 1
    return flows


def congested_travel_time(t0: float, flow: float, capacity: float) -> float:
    """Official BPR: t(f) = t0 * (1 + 0.15 * (f / capacity)^4)."""
    return t0 * (1.0 + BPR_ALPHA * (flow / capacity) ** BPR_BETA)


def compute_congested_edge_times(
    graph: nx.DiGraph,
    flows: dict[tuple[int, int], int],
) -> dict[tuple[int, int], float]:
    """Evaluate congested travel time on every directed edge using final flows."""
    times: dict[tuple[int, int], float] = {}
    for u, v, data in graph.edges(data=True):
        t0 = data[FREE_FLOW_ATTR]
        capacity = data[CAPACITY_ATTR]
        flow = flows.get((u, v), 0)
        times[(u, v)] = congested_travel_time(t0, flow, capacity)
    return times


def compute_trip_times(
    routes: list[list[int]],
    congested_edge_times: dict[tuple[int, int], float],
) -> list[float]:
    """Sum final congested edge times along each route."""
    trip_times: list[float] = []
    for route in routes:
        total = 0.0
        for u, v in route_edges(route):
            total += congested_edge_times[(u, v)]
        trip_times.append(total)
    return trip_times


def compute_congestion_ratios(
    graph: nx.DiGraph,
    flows: dict[tuple[int, int], int],
) -> dict[tuple[int, int], float]:
    """Return flow/capacity for every directed edge."""
    return {
        (u, v): flows.get((u, v), 0) / graph[u][v][CAPACITY_ATTR]
        for u, v in graph.edges()
    }


def max_congestion_ratio(
    graph: nx.DiGraph,
    flows: dict[tuple[int, int], int],
) -> float:
    ratios = compute_congestion_ratios(graph, flows)
    if not ratios:
        raise ValueError("Cannot compute max congestion ratio on an empty graph")
    return max(ratios.values())


def serialize_edge_map(values: dict[tuple[int, int], int | float]) -> dict[str, int | float]:
    return {edge_key(u, v): values[(u, v)] for u, v in sorted(values)}
