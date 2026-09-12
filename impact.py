"""Phase 8: rank trips by current bottleneck impact. Ranking only; no rerouting."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from algorithms.baseline import disrupted_edge_set, load_environment
from bottlenecks import complete_flows, load_solution_flows
from env import DEFAULT_NUM_TRIPS, Environment
from flow import CAPACITY_ATTR, edge_key, route_edges

OUTPUT_PATH = Path("output") / "impact.json"
BASELINE_PATH = Path("output") / "baseline.json"
GREEDY_PATH = Path("output") / "greedy.json"
SOLUTION_PATHS = {
    "baseline": BASELINE_PATH,
    "greedy": GREEDY_PATH,
}
ALGORITHM_NAME = "trip_bottleneck_impact"
TOP_N = 10
FLOAT_ABS_TOL = 1e-12
RANKING_RULES = {
    "primary": "higher critical_edge_impact",
    "tie_breakers": [
        "higher impact_score",
        "higher max_edge_utilization",
        "higher critical_edge_count",
        "lower trip_index",
    ],
    "interpretation": (
        "A high-impact trip currently uses important/congested edges and is a "
        "promising candidate for later counterfactual testing. Impact does not "
        "prove that an alternative exists or that rerouting would improve the network."
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=FLOAT_ABS_TOL)


def load_solution(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require("trips" in payload and "routes" in payload, f"{path.as_posix()} missing trips/routes")
    return payload


def trip_ranking_key(record: dict) -> tuple:
    return (
        -record["critical_edge_impact"],
        -record["impact_score"],
        -record["max_edge_utilization"],
        -record["critical_edge_count"],
        record["trip_index"],
    )


def analyze_trip_impact(
    env: Environment,
    trip_index: int,
    origin: int,
    destination: int,
    route: list[int],
    flows: dict[tuple[int, int], int],
) -> dict:
    """Score one trip from the bottleneck importance of the edges it currently uses."""
    _require(len(route) >= 2, f"trip {trip_index} route is too short: {route}")
    _require(route[0] == origin, f"trip {trip_index} starts at {route[0]}, expected {origin}")
    _require(route[-1] == destination, f"trip {trip_index} ends at {route[-1]}, expected {destination}")
    _require(len(route) == len(set(route)), f"trip {trip_index} route is not simple: {route}")

    blocked = disrupted_edge_set()
    edges_used: list[dict] = []
    seen_edges: set[tuple[int, int]] = set()
    for u, v in route_edges(route):
        edge = (int(u), int(v))
        _require(edge not in seen_edges, f"trip {trip_index} repeats edge {u}->{v}")
        seen_edges.add(edge)
        _require(env.graph.has_edge(u, v), f"trip {trip_index} uses missing edge {u}->{v}")
        _require(edge not in blocked, f"trip {trip_index} uses disrupted edge {u}->{v}")
        _require(edge in flows, f"trip {trip_index} edge {u}->{v} missing from edge-flow data")
        capacity = float(env.graph[u][v][CAPACITY_ATTR])
        _require(capacity > 0, f"trip {trip_index} edge {u}->{v} has non-positive capacity")
        flow = int(flows[edge])
        utilization = flow / capacity
        _require(math.isfinite(utilization), f"trip {trip_index} utilization is not finite on {u}->{v}")
        contribution = utilization
        critical_excess = max(0.0, utilization - 1.0)
        edges_used.append(
            {
                "source": int(u),
                "destination": int(v),
                "edge": edge_key(u, v),
                "flow": flow,
                "capacity": capacity,
                "utilization": utilization,
                "congestion_ratio": utilization,
                "contribution_to_impact": contribution,
                "critical_excess_utilization": critical_excess,
            }
        )

    impact_score = sum(item["contribution_to_impact"] for item in edges_used)
    critical_edge_impact = sum(item["critical_excess_utilization"] for item in edges_used)
    max_edge_utilization = max(item["utilization"] for item in edges_used)
    critical_edge_count = sum(1 for item in edges_used if item["utilization"] > 1.0)
    _require(math.isfinite(impact_score), f"trip {trip_index} impact_score is not finite")
    _require(math.isfinite(critical_edge_impact), f"trip {trip_index} critical_edge_impact is not finite")
    _require(math.isfinite(max_edge_utilization), f"trip {trip_index} max_edge_utilization is not finite")

    return {
        "trip_index": trip_index,
        "source": int(origin),
        "destination": int(destination),
        "route": [int(node) for node in route],
        "route_length": len(edges_used),
        "impact_score": impact_score,
        "critical_edge_impact": critical_edge_impact,
        "max_edge_utilization": max_edge_utilization,
        "critical_edge_count": critical_edge_count,
        "edges_used": edges_used,
    }


def validate_ranked_trips(records: list[dict]) -> None:
    _require(len(records) == DEFAULT_NUM_TRIPS, f"expected {DEFAULT_NUM_TRIPS} ranked trips, got {len(records)}")
    indices = [record["trip_index"] for record in records]
    _require(sorted(indices) == list(range(DEFAULT_NUM_TRIPS)), "trip indices are not a unique 0..119 set")
    _require(records == sorted(records, key=trip_ranking_key), "ranking is not deterministic")
    for record in records:
        contrib = sum(item["contribution_to_impact"] for item in record["edges_used"])
        critical = sum(item["critical_excess_utilization"] for item in record["edges_used"])
        _require(_close(record["impact_score"], contrib), f"impact_score inconsistent for trip {record['trip_index']}")
        _require(
            _close(record["critical_edge_impact"], critical),
            f"critical_edge_impact inconsistent for trip {record['trip_index']}",
        )
        expected_max = max(item["utilization"] for item in record["edges_used"])
        _require(
            _close(record["max_edge_utilization"], expected_max),
            f"max_edge_utilization inconsistent for trip {record['trip_index']}",
        )
        expected_count = sum(1 for item in record["edges_used"] if item["utilization"] > 1.0)
        _require(
            record["critical_edge_count"] == expected_count,
            f"critical_edge_count inconsistent for trip {record['trip_index']}",
        )
        _require(record["route_length"] == len(record["edges_used"]), "route_length does not match edges_used")
        _require(record["route_length"] == len(record["route"]) - 1, "route_length does not match node sequence")


def analyze_solution(env: Environment, name: str, path: Path) -> dict:
    solution = load_solution(path)
    trips = [[int(origin), int(destination)] for origin, destination in solution["trips"]]
    routes = [[int(node) for node in route] for route in solution["routes"]]
    _require(len(trips) == DEFAULT_NUM_TRIPS, f"{name} does not have {DEFAULT_NUM_TRIPS} trips")
    _require(len(routes) == DEFAULT_NUM_TRIPS, f"{name} does not have {DEFAULT_NUM_TRIPS} routes")
    env_trips = [[int(origin), int(destination)] for origin, destination in env.trips]
    _require(trips == env_trips, f"{name} trips do not match the official env.py trip list")

    flows = complete_flows(env.graph, load_solution_flows(path))
    records = [
        analyze_trip_impact(env, index, origin, destination, route, flows)
        for index, ((origin, destination), route) in enumerate(zip(trips, routes))
    ]
    records.sort(key=trip_ranking_key)
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank
    validate_ranked_trips(records)
    top = records[:TOP_N]
    _require(top == records[:TOP_N], "top impact trips are not the first 10 ranked trips")
    return {
        "algorithm": name,
        "seed": env.seed,
        "trip_count": len(records),
        "ranked_trip_count": len(records),
        "ranking": RANKING_RULES,
        "trips": records,
        "top_impact_trips": top,
    }


def build_output(env: Environment, analyses: dict[str, dict]) -> dict:
    return {
        "algorithm": ALGORITHM_NAME,
        "seed": env.seed,
        "number_of_nodes": env.graph.number_of_nodes(),
        "number_of_directed_edges": env.graph.number_of_edges(),
        "number_of_trips": DEFAULT_NUM_TRIPS,
        "ranking": RANKING_RULES,
        **analyses,
    }


def write_impact_json(payload: dict, path: Path = OUTPUT_PATH) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(loaded, dict), "output/impact.json is not a JSON object")
    return loaded


def print_solution_summary(name: str, section: dict) -> None:
    top = section["top_impact_trips"][0]
    print(f"algorithm: {name}")
    print("trips:", section["trip_count"])
    print("top_impact_trip:", top["trip_index"])
    print("top_impact_score:", top["impact_score"])
    print("top_critical_edge_impact:", top["critical_edge_impact"])
    print(
        "top_route:",
        f"{top['source']}->{top['destination']}",
        top["route"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank trips by bottleneck impact.")
    parser.add_argument(
        "--algorithm",
        choices=sorted(SOLUTION_PATHS),
        help="Analyze one routing solution. Default: both baseline and greedy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = load_environment()
    selected = [args.algorithm] if args.algorithm else ["baseline", "greedy"]
    analyses = {name: analyze_solution(env, name, SOLUTION_PATHS[name]) for name in selected}
    payload = build_output(env, analyses)
    write_impact_json(payload)
    for name in selected:
        print_solution_summary(name, payload[name])
        print()
    print("wrote:", OUTPUT_PATH.as_posix())
    print("determinism: PASS")
    print("validation: PASS")


if __name__ == "__main__":
    main()
