"""Phase 9: hypothetical one-trip counterfactual evaluation. No solution updates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from algorithms.baseline import compute_metrics, disrupted_edge_set, load_environment
from bottlenecks import complete_flows, parse_edge_flows
from env import DEFAULT_NUM_TRIPS, Environment
from flow import (
    compute_congested_edge_times,
    compute_trip_times,
    max_congestion_ratio,
    route_edges,
)

OUTPUT_PATH = Path("output") / "counterfactual.json"
BASELINE_PATH = Path("output") / "baseline.json"
GREEDY_PATH = Path("output") / "greedy.json"
CANDIDATES_PATH = Path("output") / "candidates.json"
IMPACT_PATH = Path("output") / "impact.json"
SOLUTION_PATHS = {
    "baseline": BASELINE_PATH,
    "greedy": GREEDY_PATH,
}
ALGORITHM_NAME = "one_trip_counterfactual"
DEFAULT_TOP_N = 10
FLOAT_TOL = 1e-12
COMPARISON_RULE = {
    "primary_objective": "lower mean travel time",
    "tie_breakers": [
        "lower p95 travel time",
        "lower max congestion ratio",
    ],
    "tolerance": FLOAT_TOL,
    "note": "A candidate is better if its (mean, p95, max congestion) tuple is lexicographically smaller. All three metrics need not improve at once.",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=FLOAT_TOL)


def _less(left: float, right: float) -> bool:
    return left < right - FLOAT_TOL


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{path.as_posix()} is not a JSON object")
    return payload


def load_solution(path: Path) -> dict:
    payload = load_json(path)
    _require(len(payload["trips"]) == DEFAULT_NUM_TRIPS, f"{path.as_posix()} does not have 120 trips")
    _require(len(payload["routes"]) == DEFAULT_NUM_TRIPS, f"{path.as_posix()} does not have 120 routes")
    return payload


def load_candidates(path: Path = CANDIDATES_PATH) -> list[dict]:
    payload = load_json(path)
    records = payload["candidate_routes"]
    _require(len(records) == DEFAULT_NUM_TRIPS, "candidates.json does not contain 120 trips")
    return records


def load_impact_section(algorithm: str, path: Path = IMPACT_PATH) -> dict:
    payload = load_json(path)
    _require(algorithm in payload, f"impact.json has no section for {algorithm}")
    section = payload[algorithm]
    _require(section["ranked_trip_count"] == DEFAULT_NUM_TRIPS, f"{algorithm} impact ranking is not 120 trips")
    _require(len(section["trips"]) == DEFAULT_NUM_TRIPS, f"{algorithm} impact trips are not 120")
    return section


def validate_route(env: Environment, origin: int, destination: int, route: list[int], label: str) -> None:
    blocked = disrupted_edge_set()
    _require(len(route) >= 2, f"{label} is too short: {route}")
    _require(route[0] == origin, f"{label} starts at {route[0]}, expected {origin}")
    _require(route[-1] == destination, f"{label} ends at {route[-1]}, expected {destination}")
    _require(len(route) == len(set(route)), f"{label} is not simple: {route}")
    for u, v in route_edges(route):
        _require(env.graph.has_edge(u, v), f"{label} uses missing edge {u}->{v}")
        _require((u, v) not in blocked, f"{label} uses disrupted edge {u}->{v}")


def remove_route_from_flow(
    flows: dict[tuple[int, int], int],
    route: list[int],
) -> dict[tuple[int, int], int]:
    updated = dict(flows)
    for u, v in route_edges(route):
        edge = (int(u), int(v))
        _require(edge in updated, f"cannot remove missing flow edge {u}->{v}")
        updated[edge] -= 1
        _require(updated[edge] >= 0, f"removing route produced negative flow on {u}->{v}")
    return updated


def add_route_to_flow(
    flows: dict[tuple[int, int], int],
    route: list[int],
) -> dict[tuple[int, int], int]:
    updated = dict(flows)
    for u, v in route_edges(route):
        edge = (int(u), int(v))
        _require(edge in updated, f"cannot add flow on missing graph edge {u}->{v}")
        updated[edge] += 1
    return updated


def apply_route_swap(
    current_flows: dict[tuple[int, int], int],
    current_route: list[int],
    candidate_route: list[int],
) -> dict[tuple[int, int], int]:
    return add_route_to_flow(remove_route_from_flow(dict(current_flows), current_route), candidate_route)


def compute_network_metrics(
    env: Environment,
    routes: list[list[int]],
    flows: dict[tuple[int, int], int],
) -> dict:
    congested = compute_congested_edge_times(env.graph, flows)
    trip_times = compute_trip_times(routes, congested)
    _require(len(trip_times) == DEFAULT_NUM_TRIPS, "trip travel times must cover 120 trips")
    _require(all(math.isfinite(value) and value > 0 for value in trip_times), "non-finite trip travel time")
    mean_travel_time, p95_travel_time = compute_metrics(trip_times)
    max_ratio = float(max_congestion_ratio(env.graph, flows))
    for name, value in (
        ("mean_travel_time", mean_travel_time),
        ("p95_travel_time", p95_travel_time),
        ("max_congestion_ratio", max_ratio),
    ):
        _require(math.isfinite(value), f"{name} is not finite")
    expected_mean = float(sum(trip_times) / len(trip_times))
    _require(_close(mean_travel_time, expected_mean), "mean travel time is inconsistent")
    return {
        "mean_travel_time": mean_travel_time,
        "p95_travel_time": p95_travel_time,
        "max_congestion_ratio": max_ratio,
        "total_trip_travel_time": float(sum(trip_times)),
        "maximum_trip_travel_time": float(max(trip_times)),
    }


def compare_metrics(candidate: dict, current: dict) -> tuple[bool, bool, str]:
    cand_tuple = (
        candidate["mean_travel_time"],
        candidate["p95_travel_time"],
        candidate["max_congestion_ratio"],
    )
    curr_tuple = (
        current["mean_travel_time"],
        current["p95_travel_time"],
        current["max_congestion_ratio"],
    )
    if all(_close(a, b) for a, b in zip(cand_tuple, curr_tuple)):
        return False, False, "equivalent within tolerance; not a reroute improvement"
    for cand_value, curr_value, name in zip(
        cand_tuple,
        curr_tuple,
        ("mean travel time", "p95 travel time", "max congestion ratio"),
    ):
        if _less(cand_value, curr_value):
            return True, False, f"improves lexicographic objective at {name}"
        if _less(curr_value, cand_value):
            return False, True, f"worse lexicographic objective at {name}"
    return False, False, "equivalent within tolerance; not a reroute improvement"


def evaluate_candidate(
    env: Environment,
    current_routes: list[list[int]],
    current_flows: dict[tuple[int, int], int],
    current_metrics: dict,
    trip_index: int,
    origin: int,
    destination: int,
    current_route: list[int],
    candidate_index: int,
    candidate_route: list[int],
    official_candidates: list[list[int]],
) -> dict:
    _require(candidate_route in official_candidates, f"trip {trip_index} candidate {candidate_index} is not in Phase 5")
    validate_route(env, origin, destination, candidate_route, f"trip {trip_index} candidate {candidate_index}")
    is_current = candidate_route == current_route
    if is_current:
        metrics = dict(current_metrics)
        better, worse, reason = False, False, "candidate is the current route; no reroute"
    else:
        cf_flows = apply_route_swap(current_flows, current_route, candidate_route)
        cf_routes = list(current_routes)
        cf_routes[trip_index] = candidate_route
        metrics = compute_network_metrics(env, cf_routes, cf_flows)
        better, worse, reason = compare_metrics(metrics, current_metrics)
        _require(not (better and worse), "candidate cannot be both better and worse")

    delta_mean = metrics["mean_travel_time"] - current_metrics["mean_travel_time"]
    delta_p95 = metrics["p95_travel_time"] - current_metrics["p95_travel_time"]
    delta_max = metrics["max_congestion_ratio"] - current_metrics["max_congestion_ratio"]
    _require(_close(delta_mean, metrics["mean_travel_time"] - current_metrics["mean_travel_time"]), "delta_mean mismatch")
    return {
        "candidate_index": candidate_index,
        "candidate_route": [int(node) for node in candidate_route],
        "route_length": len(candidate_route) - 1,
        "is_current_route": is_current,
        "candidate_mean_travel_time": metrics["mean_travel_time"],
        "candidate_p95_travel_time": metrics["p95_travel_time"],
        "candidate_max_congestion_ratio": metrics["max_congestion_ratio"],
        "delta_mean_travel_time": delta_mean,
        "delta_p95_travel_time": delta_p95,
        "delta_max_congestion_ratio": delta_max,
        "candidate_is_better": better,
        "candidate_is_worse": worse,
        "comparison_reason": reason,
    }


def candidate_objective_key(result: dict) -> tuple:
    return (
        result["candidate_mean_travel_time"],
        result["candidate_p95_travel_time"],
        result["candidate_max_congestion_ratio"],
        result["candidate_index"],
    )


def improvement_rank_key(item: dict) -> tuple:
    return (
        item["best_delta_mean"],
        item["best_delta_p95"],
        item["best_delta_max_congestion"],
        item["trip_index"],
        item["best_candidate"],
    )


def evaluate_trip(
    env: Environment,
    current_routes: list[list[int]],
    current_flows: dict[tuple[int, int], int],
    current_metrics: dict,
    impact_record: dict,
    candidate_record: dict,
) -> dict:
    trip_index = int(impact_record["trip_index"])
    origin = int(impact_record["source"])
    destination = int(impact_record["destination"])
    current_route = [int(node) for node in current_routes[trip_index]]
    _require(current_route == [int(node) for node in impact_record["route"]], f"impact route mismatch for trip {trip_index}")
    _require(candidate_record["trip_index"] == trip_index, "candidate trip_index mismatch")
    _require(candidate_record["origin"] == origin and candidate_record["destination"] == destination, "candidate endpoints mismatch")
    validate_route(env, origin, destination, current_route, f"trip {trip_index} current route")
    official = [[int(node) for node in route] for route in candidate_record["candidates"]]
    _require(official, f"trip {trip_index} has no Phase 5 candidates")

    evaluated = [
        evaluate_candidate(
            env,
            current_routes,
            current_flows,
            current_metrics,
            trip_index,
            origin,
            destination,
            current_route,
            candidate_index,
            candidate_route,
            official,
        )
        for candidate_index, candidate_route in enumerate(official)
    ]
    best = min(evaluated, key=candidate_objective_key)
    _require(best is min(evaluated, key=candidate_objective_key), "best candidate is not unique/deterministic")
    genuine = bool(best["candidate_is_better"])
    return {
        "trip_index": trip_index,
        "source": origin,
        "destination": destination,
        "current_route": current_route,
        "impact_rank": int(impact_record["rank"]),
        "impact_score": float(impact_record["impact_score"]),
        "critical_edge_impact": float(impact_record["critical_edge_impact"]),
        "candidates": evaluated,
        "best_candidate": best["candidate_index"],
        "best_candidate_metrics": {
            "mean_travel_time": best["candidate_mean_travel_time"],
            "p95_travel_time": best["candidate_p95_travel_time"],
            "max_congestion_ratio": best["candidate_max_congestion_ratio"],
        },
        "best_delta_mean": best["delta_mean_travel_time"],
        "best_delta_p95": best["delta_p95_travel_time"],
        "best_delta_max_congestion": best["delta_max_congestion_ratio"],
        "genuine_improvement": genuine,
    }


def run_algorithm(env: Environment, name: str, top_n: int) -> dict:
    solution = load_solution(SOLUTION_PATHS[name])
    candidates = load_candidates()
    impact = load_impact_section(name)
    _require(top_n > 0, "top_n must be positive")
    selected_impact = impact["trips"][:top_n]
    _require(len(selected_impact) == top_n, f"impact ranking does not contain top_n={top_n} trips")
    _require(selected_impact == impact["top_impact_trips"][:top_n], "selected trips are not the top-N impact ranking")

    trips = [[int(a), int(b)] for a, b in solution["trips"]]
    routes = [[int(node) for node in route] for route in solution["routes"]]
    env_trips = [[int(a), int(b)] for a, b in env.trips]
    _require(trips == env_trips, f"{name} trips do not match env.py")
    flows = complete_flows(env.graph, parse_edge_flows(solution["edge_flows"]))
    current_metrics = compute_network_metrics(env, routes, flows)
    _require(_close(current_metrics["mean_travel_time"], float(solution["mean_travel_time"])), f"{name} mean mismatch")
    _require(_close(current_metrics["p95_travel_time"], float(solution["p95_travel_time"])), f"{name} p95 mismatch")
    _require(
        _close(current_metrics["max_congestion_ratio"], float(solution["max_congestion_ratio"])),
        f"{name} max congestion mismatch",
    )

    candidate_by_trip = {int(record["trip_index"]): record for record in candidates}
    trip_results = [
        evaluate_trip(
            env,
            routes,
            flows,
            current_metrics,
            impact_record,
            candidate_by_trip[int(impact_record["trip_index"])],
        )
        for impact_record in selected_impact
    ]
    improvements = [result for result in trip_results if result["genuine_improvement"]]
    improvements.sort(key=improvement_rank_key)
    candidate_count = sum(len(result["candidates"]) for result in trip_results)
    validate_algorithm_result(trip_results, improvements, top_n)
    return {
        "algorithm": name,
        "seed": env.seed,
        "top_n": top_n,
        "selected_trip_count": len(trip_results),
        "candidate_count_tested": candidate_count,
        "comparison_rule": COMPARISON_RULE,
        "current_network_metrics": current_metrics,
        "trip_results": trip_results,
        "best_counterfactuals": [
            {
                "trip_index": result["trip_index"],
                "impact_rank": result["impact_rank"],
                "best_candidate": result["best_candidate"],
                "best_candidate_metrics": result["best_candidate_metrics"],
                "best_delta_mean": result["best_delta_mean"],
                "best_delta_p95": result["best_delta_p95"],
                "best_delta_max_congestion": result["best_delta_max_congestion"],
            }
            for result in improvements
        ],
        "genuine_improvements_found": len(improvements),
    }


def validate_algorithm_result(trip_results: list[dict], improvements: list[dict], top_n: int) -> None:
    _require(len(trip_results) == top_n, f"selected trip count {len(trip_results)} != top_n {top_n}")
    seen = set()
    for result in trip_results:
        _require(result["trip_index"] not in seen, f"duplicate selected trip {result['trip_index']}")
        seen.add(result["trip_index"])
        best = min(result["candidates"], key=candidate_objective_key)
        _require(best["candidate_index"] == result["best_candidate"], "stored best candidate is not the objective best")
        _require(result["genuine_improvement"] is best["candidate_is_better"], "genuine_improvement flag mismatch")
        for candidate in result["candidates"]:
            if candidate["is_current_route"]:
                _require(not candidate["candidate_is_better"], "current route marked as an improvement")
    ranked = sorted(improvements, key=improvement_rank_key)
    _require(ranked == improvements, "best_counterfactuals ranking is not deterministic")


def print_algorithm_summary(section: dict) -> None:
    print("algorithm:", section["algorithm"])
    print("selected trips:", section["selected_trip_count"])
    print("candidates evaluated:", section["candidate_count_tested"])
    print("genuine improvements found:", section["genuine_improvements_found"])
    if section["best_counterfactuals"]:
        best = section["best_counterfactuals"][0]
        print("best trip:", best["trip_index"])
        print("best candidate:", best["best_candidate"])
        print("best mean delta:", best["best_delta_mean"])
        print("best p95 delta:", best["best_delta_p95"])
        print("best max congestion delta:", best["best_delta_max_congestion"])
    else:
        print("best trip:", None)
        print("best candidate:", None)
        print("best mean delta:", None)
        print("best p95 delta:", None)
        print("best max congestion delta:", None)
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one-trip counterfactual reroutes.")
    parser.add_argument("--algorithm", choices=sorted(SOLUTION_PATHS))
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = load_environment()
    names = [args.algorithm] if args.algorithm else ["baseline", "greedy"]
    analyses = {name: run_algorithm(env, name, args.top_n) for name in names}
    payload = {
        "algorithm": ALGORITHM_NAME,
        "seed": env.seed,
        "top_n": args.top_n,
        "selected_trip_count": args.top_n,
        "comparison_rule": COMPARISON_RULE,
        **analyses,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    loaded = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    _require(isinstance(loaded, dict), "output/counterfactual.json is not valid JSON")
    for name in names:
        print_algorithm_summary(analyses[name])
    print("wrote:", OUTPUT_PATH.as_posix())
    print("determinism: PASS")
    print("validation: PASS")


if __name__ == "__main__":
    main()
