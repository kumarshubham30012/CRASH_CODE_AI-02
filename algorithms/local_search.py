"""Phase 10: deterministic one-move-per-iteration local search. Not the final optimizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from algorithms.baseline import load_environment
from bottlenecks import analyze_bottlenecks, complete_flows, parse_edge_flows
from counterfactual import (
    COMPARISON_RULE,
    DEFAULT_TOP_N,
    FLOAT_TOL,
    SOLUTION_PATHS,
    _close,
    _require,
    apply_route_swap,
    compare_metrics,
    compute_network_metrics,
    load_candidates,
    load_solution,
    validate_route,
)
from env import DEFAULT_NUM_TRIPS, Environment
from flow import compute_edge_flows, serialize_edge_map
from impact import analyze_trip_impact, trip_ranking_key

OUTPUT_PATH = Path("output") / "local_search.json"
ALGORITHM_NAME = "deterministic_local_search"
DEFAULT_MAX_ITERATIONS = 25
EXPECTED_INITIAL = {
    "baseline": {
        "mean_travel_time": 8.768970642089842,
        "p95_travel_time": 14.204617919921873,
        "max_congestion_ratio": 2.375,
    },
    "greedy": {
        "mean_travel_time": 6.030344543457032,
        "p95_travel_time": 7.853384399414063,
        "max_congestion_ratio": 1.625,
    },
}


def reconstruct_flows(env: Environment, routes: list[list[int]]) -> dict[tuple[int, int], int]:
    return compute_edge_flows(env.graph, routes)


def validate_state(env: Environment, trips: list[list[int]], routes: list[list[int]], flows: dict[tuple[int, int], int]) -> None:
    _require(len(trips) == DEFAULT_NUM_TRIPS, "state does not contain 120 trips")
    _require(len(routes) == DEFAULT_NUM_TRIPS, "state does not contain 120 routes")
    for index, ((origin, destination), route) in enumerate(zip(trips, routes)):
        validate_route(env, origin, destination, route, f"trip {index} current route")
    reconstructed = reconstruct_flows(env, routes)
    _require(reconstructed == flows, "maintained flows do not match reconstruction from all current routes")
    _require(all(value >= 0 for value in flows.values()), "negative edge flow in current state")


def compute_trip_impacts(
    env: Environment,
    trips: list[list[int]],
    routes: list[list[int]],
    flows: dict[tuple[int, int], int],
) -> list[dict]:
    records = [
        analyze_trip_impact(env, index, origin, destination, route, flows)
        for index, ((origin, destination), route) in enumerate(zip(trips, routes))
    ]
    records.sort(key=trip_ranking_key)
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank
    return records


def evaluate_move(
    env: Environment,
    trips: list[list[int]],
    routes: list[list[int]],
    flows: dict[tuple[int, int], int],
    current_metrics: dict,
    trip_index: int,
    candidate_index: int,
    candidate_route: list[int],
    official_candidates: list[list[int]],
) -> dict | None:
    origin, destination = trips[trip_index]
    current_route = routes[trip_index]
    if candidate_route == current_route:
        return None
    _require(candidate_route in official_candidates, f"trip {trip_index} candidate {candidate_index} is not in Phase 5")
    validate_route(env, origin, destination, candidate_route, f"trip {trip_index} candidate {candidate_index}")
    cf_flows = apply_route_swap(flows, current_route, candidate_route)
    cf_routes = list(routes)
    cf_routes[trip_index] = candidate_route
    after_metrics = compute_network_metrics(env, cf_routes, cf_flows)
    better, _worse, reason = compare_metrics(after_metrics, current_metrics)
    if not better:
        return None
    return {
        "trip_index": trip_index,
        "source": origin,
        "destination": destination,
        "old_route": [int(node) for node in current_route],
        "new_route": [int(node) for node in candidate_route],
        "candidate_index": candidate_index,
        "after_metrics": after_metrics,
        "comparison_reason": reason,
        "delta_mean_travel_time": after_metrics["mean_travel_time"] - current_metrics["mean_travel_time"],
        "delta_p95_travel_time": after_metrics["p95_travel_time"] - current_metrics["p95_travel_time"],
        "delta_max_congestion_ratio": after_metrics["max_congestion_ratio"] - current_metrics["max_congestion_ratio"],
    }


def move_selection_key(move: dict) -> tuple:
    return (
        move["after_metrics"]["mean_travel_time"],
        move["after_metrics"]["p95_travel_time"],
        move["after_metrics"]["max_congestion_ratio"],
        move["trip_index"],
        move["candidate_index"],
    )


def select_best_move(moves: list[dict]) -> dict | None:
    if not moves:
        return None
    return min(moves, key=move_selection_key)


def apply_move(
    env: Environment,
    trips: list[list[int]],
    routes: list[list[int]],
    flows: dict[tuple[int, int], int],
    move: dict,
) -> tuple[list[list[int]], dict[tuple[int, int], int]]:
    trip_index = move["trip_index"]
    new_routes = list(routes)
    new_routes[trip_index] = list(move["new_route"])
    new_flows = apply_route_swap(flows, move["old_route"], move["new_route"])
    validate_state(env, trips, new_routes, new_flows)
    return new_routes, new_flows


def metrics_no_worse(final_metrics: dict, initial_metrics: dict) -> bool:
    better, worse, _reason = compare_metrics(final_metrics, initial_metrics)
    equivalent = all(
        _close(final_metrics[key], initial_metrics[key])
        for key in ("mean_travel_time", "p95_travel_time", "max_congestion_ratio")
    )
    return better or equivalent and not worse


def run_local_search(
    env: Environment,
    name: str,
    top_n: int,
    max_iterations: int,
) -> dict:
    solution = load_solution(SOLUTION_PATHS[name])
    candidate_records = load_candidates()
    candidates_by_trip = {
        int(record["trip_index"]): [[int(node) for node in route] for route in record["candidates"]]
        for record in candidate_records
    }
    trips = [[int(origin), int(destination)] for origin, destination in solution["trips"]]
    _require(trips == [[int(origin), int(destination)] for origin, destination in env.trips], f"{name} trips do not match env.py")
    routes = [[int(node) for node in route] for route in solution["routes"]]
    flows = complete_flows(env.graph, parse_edge_flows(solution["edge_flows"]))
    validate_state(env, trips, routes, flows)
    initial_metrics = compute_network_metrics(env, routes, flows)
    expected = EXPECTED_INITIAL[name]
    for key, value in expected.items():
        _require(_close(initial_metrics[key], value), f"{name} initial {key} is {initial_metrics[key]}, expected {value}")

    initial_routes = [list(route) for route in routes]
    initial_flows = dict(flows)
    accepted_moves: list[dict] = []
    history: list[dict] = []
    stop_reason = "no_improving_move"
    iterations_completed = 0

    for iteration in range(1, max_iterations + 1):
        iterations_completed = iteration
        analyze_bottlenecks(env, flows)
        impacts = compute_trip_impacts(env, trips, routes, flows)
        selected = impacts[:top_n]
        _require(len(selected) == top_n, f"expected top_n={top_n} impact trips")
        metrics_before = compute_network_metrics(env, routes, flows)
        evaluated_moves: list[dict] = []
        candidate_moves_evaluated = 0
        for impact_record in selected:
            trip_index = int(impact_record["trip_index"])
            official = candidates_by_trip[trip_index]
            for candidate_index, candidate_route in enumerate(official):
                candidate_moves_evaluated += 1
                move = evaluate_move(
                    env,
                    trips,
                    routes,
                    flows,
                    metrics_before,
                    trip_index,
                    candidate_index,
                    candidate_route,
                    official,
                )
                if move is not None:
                    evaluated_moves.append(move)

        best = select_best_move(evaluated_moves)
        if best is None:
            history.append(
                {
                    "iteration": iteration,
                    "metrics_before": metrics_before,
                    "selected_trip_count": len(selected),
                    "candidate_moves_evaluated": candidate_moves_evaluated,
                    "best_candidate_move": None,
                    "accepted": False,
                    "metrics_after": metrics_before,
                    "stop_reason_if_any": "no_improving_move",
                }
            )
            stop_reason = "no_improving_move"
            break

        better, _worse, _reason = compare_metrics(best["after_metrics"], metrics_before)
        _require(better, "selected move does not improve the lexicographic objective")
        routes, flows = apply_move(env, trips, routes, flows, best)
        metrics_after = compute_network_metrics(env, routes, flows)
        for key in ("mean_travel_time", "p95_travel_time", "max_congestion_ratio"):
            _require(_close(metrics_after[key], best["after_metrics"][key]), f"applied {key} does not match evaluated move")
        accepted = {
            "iteration": iteration,
            "trip_index": best["trip_index"],
            "source": best["source"],
            "destination": best["destination"],
            "old_route": best["old_route"],
            "new_route": best["new_route"],
            "candidate_index": best["candidate_index"],
            "before_metrics": metrics_before,
            "after_metrics": metrics_after,
            "delta_mean_travel_time": metrics_after["mean_travel_time"] - metrics_before["mean_travel_time"],
            "delta_p95_travel_time": metrics_after["p95_travel_time"] - metrics_before["p95_travel_time"],
            "delta_max_congestion_ratio": metrics_after["max_congestion_ratio"] - metrics_before["max_congestion_ratio"],
            "improvement_magnitude": {
                "mean": metrics_before["mean_travel_time"] - metrics_after["mean_travel_time"],
                "p95": metrics_before["p95_travel_time"] - metrics_after["p95_travel_time"],
                "max_congestion": metrics_before["max_congestion_ratio"] - metrics_after["max_congestion_ratio"],
            },
        }
        accepted_moves.append(accepted)
        history.append(
            {
                "iteration": iteration,
                "metrics_before": metrics_before,
                "selected_trip_count": len(selected),
                "candidate_moves_evaluated": candidate_moves_evaluated,
                "best_candidate_move": {
                    "trip_index": best["trip_index"],
                    "candidate_index": best["candidate_index"],
                },
                "accepted": True,
                "metrics_after": metrics_after,
                "stop_reason_if_any": None,
            }
        )
        if iteration == max_iterations:
            stop_reason = "max_iterations_reached"

    final_metrics = compute_network_metrics(env, routes, flows)
    _require(metrics_no_worse(final_metrics, initial_metrics), "local search worsened the network versus the start state")
    validate_state(env, trips, routes, flows)
    return {
        "algorithm": name,
        "seed": env.seed,
        "top_n": top_n,
        "max_iterations": max_iterations,
        "comparison_rule": COMPARISON_RULE,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "iterations_completed": iterations_completed,
        "accepted_move_count": len(accepted_moves),
        "stop_reason": stop_reason,
        "convergence_statement": (
            "local search converged with respect to the evaluated neighborhood"
            if stop_reason == "no_improving_move"
            else "local search stopped at the iteration limit; not a claim of global optimality"
        ),
        "initial_routes": initial_routes,
        "final_routes": routes,
        "initial_edge_flows": serialize_edge_map(initial_flows),
        "final_edge_flows": serialize_edge_map(flows),
        "accepted_moves": accepted_moves,
        "iteration_history": history,
        "improvement_summary": {
            "mean": initial_metrics["mean_travel_time"] - final_metrics["mean_travel_time"],
            "p95": initial_metrics["p95_travel_time"] - final_metrics["p95_travel_time"],
            "max_congestion": initial_metrics["max_congestion_ratio"] - final_metrics["max_congestion_ratio"],
        },
    }


def print_summary(section: dict) -> None:
    initial = section["initial_metrics"]
    final = section["final_metrics"]
    summary = section["improvement_summary"]
    print("algorithm:", section["algorithm"])
    print("initial mean:", initial["mean_travel_time"])
    print("final mean:", final["mean_travel_time"])
    print("initial p95:", initial["p95_travel_time"])
    print("final p95:", final["p95_travel_time"])
    print("initial max congestion:", initial["max_congestion_ratio"])
    print("final max congestion:", final["max_congestion_ratio"])
    print("iterations completed:", section["iterations_completed"])
    print("accepted moves:", section["accepted_move_count"])
    print("stop reason:", section["stop_reason"])
    print("total mean improvement:", summary["mean"])
    print("total p95 improvement:", summary["p95"])
    print("total max congestion improvement:", summary["max_congestion"])
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic one-move local search.")
    parser.add_argument("--algorithm", choices=sorted(SOLUTION_PATHS))
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require(args.top_n > 0, "top_n must be positive")
    _require(args.max_iterations > 0, "max_iterations must be positive")
    env = load_environment()
    names = [args.algorithm] if args.algorithm else ["baseline", "greedy"]
    analyses = {
        name: run_local_search(env, name, args.top_n, args.max_iterations) for name in names
    }
    payload = {
        "algorithm": ALGORITHM_NAME,
        "seed": env.seed,
        "top_n": args.top_n,
        "max_iterations": args.max_iterations,
        "comparison_rule": COMPARISON_RULE,
        **analyses,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    loaded = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    _require(isinstance(loaded, dict), "output/local_search.json is not valid JSON")
    for name in names:
        print_summary(analyses[name])
    print("wrote:", OUTPUT_PATH.as_posix())
    print("determinism: PASS")
    print("validation: PASS")


if __name__ == "__main__":
    main()
