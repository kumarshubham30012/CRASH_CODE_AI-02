"""Phase 15: optional robustness experiment on alternative deterministic trip seeds."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import networkx as nx

from algorithms.baseline import compute_baseline_routes
from algorithms.greedy import compute_greedy_routes
from algorithms.local_search import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TOP_N,
    apply_move,
    compute_trip_impacts,
    evaluate_move,
    reconstruct_flows,
    select_best_move,
    validate_state,
)
from bottlenecks import analyze_bottlenecks
from candidates import generate_candidates
from counterfactual import (
    _close,
    _require,
    compare_metrics,
    compute_network_metrics,
    validate_route,
)
from env import DEFAULT_NUM_TRIPS, DEFAULT_SEED, MIN_TRIP_DISTANCE, create_environment
from flow import compute_edge_flows

OUTPUT_DIR = Path("output") / "robustness"
RESULTS_PATH = OUTPUT_DIR / "results.json"
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
ROBUSTNESS_SEEDS = (20260912, 20260913, 20260914)
OFFICIAL_FILES = [
    Path("output") / "baseline.json",
    Path("output") / "greedy.json",
    Path("output") / "candidates.json",
    Path("output") / "bottlenecks.json",
    Path("output") / "impact.json",
    Path("output") / "counterfactual.json",
    Path("output") / "local_search.json",
    Path("output") / "optimizer.json",
    Path("comparison.csv"),
    Path("output") / "explanations.json",
    Path("output") / "explanations.txt",
]


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_official() -> dict[str, str]:
    return {str(path): file_digest(path) for path in OFFICIAL_FILES if path.exists()}


def assert_official_unchanged(before: dict[str, str]) -> None:
    after = snapshot_official()
    _require(after == before, "official Phase 0-14 outputs changed during robustness")
    env = create_environment()
    _require(env.seed == DEFAULT_SEED == 20260911, "official seed is no longer 20260911")


def validate_trips(env) -> None:
    _require(env.seed != DEFAULT_SEED, "robustness seed must not replace the official seed")
    _require(len(env.trips) == DEFAULT_NUM_TRIPS, f"seed {env.seed} did not generate 120 trips")
    distances = dict(nx.all_pairs_shortest_path_length(env.graph))
    for index, (origin, destination) in enumerate(env.trips):
        _require(origin != destination, f"seed {env.seed} trip {index} has origin == destination")
        _require(
            distances[origin][destination] >= MIN_TRIP_DISTANCE,
            f"seed {env.seed} trip {index} distance {distances[origin][destination]} < {MIN_TRIP_DISTANCE}",
        )


def evaluate_assignment(env, routes: list[list[int]], label: str) -> dict:
    trips = [[int(origin), int(destination)] for origin, destination in env.trips]
    _require(len(routes) == DEFAULT_NUM_TRIPS, f"{label} does not have 120 routes")
    for index, ((origin, destination), route) in enumerate(zip(trips, routes)):
        validate_route(env, origin, destination, route, f"{label} trip {index}")
    flows = compute_edge_flows(env.graph, routes)
    _require(sum(flows.values()) == sum(len(route) - 1 for route in routes), f"{label} flow reconstruction failed")
    metrics = compute_network_metrics(env, routes, flows)
    return {"routes": routes, "flows": flows, "metrics": metrics}


def run_local_search_from_state(env, routes: list[list[int]], candidates_by_trip: dict[int, list[list[int]]]) -> dict:
    trips = [[int(origin), int(destination)] for origin, destination in env.trips]
    flows = reconstruct_flows(env, routes)
    validate_state(env, trips, routes, flows)
    current_routes = [list(route) for route in routes]
    current_flows = dict(flows)
    accepted = 0
    stop_reason = "no_improving_move"
    iterations_completed = 0
    for iteration in range(1, DEFAULT_MAX_ITERATIONS + 1):
        iterations_completed = iteration
        analyze_bottlenecks(env, current_flows)
        impacts = compute_trip_impacts(env, trips, current_routes, current_flows)
        selected = impacts[:DEFAULT_TOP_N]
        metrics_before = compute_network_metrics(env, current_routes, current_flows)
        evaluated = []
        for impact_record in selected:
            trip_index = int(impact_record["trip_index"])
            official = candidates_by_trip[trip_index]
            for candidate_index, candidate_route in enumerate(official):
                move = evaluate_move(
                    env,
                    trips,
                    current_routes,
                    current_flows,
                    metrics_before,
                    trip_index,
                    candidate_index,
                    candidate_route,
                    official,
                )
                if move is not None:
                    evaluated.append(move)
        best = select_best_move(evaluated)
        if best is None:
            stop_reason = "no_improving_move"
            break
        current_routes, current_flows = apply_move(env, trips, current_routes, current_flows, best)
        accepted += 1
        if iteration == DEFAULT_MAX_ITERATIONS:
            stop_reason = "max_iterations_reached"
    metrics = compute_network_metrics(env, current_routes, current_flows)
    return {
        "metrics": metrics,
        "accepted_moves": accepted,
        "iterations": iterations_completed,
        "stop_reason": stop_reason,
        "routes": current_routes,
    }


def row(
    seed: int,
    algorithm: str,
    metrics: dict,
    accepted_moves: str | int = "N/A",
    iterations: str | int = "N/A",
    stop_reason: str = "N/A",
) -> dict:
    return {
        "seed": seed,
        "algorithm": algorithm,
        "trips": DEFAULT_NUM_TRIPS,
        "mean_travel_time": metrics["mean_travel_time"],
        "p95_travel_time": metrics["p95_travel_time"],
        "max_congestion_ratio": metrics["max_congestion_ratio"],
        "accepted_moves": accepted_moves,
        "iterations": iterations,
        "stop_reason": stop_reason,
    }


def better_metric(left: float, right: float) -> bool:
    return left < right - 1e-12


def summarize(results: list[dict]) -> dict:
    by_seed: dict[int, dict[str, dict]] = {}
    for item in results:
        by_seed.setdefault(item["seed"], {})[item["algorithm"]] = item

    greedy_mean = greedy_p95 = greedy_max = 0
    ls_mean = ls_lex = ls_max = 0
    deltas = []
    for seed in ROBUSTNESS_SEEDS:
        base = by_seed[seed]["baseline"]
        greedy = by_seed[seed]["greedy"]
        local = by_seed[seed]["local_search"]
        if better_metric(greedy["mean_travel_time"], base["mean_travel_time"]):
            greedy_mean += 1
        if better_metric(greedy["p95_travel_time"], base["p95_travel_time"]):
            greedy_p95 += 1
        if better_metric(greedy["max_congestion_ratio"], base["max_congestion_ratio"]):
            greedy_max += 1
        if better_metric(local["mean_travel_time"], base["mean_travel_time"]):
            ls_mean += 1
        better, _worse, _reason = compare_metrics(local, base)
        if better:
            ls_lex += 1
        if better_metric(local["max_congestion_ratio"], base["max_congestion_ratio"]):
            ls_max += 1
        deltas.append(
            {
                "seed": seed,
                "greedy_mean_delta": greedy["mean_travel_time"] - base["mean_travel_time"],
                "greedy_p95_delta": greedy["p95_travel_time"] - base["p95_travel_time"],
                "greedy_max_congestion_delta": greedy["max_congestion_ratio"] - base["max_congestion_ratio"],
                "local_search_mean_delta": local["mean_travel_time"] - base["mean_travel_time"],
                "local_search_p95_delta": local["p95_travel_time"] - base["p95_travel_time"],
                "local_search_max_congestion_delta": local["max_congestion_ratio"] - base["max_congestion_ratio"],
            }
        )

    def averages(algorithm: str) -> dict:
        items = [item for item in results if item["algorithm"] == algorithm]
        return {
            "n_seeds": len(items),
            "average_mean_travel_time": sum(item["mean_travel_time"] for item in items) / len(items),
            "average_p95_travel_time": sum(item["p95_travel_time"] for item in items) / len(items),
            "average_max_congestion_ratio": sum(item["max_congestion_ratio"] for item in items) / len(items),
        }

    n = len(ROBUSTNESS_SEEDS)
    conclusion_parts = [
        f"Across the three additional deterministic seeds {list(ROBUSTNESS_SEEDS)}, "
        f"greedy improved baseline mean on {greedy_mean}/{n} seeds, p95 on {greedy_p95}/{n}, "
        f"and max congestion on {greedy_max}/{n}.",
        f"Baseline-start local search improved baseline mean on {ls_mean}/{n} seeds "
        f"and the lexicographic objective on {ls_lex}/{n}.",
    ]
    stable = greedy_mean == n and greedy_p95 == n and ls_mean == n
    conclusion_parts.append(
        "The qualitative behavior was stable in this small robustness sample."
        if stable
        else "The qualitative behavior was not uniformly stable in this small robustness sample."
    )
    conclusion_parts.append(
        "This is not statistical significance, universal performance, or a global-optimum claim."
    )
    return {
        "by_algorithm": {
            "baseline": averages("baseline"),
            "greedy": averages("greedy"),
            "local_search": averages("local_search"),
        },
        "greedy_beats_baseline_mean": greedy_mean,
        "greedy_beats_baseline_p95": greedy_p95,
        "greedy_reduces_max_congestion": greedy_max,
        "local_search_beats_baseline_mean": ls_mean,
        "local_search_beats_baseline_lexicographic": ls_lex,
        "local_search_reduces_max_congestion": ls_max,
        "n_seeds": n,
        "pairwise_deltas": deltas,
        "qualitative_answers": {
            "A_greedy_mean": f"{greedy_mean}/{n}",
            "B_greedy_p95": f"{greedy_p95}/{n}",
            "C_greedy_max_congestion": f"{greedy_max}/{n}",
            "D_local_search_improves_baseline_mean": f"{ls_mean}/{n}",
            "E_ranking_stable": stable,
        },
        "conclusion": " ".join(conclusion_parts),
    }


def write_csv(results: list[dict]) -> None:
    fieldnames = [
        "seed",
        "algorithm",
        "trips",
        "mean_travel_time",
        "p95_travel_time",
        "max_congestion_ratio",
        "accepted_moves",
        "iterations",
        "stop_reason",
    ]
    with SUMMARY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "seed": item["seed"],
                    "algorithm": item["algorithm"],
                    "trips": item["trips"],
                    "mean_travel_time": repr(item["mean_travel_time"]),
                    "p95_travel_time": repr(item["p95_travel_time"]),
                    "max_congestion_ratio": repr(item["max_congestion_ratio"]),
                    "accepted_moves": item["accepted_moves"],
                    "iterations": item["iterations"],
                    "stop_reason": item["stop_reason"],
                }
            )


def print_report(summary: dict) -> None:
    answers = summary["qualitative_answers"]
    print("Phase 15: robustness")
    print("seeds tested: 3")
    print("algorithms: baseline, greedy, local_search")
    print("trips per seed: 120")
    print()
    print("Greedy beats baseline:")
    print("mean:", answers["A_greedy_mean"])
    print("p95:", answers["B_greedy_p95"])
    print("max congestion:", answers["C_greedy_max_congestion"])
    print()
    print("Local search improves baseline mean:")
    print(answers["D_local_search_improves_baseline_mean"])
    print()
    print("qualitative conclusion:")
    print(summary["conclusion"])


def main() -> None:
    before = snapshot_official()
    official = create_environment()
    _require(official.seed == 20260911, "create_environment() no longer uses the official seed")

    results: list[dict] = []
    for seed in ROBUSTNESS_SEEDS:
        env = create_environment(seed=seed)
        validate_trips(env)
        baseline_routes = compute_baseline_routes(env.graph, env.trips)
        baseline = evaluate_assignment(env, baseline_routes, f"seed {seed} baseline")
        greedy_routes, _live = compute_greedy_routes(env.graph, env.trips)
        greedy = evaluate_assignment(env, greedy_routes, f"seed {seed} greedy")
        candidate_records = generate_candidates(env)
        candidates_by_trip = {
            int(record["trip_index"]): [[int(node) for node in route] for route in record["candidates"]]
            for record in candidate_records
        }
        local = run_local_search_from_state(env, baseline["routes"], candidates_by_trip)
        evaluate_assignment(env, local["routes"], f"seed {seed} local_search")
        results.extend(
            [
                row(seed, "baseline", baseline["metrics"]),
                row(seed, "greedy", greedy["metrics"]),
                row(
                    seed,
                    "local_search",
                    local["metrics"],
                    accepted_moves=local["accepted_moves"],
                    iterations=local["iterations"],
                    stop_reason=local["stop_reason"],
                ),
            ]
        )

    summary = summarize(results)
    payload = {
        "official_seed": DEFAULT_SEED,
        "robustness_seeds": list(ROBUSTNESS_SEEDS),
        "algorithms": ["baseline", "greedy", "local_search"],
        "note": (
            "Optional robustness experiment. Official benchmark seed remains 20260911. "
            "Local search reuses Phase 10 helpers on in-memory robustness instances and "
            "does not overwrite official outputs."
        ),
        "results": results,
        "summary": summary,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(results)
    assert_official_unchanged(before)
    print_report(summary)
    print()
    print("wrote:", RESULTS_PATH.as_posix())
    print("wrote:", SUMMARY_PATH.as_posix())


if __name__ == "__main__":
    main()
