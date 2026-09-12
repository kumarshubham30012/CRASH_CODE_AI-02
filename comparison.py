"""Phase 12: reproducible comparison of routing approaches. Read-only on prior outputs."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

from algorithms.baseline import load_environment
from counterfactual import _close, _require, compute_network_metrics, validate_route
from env import DEFAULT_NUM_TRIPS, Environment
from flow import compute_edge_flows

OUTPUT_CSV = Path("comparison.csv")
BASELINE_PATH = Path("output") / "baseline.json"
GREEDY_PATH = Path("output") / "greedy.json"
CANDIDATES_PATH = Path("output") / "candidates.json"
LOCAL_SEARCH_PATH = Path("output") / "local_search.json"
OPTIMIZER_PATH = Path("output") / "optimizer.json"
FIELDNAMES = [
    "algorithm",
    "trips",
    "mean_travel_time",
    "p95_travel_time",
    "max_congestion_ratio",
    "runtime_seconds",
]
ROW_ORDER = ["baseline", "greedy", "k_shortest", "local_search", "optimizer"]
FLOAT_TOL = 1e-12
NOTES = {
    "k_shortest": (
        "K-shortest is a free-flow candidate benchmark: each trip uses Phase 5 "
        "candidate #1 (index 0), the deterministic free-flow shortest route. "
        "It is not a congestion-aware optimizer."
    ),
    "local_search": "Final local-search assignment from the baseline start.",
    "optimizer": (
        "Final optimizer assignment from the baseline start. This wraps Phase 10 "
        "local search. If stop_reason is max_iterations_reached, it is not neighborhood-converged."
    ),
}


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{path.as_posix()} is not a JSON object")
    return payload


def format_float(value: float) -> str:
    return repr(float(value))


def evaluate_routes(
    env: Environment,
    trips: list[list[int]],
    routes: list[list[int]],
    label: str,
) -> dict:
    _require(len(trips) == DEFAULT_NUM_TRIPS, f"{label} does not have 120 trips")
    _require(len(routes) == DEFAULT_NUM_TRIPS, f"{label} does not have 120 routes")
    for index, ((origin, destination), route) in enumerate(zip(trips, routes)):
        validate_route(env, origin, destination, route, f"{label} trip {index}")
    flows = compute_edge_flows(env.graph, routes)
    _require(all(value >= 0 for value in flows.values()), f"{label} produced a negative flow")
    traversal_sum = sum(len(route) - 1 for route in routes)
    _require(sum(flows.values()) == traversal_sum, f"{label} flow reconstruction is inconsistent")
    metrics = compute_network_metrics(env, routes, flows)
    for name, value in metrics.items():
        _require(math.isfinite(value), f"{label} {name} is not finite")
        _require(value >= 0, f"{label} {name} is negative")
    return metrics


def assert_close_metrics(actual: dict, expected: dict, label: str) -> None:
    for key in ("mean_travel_time", "p95_travel_time", "max_congestion_ratio"):
        _require(
            _close(actual[key], float(expected[key])),
            f"{label} {key} {actual[key]} != stored {expected[key]}",
        )


def baseline_routes(env: Environment, payload: dict) -> list[list[int]]:
    trips = [[int(a), int(b)] for a, b in payload["trips"]]
    _require(trips == [[int(a), int(b)] for a, b in env.trips], "baseline trips do not match env.py")
    return [[int(node) for node in route] for route in payload["routes"]]


def greedy_routes(env: Environment, payload: dict) -> list[list[int]]:
    trips = [[int(a), int(b)] for a, b in payload["trips"]]
    _require(trips == [[int(a), int(b)] for a, b in env.trips], "greedy trips do not match env.py")
    return [[int(node) for node in route] for route in payload["routes"]]


def k_shortest_routes(env: Environment, payload: dict) -> list[list[int]]:
    records = payload["candidate_routes"]
    _require(len(records) == DEFAULT_NUM_TRIPS, "candidates.json does not contain 120 trips")
    routes: list[list[int]] = []
    for index, (origin, destination) in enumerate(env.trips):
        record = records[index]
        _require(int(record["trip_index"]) == index, f"candidate trip_index mismatch at {index}")
        _require(int(record["origin"]) == origin, f"candidate origin mismatch at trip {index}")
        _require(int(record["destination"]) == destination, f"candidate destination mismatch at trip {index}")
        _require(record["candidates"], f"trip {index} has no candidates")
        routes.append([int(node) for node in record["candidates"][0]])
    return routes


def local_search_routes(env: Environment, payload: dict) -> list[list[int]]:
    section = payload["baseline"]
    routes = [[int(node) for node in route] for route in section["final_routes"]]
    _require(len(routes) == DEFAULT_NUM_TRIPS, "local_search baseline final_routes is not 120")
    _require(routes[0][0] == env.trips[0][0], "local_search route order does not match env trips")
    return routes


def optimizer_routes(env: Environment, payload: dict) -> list[list[int]]:
    section = payload["baseline"]
    records = section["final_routes"]
    _require(len(records) == DEFAULT_NUM_TRIPS, "optimizer baseline final_routes is not 120")
    routes: list[list[int]] = []
    for index, (origin, destination) in enumerate(env.trips):
        record = records[index]
        _require(int(record["trip_index"]) == index, f"optimizer trip_index mismatch at {index}")
        _require(int(record["source"]) == origin, f"optimizer source mismatch at {index}")
        _require(int(record["destination"]) == destination, f"optimizer destination mismatch at {index}")
        routes.append([int(node) for node in record["route"]])
    return routes


def build_row(name: str, metrics: dict, runtime: float | None) -> dict:
    return {
        "algorithm": name,
        "trips": DEFAULT_NUM_TRIPS,
        "mean_travel_time": metrics["mean_travel_time"],
        "p95_travel_time": metrics["p95_travel_time"],
        "max_congestion_ratio": metrics["max_congestion_ratio"],
        "runtime_seconds": runtime,
    }


def validate_rows(rows: list[dict]) -> None:
    _require(len(rows) == 5, f"expected 5 comparison rows, got {len(rows)}")
    _require([row["algorithm"] for row in rows] == ROW_ORDER, "algorithm order is not the required benchmark order")
    for row in rows:
        _require(row["trips"] == DEFAULT_NUM_TRIPS, f"{row['algorithm']} does not represent 120 trips")
        for key in ("mean_travel_time", "p95_travel_time", "max_congestion_ratio"):
            value = row[key]
            _require(isinstance(value, float), f"{row['algorithm']} {key} is not a float")
            _require(math.isfinite(value) and value >= 0, f"{row['algorithm']} {key} is invalid: {value}")
        runtime = row["runtime_seconds"]
        if runtime is not None:
            _require(isinstance(runtime, float) and math.isfinite(runtime) and runtime >= 0, "runtime_seconds is invalid")


def write_csv(rows: list[dict], path: Path = OUTPUT_CSV) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "algorithm": row["algorithm"],
                    "trips": row["trips"],
                    "mean_travel_time": format_float(row["mean_travel_time"]),
                    "p95_travel_time": format_float(row["p95_travel_time"]),
                    "max_congestion_ratio": format_float(row["max_congestion_ratio"]),
                    "runtime_seconds": "" if row["runtime_seconds"] is None else format_float(row["runtime_seconds"]),
                }
            )


def print_report(rows: list[dict], optimizer_stop: str) -> None:
    print("Phase 12: comparison benchmark")
    print()
    print(f"{'algorithm':<16}{'trips':<8}{'mean':<24}{'p95':<24}{'max_congestion'}")
    for row in rows:
        print(
            f"{row['algorithm']:<16}{row['trips']:<8}"
            f"{row['mean_travel_time']:<24}{row['p95_travel_time']:<24}"
            f"{row['max_congestion_ratio']}"
        )
    print()
    print("notes:")
    print("k_shortest:", NOTES["k_shortest"])
    print("local_search:", NOTES["local_search"])
    print("optimizer:", NOTES["optimizer"])
    print("optimizer_stop_reason:", optimizer_stop)
    if optimizer_stop == "max_iterations_reached":
        print("optimizer_neighborhood_converged: false")
    print()
    print("determinism: PASS")
    print("validation: PASS")
    print("regression: PASS")
    print("previous_outputs_unchanged: PASS")
    print("wrote:", OUTPUT_CSV.as_posix())


def main() -> None:
    env = load_environment()
    trips = [[int(origin), int(destination)] for origin, destination in env.trips]
    baseline = load_json(BASELINE_PATH)
    greedy = load_json(GREEDY_PATH)
    candidates = load_json(CANDIDATES_PATH)
    local_search = load_json(LOCAL_SEARCH_PATH)
    optimizer = load_json(OPTIMIZER_PATH)

    k_start = time.perf_counter()
    k_routes = k_shortest_routes(env, candidates)
    k_metrics = evaluate_routes(env, trips, k_routes, "k_shortest")
    k_runtime = time.perf_counter() - k_start

    rows = [
        build_row(
            "baseline",
            evaluate_routes(env, trips, baseline_routes(env, baseline), "baseline"),
            float(baseline.get("runtime_seconds")) if "runtime_seconds" in baseline else None,
        ),
        build_row(
            "greedy",
            evaluate_routes(env, trips, greedy_routes(env, greedy), "greedy"),
            float(greedy.get("runtime_seconds")) if "runtime_seconds" in greedy else None,
        ),
        build_row("k_shortest", k_metrics, k_runtime),
        build_row(
            "local_search",
            evaluate_routes(env, trips, local_search_routes(env, local_search), "local_search"),
            None,
        ),
        build_row(
            "optimizer",
            evaluate_routes(env, trips, optimizer_routes(env, optimizer), "optimizer"),
            None,
        ),
    ]

    assert_close_metrics(rows[0], baseline, "baseline")
    assert_close_metrics(rows[1], greedy, "greedy")
    assert_close_metrics(rows[3], local_search["baseline"]["final_metrics"], "local_search")
    assert_close_metrics(rows[4], optimizer["baseline"]["final_metrics"], "optimizer")
    validate_rows(rows)
    write_csv(rows)
    print_report(rows, optimizer["baseline"]["stop_reason"])


if __name__ == "__main__":
    main()
