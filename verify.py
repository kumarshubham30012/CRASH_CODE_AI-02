"""Phase 3 verification gate for the official environment and Phase 2 baseline."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np

from algorithms.baseline import (
    DETERMINISTIC_BENCHMARK_KEYS,
    OUTPUT_PATH,
    REQUIRED_OUTPUT_KEYS,
    RUNTIME_FIELD,
    extract_deterministic_benchmark,
    run_baseline,
)
from env import (
    DEFAULT_NUM_TRIPS,
    DEFAULT_SEED,
    MIN_TRIP_DISTANCE,
    REMOVED_COORD_PAIRS,
    create_environment,
    node_id,
)
from flow import (
    CAPACITY_ATTR,
    FREE_FLOW_ATTR,
    congested_travel_time,
    max_congestion_ratio,
    parse_edge_key,
)

FLOAT_REL_TOL = 1e-9
FLOAT_ABS_TOL = 1e-9


class VerificationError(Exception):
    pass


def _fail(check: str, message: str) -> None:
    raise VerificationError(f"{check}: {message}")


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL)


def disrupted_edges() -> set[tuple[int, int]]:
    return {
        (node_id(x1, y1), node_id(x2, y2))
        for (x1, y1), (x2, y2) in REMOVED_COORD_PAIRS
    }


def load_baseline(path: Path = OUTPUT_PATH) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
    except FileNotFoundError as exc:
        _fail("JSON validity", f"{path.as_posix()} does not exist")
        raise exc
    except json.JSONDecodeError as exc:
        _fail("JSON validity", f"{path.as_posix()} is not valid JSON: {exc}")
        raise exc
    if not isinstance(payload, dict):
        _fail("JSON validity", f"{path.as_posix()} is not a JSON object")
    return payload


def assert_finite_json(value: object, location: str = "$") -> None:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            _fail("JSON validity", f"non-finite number at {location}: {value}")
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_finite_json(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite_json(item, f"{location}[{index}]")


def check_json_schema(baseline: dict) -> None:
    assert_finite_json(baseline)
    missing = [key for key in REQUIRED_OUTPUT_KEYS if key not in baseline]
    if missing:
        _fail("JSON validity", f"missing required fields: {missing}")
    if baseline["number_of_trips"] != DEFAULT_NUM_TRIPS:
        _fail("JSON validity", f"number_of_trips is {baseline['number_of_trips']}, expected {DEFAULT_NUM_TRIPS}")
    if baseline["number_of_routes"] != DEFAULT_NUM_TRIPS:
        _fail("JSON validity", f"number_of_routes is {baseline['number_of_routes']}, expected {DEFAULT_NUM_TRIPS}")
    if len(baseline.get("routes", [])) != DEFAULT_NUM_TRIPS:
        _fail("JSON validity", f"routes list length is {len(baseline.get('routes', []))}")
    if len(baseline.get("trip_travel_times", [])) != DEFAULT_NUM_TRIPS:
        _fail("JSON validity", f"trip_travel_times length is {len(baseline.get('trip_travel_times', []))}")


def check_environment(env) -> None:
    if env.graph.number_of_nodes() != 25:
        _fail("node count", f"found {env.graph.number_of_nodes()} nodes, expected 25")
    if env.graph.number_of_edges() != 78:
        _fail("directed edge count", f"found {env.graph.number_of_edges()} directed edges, expected 78")
    if len(env.trips) != DEFAULT_NUM_TRIPS:
        _fail("trip count", f"found {len(env.trips)} trips, expected {DEFAULT_NUM_TRIPS}")
    if env.seed != DEFAULT_SEED:
        _fail("seed", f"environment seed is {env.seed}, expected {DEFAULT_SEED}")

    blocked = disrupted_edges()
    present = [edge for edge in blocked if env.graph.has_edge(*edge)]
    if present:
        _fail("disrupted graph", f"disrupted edges still present: {present}")

    distances = dict(nx.all_pairs_shortest_path_length(env.graph))
    for index, (origin, destination) in enumerate(env.trips):
        if origin == destination:
            _fail("trip distance constraint", f"trip {index} has origin == destination ({origin})")
        try:
            distance = distances[origin][destination]
        except KeyError:
            _fail(
                "trip distance constraint",
                f"trip {index} ({origin} -> {destination}) is disconnected on the disrupted graph",
            )
        if distance < MIN_TRIP_DISTANCE:
            _fail(
                "trip distance constraint",
                f"trip {index} ({origin} -> {destination}) has distance {distance} < {MIN_TRIP_DISTANCE}",
            )


def check_official_trips_present(env, baseline: dict) -> None:
    saved_trips = [tuple(trip) for trip in baseline.get("trips", [])]
    env_trips = [(int(origin), int(destination)) for origin, destination in env.trips]
    if saved_trips != env_trips:
        _fail(
            "trip count",
            "baseline.json trips do not match the official env.py trip list",
        )


def check_routes(env, baseline: dict) -> list[list[int]]:
    trips = [(int(origin), int(destination)) for origin, destination in env.trips]
    routes = [[int(node) for node in route] for route in baseline["routes"]]
    if len(routes) != DEFAULT_NUM_TRIPS:
        _fail("route count", f"found {len(routes)} routes, expected {DEFAULT_NUM_TRIPS}")
    if len(routes) != len(trips):
        _fail("route count", "route count does not match trip count")

    blocked = disrupted_edges()
    for index, (trip, route) in enumerate(zip(trips, routes)):
        origin, destination = trip
        if len(route) < 2:
            _fail("route endpoints", f"trip {index} route has fewer than two nodes: {route}")
        if route[0] != origin:
            _fail("route endpoints", f"trip {index} starts at {route[0]}, expected origin {origin}")
        if route[-1] != destination:
            _fail("route endpoints", f"trip {index} ends at {route[-1]}, expected destination {destination}")
        if len(route) != len(set(route)):
            _fail("route simplicity", f"trip {index} repeats a node: {route}")
        for u, v in zip(route, route[1:]):
            if not env.graph.has_edge(u, v):
                _fail(
                    "route edge validity",
                    f"trip {index} uses missing directed edge {u}->{v}",
                )
            if (u, v) in blocked:
                _fail(
                    "disrupted edge exclusion",
                    f"trip {index} uses disrupted edge {u}->{v}",
                )
    return routes


def reconstruct_flows(env, routes: list[list[int]]) -> dict[tuple[int, int], int]:
    flows = {(u, v): 0 for u, v in env.graph.edges()}
    for route in routes:
        for u, v in zip(route, route[1:]):
            if (u, v) not in flows:
                _fail("edge flow reconstruction", f"route uses edge {u}->{v} absent from the graph")
            flows[(u, v)] += 1
    return flows


def parse_saved_flows(baseline: dict) -> dict[tuple[int, int], int]:
    saved: dict[tuple[int, int], int] = {}
    for key, value in baseline["edge_flows"].items():
        edge = parse_edge_key(key)
        saved[edge] = int(value)
    return saved


def check_flows(env, baseline: dict, routes: list[list[int]]) -> dict[tuple[int, int], int]:
    reconstructed = reconstruct_flows(env, routes)
    saved = parse_saved_flows(baseline)

    extra = set(saved) - set(reconstructed)
    missing = set(reconstructed) - set(saved)
    if extra or missing:
        _fail(
            "edge flow reconstruction",
            f"edge set mismatch; extra={sorted(extra)[:10]} missing={sorted(missing)[:10]}",
        )
    mismatches = [
        (edge, reconstructed[edge], saved[edge])
        for edge in reconstructed
        if reconstructed[edge] != saved[edge]
    ]
    if mismatches:
        sample = mismatches[:5]
        _fail("edge flow reconstruction", f"flow counts differ from routes, sample={sample}")

    traversal_sum = sum(len(route) - 1 for route in routes)
    flow_sum = sum(reconstructed.values())
    if traversal_sum != flow_sum:
        _fail(
            "flow conservation",
            f"sum(edge flows)={flow_sum} != total traversals={traversal_sum}",
        )
    return reconstructed


def calculate_expected_trip_times(
    env,
    routes: list[list[int]],
    flows: dict[tuple[int, int], int],
) -> list[float]:
    trip_times: list[float] = []
    for route in routes:
        total = 0.0
        for u, v in zip(route, route[1:]):
            data = env.graph[u][v]
            total += congested_travel_time(
                data[FREE_FLOW_ATTR],
                flows[(u, v)],
                data[CAPACITY_ATTR],
            )
        trip_times.append(total)
    return trip_times


def calculate_expected_metrics(
    env,
    flows: dict[tuple[int, int], int],
    trip_times: list[float],
) -> tuple[float, float, float]:
    times = np.asarray(trip_times, dtype=float)
    mean_travel_time = float(np.mean(times))
    p95_travel_time = float(np.percentile(times, 95, method="linear"))
    max_ratio = float(max_congestion_ratio(env.graph, flows))
    return mean_travel_time, p95_travel_time, max_ratio


def check_trip_times_and_metrics(
    env,
    baseline: dict,
    routes: list[list[int]],
    flows: dict[tuple[int, int], int],
) -> None:
    expected_times = calculate_expected_trip_times(env, routes, flows)
    saved_times = [float(value) for value in baseline["trip_travel_times"]]
    if len(saved_times) != len(expected_times):
        _fail(
            "trip travel times",
            f"saved {len(saved_times)} times, independently calculated {len(expected_times)}",
        )
    for index, (expected, saved) in enumerate(zip(expected_times, saved_times)):
        if not math.isfinite(expected) or not math.isfinite(saved) or not _close(expected, saved):
            _fail(
                "trip travel times",
                f"trip {index} expected {expected}, baseline.json has {saved}",
            )

    expected_mean, expected_p95, expected_max_ratio = calculate_expected_metrics(
        env, flows, expected_times
    )
    if not _close(expected_mean, float(baseline["mean_travel_time"])):
        _fail(
            "mean travel time",
            f"expected {expected_mean}, baseline.json has {baseline['mean_travel_time']}",
        )
    if not _close(expected_p95, float(baseline["p95_travel_time"])):
        _fail(
            "p95 travel time",
            f"expected {expected_p95}, baseline.json has {baseline['p95_travel_time']}",
        )
    if not _close(expected_max_ratio, float(baseline["max_congestion_ratio"])):
        _fail(
            "max congestion ratio",
            f"expected {expected_max_ratio}, baseline.json has {baseline['max_congestion_ratio']}",
        )


def check_reproducibility(env, baseline: dict) -> None:
    if baseline.get("seed") != DEFAULT_SEED:
        _fail("reproducibility", f"saved seed is {baseline.get('seed')}, expected {DEFAULT_SEED}")

    first = create_environment()
    second = create_environment()
    if first.trips != second.trips or first.trips != list(env.trips):
        _fail("reproducibility", "official trip list is not deterministic across env.py recreation")

    rerun = run_baseline(env)
    saved_deterministic = extract_deterministic_benchmark(baseline)
    rerun_deterministic = extract_deterministic_benchmark(rerun)
    if saved_deterministic != rerun_deterministic:
        mismatched = [
            key
            for key in DETERMINISTIC_BENCHMARK_KEYS
            if saved_deterministic[key] != rerun_deterministic[key]
        ]
        _fail(
            "reproducibility",
            f"baseline re-run differs from baseline.json on non-runtime fields: {mismatched}",
        )


def main() -> int:
    print("========================================")
    print("AI-02 PHASE 3 VERIFICATION")
    print("========================================")
    print()

    passed: list[str] = []

    def pass_check(name: str) -> None:
        passed.append(name)
        print(f"[PASS] {name}")

    try:
        env = create_environment()
        check_environment(env)
        pass_check("node count")
        pass_check("directed edge count")
        pass_check("trip count")
        pass_check("seed")
        pass_check("disrupted graph")
        pass_check("trip distance constraint")

        baseline = load_baseline()
        check_json_schema(baseline)
        check_official_trips_present(env, baseline)
        pass_check("JSON validity")

        routes = check_routes(env, baseline)
        pass_check("route count")
        pass_check("route endpoints")
        pass_check("route simplicity")
        pass_check("route edge validity")
        pass_check("disrupted edge exclusion")

        flows = check_flows(env, baseline, routes)
        pass_check("edge flow reconstruction")
        pass_check("flow conservation")

        check_trip_times_and_metrics(env, baseline, routes, flows)
        pass_check("trip travel times")
        pass_check("mean travel time")
        pass_check("p95 travel time")
        pass_check("max congestion ratio")

        check_reproducibility(env, baseline)
        pass_check("reproducibility")
    except VerificationError as exc:
        print(f"[FAIL] {exc}")
        print()
        print("========================================")
        print("PHASE 3 VERIFICATION FAILED")
        print("========================================")
        return 1

    print()
    print("========================================")
    print("ALL PHASE 3 CHECKS PASSED")
    print("========================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
