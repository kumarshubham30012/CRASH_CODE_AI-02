"""Phase 14: deterministic explanations of baseline-start local-search reroutes."""

from __future__ import annotations

import json
import math
from pathlib import Path

from algorithms.baseline import load_environment
from counterfactual import _require, validate_route
from env import DEFAULT_NUM_TRIPS, Environment
from flow import CAPACITY_ATTR, edge_key, route_edges

OUTPUT_JSON = Path("output") / "explanations.json"
OUTPUT_TXT = Path("output") / "explanations.txt"
OPTIMIZER_PATH = Path("output") / "optimizer.json"
BOTTLENECKS_PATH = Path("output") / "bottlenecks.json"
IMPACT_PATH = Path("output") / "impact.json"
COUNTERFACTUAL_PATH = Path("output") / "counterfactual.json"
TOP_N = 10
OBJECTIVE = ["mean_travel_time", "p95_travel_time", "max_congestion_ratio"]


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{path.as_posix()} is not a JSON object")
    return payload


def finite(value: float, name: str) -> float:
    number = float(value)
    _require(math.isfinite(number), f"{name} is not finite")
    return number


def format_route(route: list[int]) -> str:
    return " → ".join(str(node) for node in route)


def bottleneck_lookup(records: list[dict]) -> dict[tuple[int, int], dict]:
    return {(int(item["source"]), int(item["destination"])): item for item in records}


def route_bottlenecks(route: list[int], lookup: dict[tuple[int, int], dict]) -> list[dict]:
    found: list[dict] = []
    for u, v in route_edges(route):
        record = lookup.get((int(u), int(v)))
        if record is None:
            continue
        found.append(
            {
                "edge": edge_key(u, v),
                "flow": record["flow"],
                "capacity": record["capacity"],
                "utilization": record["utilization"],
                "rank": record["rank"],
                "status": "overloaded" if record["utilization"] > 1.0 else "within_capacity",
            }
        )
    found.sort(key=lambda item: (item["rank"], item["edge"]))
    return found


def describe_bottlenecks(items: list[dict]) -> str:
    if not items:
        return (
            "Phase 7 did not list any of the replaced-route edges among the initial "
            "baseline bottleneck records."
        )
    overloaded = [item for item in items if item["status"] == "overloaded"]
    parts = []
    if overloaded:
        details = ", ".join(
            f"{item['edge']} (utilization {item['utilization']}, rank {item['rank']})"
            for item in overloaded
        )
        parts.append(
            "On the initial baseline network, the replaced route used overloaded edges "
            f"with utilization > 1: {details}."
        )
    within = [item for item in items if item["status"] == "within_capacity"]
    if within:
        details = ", ".join(
            f"{item['edge']} (utilization {item['utilization']})"
            for item in within[:3]
        )
        parts.append(
            "Other replaced-route edges were within capacity on the initial baseline "
            f"assignment, including {details}."
        )
    return " ".join(parts)


def describe_impact(impact: dict | None, trip_index: int) -> str:
    if impact is None:
        return f"Phase 8 has no baseline impact record for trip {trip_index}."
    return (
        f"On the initial baseline assignment, Phase 8 ranked this trip at impact rank "
        f"{impact['rank']} with impact_score={impact['impact_score']}, "
        f"critical_edge_impact={impact['critical_edge_impact']}, "
        f"max_edge_utilization={impact['max_edge_utilization']}, and "
        f"critical_edge_count={impact['critical_edge_count']}. "
        "Those scores measure use of high-utilization edges; they do not prove that "
        "an alternative existed."
    )


def describe_counterfactual(cf: dict | None, candidate_index: int) -> str:
    if cf is None:
        return (
            "Phase 9 did not evaluate this trip in its default top-N counterfactual set, "
            "so no Phase 9 delta is attached. The move was accepted later by Phase 10 "
            "local search using the same lexicographic network objective."
        )
    match = next((item for item in cf["candidates"] if item["candidate_index"] == candidate_index), None)
    if match is None:
        return (
            f"Phase 9 evaluated trip {cf['trip_index']}, but candidate {candidate_index} "
            "was not listed in that snapshot."
        )
    return (
        f"Phase 9 evaluated candidate {candidate_index} for this trip. "
        f"is_current_route={match['is_current_route']}, "
        f"candidate_is_better={match['candidate_is_better']}, "
        f"delta_mean={match['delta_mean_travel_time']}, "
        f"delta_p95={match['delta_p95_travel_time']}, "
        f"delta_max_congestion={match['delta_max_congestion_ratio']}. "
        f"Comparison reason: {match['comparison_reason']}."
    )


def describe_reason(trip_index: int, origin: int, destination: int, bottlenecks: list[dict], impact: dict | None) -> str:
    pieces = [
        f"Trip {trip_index} travels from {origin} to {destination}."
    ]
    if impact is not None:
        pieces.append(
            f"It was a high-priority trip in the initial baseline impact ranking "
            f"(rank {impact['rank']}, critical_edge_impact={impact['critical_edge_impact']})."
        )
    overloaded = [item for item in bottlenecks if item["status"] == "overloaded"]
    if overloaded:
        worst = overloaded[0]
        pieces.append(
            f"Its replaced route used high-utilization edge {worst['edge']} "
            f"(flow {worst['flow']} / capacity {worst['capacity']} = {worst['utilization']})."
        )
    else:
        pieces.append(
            "The replaced route is recorded in the accepted-move history; Phase 7's "
            "initial baseline snapshot does not show overloaded edges on that path."
        )
    return " ".join(pieces)


def describe_effect(move: dict) -> str:
    before = move["before_metrics"]
    after = move["after_metrics"]
    mag = move["improvement_magnitude"]
    return (
        "After substituting the candidate, the evaluated complete-network objective changed "
        f"from mean={before['mean_travel_time']}, p95={before['p95_travel_time']}, "
        f"max congestion={before['max_congestion_ratio']} to mean={after['mean_travel_time']}, "
        f"p95={after['p95_travel_time']}, max congestion={after['max_congestion_ratio']}. "
        f"Measured improvements (before minus after) were mean={mag['mean']}, "
        f"p95={mag['p95']}, max congestion={mag['max_congestion']}. "
        "These are one-move local-search results, not a claim that traffic was eliminated."
    )


def describe_decision(move: dict) -> str:
    return (
        f"Candidate {move['candidate_index']} was accepted as the best evaluated move in "
        f"iteration {move['iteration']} because the resulting "
        "(mean travel time, p95 travel time, max congestion ratio) tuple was lexicographically "
        "better than the current assignment. Lower is better. This does not prove a global optimum."
    )


def build_text(payload: dict) -> str:
    lines = [
        "CRASH_CODE(AI-02) Phase 14 explanations",
        f"algorithm: {payload['algorithm']}",
        f"stop_reason: {payload['stop_reason']}",
        f"converged: {payload['converged']}",
        "",
        payload["context"],
        "",
    ]
    for item in payload["explanations"]:
        lines.extend(
            [
                f"Trip {item['trip_index']} ({item['origin']} → {item['destination']}):",
                "",
                f"Reason: {item['reason']}",
                "",
                (
                    f"Change: Iteration {item['iteration']} replaced the original route with "
                    f"candidate {item['candidate_index']}:"
                ),
                f"old route {format_route(item['old_route'])}",
                f"new route {format_route(item['new_route'])}",
                "",
                f"Bottlenecks: {item['bottleneck_summary']}",
                f"Impact: {item['impact_summary']}",
                f"Counterfactual: {item['counterfactual_summary']}",
                f"Effect: {item['effect']}",
                f"Decision: {item['decision']}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def validate_explanations(env: Environment, payload: dict, moves: list[dict]) -> None:
    explanations = payload["explanations"]
    _require(len(explanations) <= TOP_N, "too many explanations")
    _require(payload["stop_reason"] == "max_iterations_reached", "stop_reason does not match optimizer output")
    _require(payload["converged"] is False, "baseline-start run must not be marked converged")
    _require(payload["objective"] == OBJECTIVE, "objective order is incorrect")
    accepted = {(move["iteration"], move["trip_index"], move["candidate_index"]) for move in moves}
    for item in explanations:
        trip_index = item["trip_index"]
        _require(0 <= trip_index < DEFAULT_NUM_TRIPS, f"invalid trip_index {trip_index}")
        origin = item["origin"]
        destination = item["destination"]
        validate_route(env, origin, destination, item["old_route"], f"trip {trip_index} old route")
        validate_route(env, origin, destination, item["new_route"], f"trip {trip_index} new route")
        _require(isinstance(item["candidate_index"], int) and item["candidate_index"] >= 0, "invalid candidate_index")
        key = (item["iteration"], trip_index, item["candidate_index"])
        _require(key in accepted, f"explanation does not match an accepted move: {key}")
        for name in (
            "old_mean_travel_time",
            "new_mean_travel_time",
            "mean_improvement",
            "old_p95_travel_time",
            "new_p95_travel_time",
            "p95_improvement",
            "old_max_congestion",
            "new_max_congestion",
            "max_congestion_improvement",
        ):
            finite(item[name], name)
        _require(item["decision_flag"] == "accepted", "explained move is not marked accepted")
        _require(item["reason"] and item["effect"] and item["decision"], "missing explanation text")


def main() -> None:
    env = load_environment()
    optimizer = load_json(OPTIMIZER_PATH)["baseline"]
    bottlenecks = bottleneck_lookup(load_json(BOTTLENECKS_PATH)["baseline"]["edges"])
    impact_by_trip = {
        int(record["trip_index"]): record for record in load_json(IMPACT_PATH)["baseline"]["trips"]
    }
    counterfactual_by_trip = {
        int(record["trip_index"]): record
        for record in load_json(COUNTERFACTUAL_PATH)["baseline"]["trip_results"]
    }
    moves = optimizer["accepted_moves"]
    selected = moves[:TOP_N]
    explanations = []
    for move in selected:
        trip_index = int(move["trip_index"])
        origin = int(move["source"])
        destination = int(move["destination"])
        old_route = [int(node) for node in move["old_route"]]
        new_route = [int(node) for node in move["new_route"]]
        validate_route(env, origin, destination, old_route, f"trip {trip_index} old route")
        validate_route(env, origin, destination, new_route, f"trip {trip_index} new route")
        bn = route_bottlenecks(old_route, bottlenecks)
        impact = impact_by_trip.get(trip_index)
        cf = counterfactual_by_trip.get(trip_index)
        before = move["before_metrics"]
        after = move["after_metrics"]
        mag = move["improvement_magnitude"]
        explanations.append(
            {
                "trip_index": trip_index,
                "origin": origin,
                "destination": destination,
                "iteration": int(move["iteration"]),
                "candidate_index": int(move["candidate_index"]),
                "old_route": old_route,
                "new_route": new_route,
                "old_mean_travel_time": finite(before["mean_travel_time"], "old mean"),
                "new_mean_travel_time": finite(after["mean_travel_time"], "new mean"),
                "mean_improvement": finite(mag["mean"], "mean improvement"),
                "old_p95_travel_time": finite(before["p95_travel_time"], "old p95"),
                "new_p95_travel_time": finite(after["p95_travel_time"], "new p95"),
                "p95_improvement": finite(mag["p95"], "p95 improvement"),
                "old_max_congestion": finite(before["max_congestion_ratio"], "old max congestion"),
                "new_max_congestion": finite(after["max_congestion_ratio"], "new max congestion"),
                "max_congestion_improvement": finite(mag["max_congestion"], "max congestion improvement"),
                "bottlenecks": bn,
                "bottleneck_summary": describe_bottlenecks(bn),
                "impact": None
                if impact is None
                else {
                    "rank": impact["rank"],
                    "impact_score": impact["impact_score"],
                    "critical_edge_impact": impact["critical_edge_impact"],
                    "max_edge_utilization": impact["max_edge_utilization"],
                    "critical_edge_count": impact["critical_edge_count"],
                },
                "impact_summary": describe_impact(impact, trip_index),
                "counterfactual": None
                if cf is None
                else {
                    "impact_rank": cf["impact_rank"],
                    "best_candidate": cf["best_candidate"],
                    "genuine_improvement": cf["genuine_improvement"],
                },
                "counterfactual_summary": describe_counterfactual(cf, int(move["candidate_index"])),
                "reason": describe_reason(trip_index, origin, destination, bn, impact),
                "effect": describe_effect(move),
                "decision": describe_decision(move),
                "decision_flag": "accepted",
            }
        )

    payload = {
        "algorithm": "baseline_start_local_search",
        "top_n": TOP_N,
        "objective": OBJECTIVE,
        "stop_reason": optimizer["stop_reason"],
        "converged": False,
        "accepted_move_count": optimizer["accepted_move_count"],
        "initial_metrics": optimizer["initial_metrics"],
        "final_metrics": optimizer["final_metrics"],
        "context": (
            "Baseline-start Phase 10 local search, reported by the Phase 11 optimizer wrapper, "
            f"accepted {optimizer['accepted_move_count']} moves and stopped because "
            f"{optimizer['stop_reason']}. The evaluated assignment improved from mean "
            f"{optimizer['initial_metrics']['mean_travel_time']} to "
            f"{optimizer['final_metrics']['mean_travel_time']}, but this is not neighborhood "
            "convergence and not a global optimum. These explanations cover the first "
            f"{len(explanations)} accepted reroutes in iteration order."
        ),
        "explanations": explanations,
    }
    validate_explanations(env, payload, moves)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUTPUT_TXT.write_text(build_text(payload), encoding="utf-8")
    loaded = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    _require(loaded == payload, "explanations.json is not a faithful JSON round-trip")
    print("Phase 14: explainability")
    print("explanations_generated: PASS")
    print("validation: PASS")
    print("wrote:", OUTPUT_JSON.as_posix())
    print("wrote:", OUTPUT_TXT.as_posix())
    print("count:", len(explanations))


if __name__ == "__main__":
    main()
