"""Phase 7: deterministic bottleneck ranking from routing-solution flows."""

from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx

from algorithms.baseline import load_environment
from algorithms.marginal_cost import marginal_cost
from env import Environment
from flow import (
    BPR_ALPHA,
    BPR_BETA,
    CAPACITY_ATTR,
    FREE_FLOW_ATTR,
    congested_travel_time,
    edge_key,
    parse_edge_key,
)

OUTPUT_PATH = Path("output") / "bottlenecks.json"
BASELINE_PATH = Path("output") / "baseline.json"
GREEDY_PATH = Path("output") / "greedy.json"
ALGORITHM_NAME = "bottleneck_analysis"
FLOAT_ABS_TOL = 1e-12


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=FLOAT_ABS_TOL)


def parse_edge_flows(raw_flows: dict[str, int | float]) -> dict[tuple[int, int], int]:
    flows: dict[tuple[int, int], int] = {}
    for key, value in raw_flows.items():
        edge = parse_edge_key(key)
        flow = int(value)
        _require(flow >= 0, f"edge {key} has negative flow {value}")
        flows[edge] = flow
    return flows


def load_solution_flows(path: Path) -> dict[tuple[int, int], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require("edge_flows" in payload, f"{path.as_posix()} has no edge_flows")
    return parse_edge_flows(payload["edge_flows"])


def complete_flows(
    graph: nx.DiGraph,
    flows: dict[tuple[int, int], int],
) -> dict[tuple[int, int], int]:
    complete = {(u, v): 0 for u, v in graph.edges()}
    extra = set(flows) - set(complete)
    _require(not extra, f"flows contain edges absent from the disrupted graph: {sorted(extra)[:5]}")
    complete.update(flows)
    return complete


def ranking_key(record: dict) -> tuple:
    """Higher utilization first; ties: higher flow, lower capacity, then node IDs."""
    return (
        -record["bottleneck_score"],
        -record["flow"],
        record["capacity"],
        record["source"],
        record["destination"],
    )


def analyze_edge(
    graph: nx.DiGraph,
    u: int,
    v: int,
    flow: int,
) -> dict:
    data = graph[u][v]
    t0 = float(data[FREE_FLOW_ATTR])
    capacity = float(data[CAPACITY_ATTR])
    _require(capacity > 0, f"edge {u}->{v} has non-positive capacity {capacity}")
    _require(flow >= 0, f"edge {u}->{v} has negative flow {flow}")
    utilization = flow / capacity
    congested = congested_travel_time(t0, flow, capacity)
    mc = marginal_cost(flow)
    multiplier = 1.0 + BPR_ALPHA * (flow / capacity) ** BPR_BETA
    return {
        "edge": edge_key(u, v),
        "source": int(u),
        "destination": int(v),
        "flow": int(flow),
        "capacity": capacity,
        "utilization": utilization,
        "congestion_ratio": utilization,
        "free_flow_time": t0,
        "congested_travel_time": congested,
        "official_congestion_multiplier": multiplier,
        "excess_flow": max(flow - capacity, 0.0),
        "marginal_social_cost": mc,
        "bottleneck_score": utilization,
    }


def analyze_bottlenecks(
    env: Environment,
    edge_flows: dict[tuple[int, int], int],
) -> list[dict]:
    """Rank every disrupted-graph edge by utilization/congestion severity."""
    flows = complete_flows(env.graph, edge_flows)
    records = [
        analyze_edge(env.graph, u, v, flows[(u, v)])
        for u, v in env.graph.edges()
    ]
    records.sort(key=ranking_key)
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank
    validate_bottleneck_records(env, records, flows)
    return records


def validate_bottleneck_records(
    env: Environment,
    records: list[dict],
    flows: dict[tuple[int, int], int],
) -> None:
    graph_edges = {(int(u), int(v)) for u, v in env.graph.edges()}
    _require(len(records) == env.graph.number_of_edges(), "analyzed edge count does not match the graph")
    _require(len(records) == 78, f"expected 78 directed edges, got {len(records)}")

    seen_edges: set[tuple[int, int]] = set()
    seen_ranks: set[int] = set()
    for record in records:
        edge = (record["source"], record["destination"])
        _require(edge in graph_edges, f"analyzed edge {edge} is not in the disrupted graph")
        _require(edge not in seen_edges, f"duplicate edge record {edge}")
        seen_edges.add(edge)
        rank = record["rank"]
        _require(rank not in seen_ranks, f"duplicate rank {rank}")
        seen_ranks.add(rank)
        _require(record["flow"] >= 0, f"negative flow on {edge}")
        _require(record["capacity"] > 0, f"non-positive capacity on {edge}")
        _require(record["flow"] == flows[edge], f"flow mismatch on {edge}")

        expected_util = record["flow"] / record["capacity"]
        _require(_close(record["utilization"], expected_util), f"utilization mismatch on {edge}")
        _require(_close(record["congestion_ratio"], expected_util), f"congestion ratio mismatch on {edge}")
        _require(_close(record["bottleneck_score"], expected_util), f"bottleneck score mismatch on {edge}")

        expected_time = congested_travel_time(
            record["free_flow_time"], record["flow"], record["capacity"]
        )
        _require(
            _close(record["congested_travel_time"], expected_time),
            f"congested travel time mismatch on {edge}",
        )
        expected_mc = marginal_cost(record["flow"])
        _require(
            _close(record["marginal_social_cost"], expected_mc),
            f"marginal social cost mismatch on {edge}",
        )
        for field in (
            "utilization",
            "congestion_ratio",
            "congested_travel_time",
            "marginal_social_cost",
            "bottleneck_score",
        ):
            _require(math.isfinite(record[field]), f"{field} is not finite on {edge}")

    _require(seen_edges == graph_edges, "analysis is missing disrupted-graph edges")
    _require(seen_ranks == set(range(1, len(records) + 1)), "ranks are not a unique 1..N sequence")
    ordered = sorted(records, key=ranking_key)
    _require(ordered == records, "records are not in deterministic rank order")


def build_output(env: Environment, analyses: dict[str, list[dict]]) -> dict:
    payload: dict = {
        "algorithm": ALGORITHM_NAME,
        "seed": env.seed,
        "number_of_nodes": env.graph.number_of_nodes(),
        "number_of_directed_edges": env.graph.number_of_edges(),
    }
    for name, records in analyses.items():
        payload[name] = {
            "number_of_edges": len(records),
            "max_utilization": max(record["utilization"] for record in records),
            "edges": records,
        }
    return payload


def write_bottlenecks_json(payload: dict, path: Path = OUTPUT_PATH) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(loaded, dict), "output/bottlenecks.json is not a JSON object")
    return loaded


def print_summary(payload: dict) -> None:
    print("algorithm:", payload["algorithm"])
    print("seed:", payload["seed"])
    print("number_of_nodes:", payload["number_of_nodes"])
    print("number_of_directed_edges:", payload["number_of_directed_edges"])
    for name in ("baseline", "greedy"):
        section = payload[name]
        print()
        print(f"{name.capitalize()}:")
        print("max utilization:", section["max_utilization"])
        print("top 5 bottlenecks:")
        for record in section["edges"][:5]:
            print(
                f"  rank {record['rank']}: {record['edge']} "
                f"flow={record['flow']} utilization={record['utilization']} "
                f"MC={record['marginal_social_cost']}"
            )
    print()
    print("wrote:", OUTPUT_PATH.as_posix())


def main() -> None:
    env = load_environment()
    analyses = {
        "baseline": analyze_bottlenecks(env, load_solution_flows(BASELINE_PATH)),
        "greedy": analyze_bottlenecks(env, load_solution_flows(GREEDY_PATH)),
    }
    payload = build_output(env, analyses)
    write_bottlenecks_json(payload)
    print_summary(payload)


if __name__ == "__main__":
    main()
