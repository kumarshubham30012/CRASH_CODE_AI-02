"""Phase 6: internal marginal social-cost heuristic.

This module is an INTERNAL optimization heuristic.

It is NOT the official benchmark travel-time formula:

    t(f) = t0 * (1 + 0.15 * (f / capacity)^4)

MC(f) is the marginal social-cost factor of adding one more unit of
flow to an edge that currently carries flow f:

    MC(f) = 1 + 0.75 * (f / 8)^4

The denominator 8 is part of this heuristic and is not taken from
edge capacity. Cost helpers are pure: they do not mutate flows.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

MC_INTERCEPT = 1.0
MC_COEFFICIENT = 0.75
MC_DENOMINATOR = 8.0
MC_EXPONENT = 4.0


def _as_flow(flow: float | int) -> float:
    try:
        value = float(flow)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"flow must be numeric, got {flow!r}") from exc
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"flow must be finite, got {flow!r}")
    if value < 0:
        raise ValueError(f"flow must be non-negative, got {value}")
    return value


def marginal_cost(flow: float | int) -> float:
    """Return MC(f) = 1 + 0.75 * (f / 8)^4 for current edge flow f.

    MC(f) is the incremental social-cost factor of adding another unit of
    flow when the edge already carries f. It is not a travel time.
    """
    value = _as_flow(flow)
    result = MC_INTERCEPT + MC_COEFFICIENT * (value / MC_DENOMINATOR) ** MC_EXPONENT
    if not math.isfinite(result):
        raise ValueError(f"marginal cost is not finite for flow={flow!r}")
    return result


def route_marginal_cost(
    route: Sequence[int],
    edge_flows: Mapping[tuple[int, int], float | int],
    graph_edges: set[tuple[int, int]] | None = None,
) -> float:
    """Sum MC(current_flow(edge)) along a route without mutating flows."""
    if len(route) < 2:
        raise ValueError(f"route must contain at least two nodes, got {list(route)}")
    total = 0.0
    for u, v in zip(route, route[1:]):
        edge = (int(u), int(v))
        if graph_edges is not None and edge not in graph_edges:
            raise ValueError(f"route uses missing directed edge {u}->{v}")
        if edge not in edge_flows:
            raise ValueError(f"no current flow recorded for edge {u}->{v}")
        total += marginal_cost(edge_flows[edge])
    if not math.isfinite(total):
        raise ValueError(f"route marginal cost is not finite for route {list(route)}")
    return total


def _self_test() -> None:
    def close(actual: float, expected: float) -> None:
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(f"expected {expected}, got {actual}")

    close(marginal_cost(0), 1.0)
    close(marginal_cost(8), 1.75)
    close(marginal_cost(16), 13.0)

    for invalid in (-1, float("nan"), float("inf"), float("-inf")):
        try:
            marginal_cost(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for flow={invalid!r}")

    for flow in (0, 1, 3, 8, 12, 16, 24):
        value = marginal_cost(flow)
        if not math.isfinite(value):
            raise AssertionError(f"MC({flow}) is not finite: {value}")

    route_cost = route_marginal_cost(
        [0, 1, 2],
        {(0, 1): 0, (1, 2): 8},
        graph_edges={(0, 1), (1, 2)},
    )
    close(route_cost, 2.75)


def main() -> None:
    _self_test()
    print("algorithm: marginal_social_cost")
    print("formula: MC(f) = 1 + 0.75 * (f / 8)^4")
    print("MC(0):", marginal_cost(0))
    print("MC(8):", marginal_cost(8))
    print("MC(16):", marginal_cost(16))
    print("self_test: PASS")


if __name__ == "__main__":
    main()
