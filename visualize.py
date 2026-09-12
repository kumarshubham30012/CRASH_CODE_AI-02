"""Phase 13: deterministic figures from validated benchmark outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import Normalize
from matplotlib import cm

from algorithms.baseline import load_environment
from env import (
    CAPACITY,
    FREE_FLOW_TIME,
    GRID_SIZE,
    REMOVED_COORD_PAIRS,
    build_network,
    node_id,
)
from flow import parse_edge_key

FIGURE_DIR = Path("output") / "figures"
DPI = 200
BASELINE_PATH = Path("output") / "baseline.json"
OPTIMIZER_PATH = Path("output") / "optimizer.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def node_positions() -> dict[int, tuple[float, float]]:
    return {node_id(x, y): (float(x), float(y)) for y in range(GRID_SIZE) for x in range(GRID_SIZE)}


def undisrupted_graph():
    graph = build_network()
    for (x1, y1), (x2, y2) in REMOVED_COORD_PAIRS:
        graph.add_edge(
            node_id(x1, y1),
            node_id(x2, y2),
            free_flow_time=FREE_FLOW_TIME,
            capacity=CAPACITY,
        )
    return graph


def offset_endpoints(
    start: tuple[float, float],
    end: tuple[float, float],
    amount: float = 0.06,
) -> tuple[tuple[float, float], tuple[float, float]]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return start, end
    nx, ny = -dy / length * amount, dx / length * amount
    return (start[0] + nx, start[1] + ny), (end[0] + nx, end[1] + ny)


def draw_directed_edges(ax, graph, positions, edge_kwargs, color_fn=None, width_fn=None):
    for u, v in sorted(graph.edges()):
        start, end = offset_endpoints(positions[u], positions[v])
        color = color_fn(u, v) if color_fn else edge_kwargs.get("color", "#4a4a4a")
        width = width_fn(u, v) if width_fn else edge_kwargs.get("lw", 1.4)
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            lw=width,
            color=color,
            shrinkA=8,
            shrinkB=8,
            zorder=1,
        )
        ax.add_patch(arrow)


def style_grid_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_aspect("equal")
    ax.set_xlim(-0.6, GRID_SIZE - 0.4)
    ax.set_ylim(-0.6, GRID_SIZE - 0.4)
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)


def draw_nodes(ax, positions) -> None:
    xs = [positions[n][0] for n in sorted(positions)]
    ys = [positions[n][1] for n in sorted(positions)]
    ax.scatter(xs, ys, s=420, c="white", edgecolors="#1f4e79", linewidths=1.6, zorder=3)
    for node, (x, y) in positions.items():
        ax.text(x, y, str(node), ha="center", va="center", fontsize=8, zorder=4)


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_network_before(path: Path) -> None:
    graph = undisrupted_graph()
    positions = node_positions()
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    draw_directed_edges(ax, graph, positions, {"color": "#4a4a4a", "lw": 1.3})
    draw_nodes(ax, positions)
    style_grid_axis(ax, "Official 5×5 grid before disruption")
    ax.legend(
        handles=[
            Line2D([0], [0], color="#4a4a4a", lw=1.5, label="Directed grid edges"),
            Line2D([0], [0], marker="o", color="white", markeredgecolor="#1f4e79", markersize=10, linestyle="None", label="Node ID"),
        ],
        loc="upper right",
        framealpha=0.95,
    )
    save_figure(fig, path)


def plot_network_after(path: Path) -> None:
    disrupted = build_network()
    full = undisrupted_graph()
    positions = node_positions()
    missing = [(u, v) for u, v in full.edges() if not disrupted.has_edge(u, v)]
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    draw_directed_edges(ax, disrupted, positions, {"color": "#4a4a4a", "lw": 1.3})
    for u, v in sorted(missing):
        start, end = offset_endpoints(positions[u], positions[v])
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            lw=2.0,
            color="#c0392b",
            linestyle=(0, (4, 3)),
            shrinkA=8,
            shrinkB=8,
            zorder=2,
        )
        ax.add_patch(arrow)
    draw_nodes(ax, positions)
    style_grid_axis(ax, "Disrupted 5×5 graph used by the benchmark")
    ax.legend(
        handles=[
            Line2D([0], [0], color="#4a4a4a", lw=1.5, label="Remaining directed edges"),
            Line2D([0], [0], color="#c0392b", lw=2.0, linestyle="--", label="Removed: (2,2) ↔ (3,2)"),
        ],
        loc="upper right",
        framealpha=0.95,
    )
    save_figure(fig, path)


def parse_flows(raw: dict[str, int | float]) -> dict[tuple[int, int], float]:
    return {parse_edge_key(key): float(value) for key, value in raw.items()}


def plot_congestion(path: Path) -> None:
    env = load_environment()
    graph = env.graph
    positions = node_positions()
    baseline = parse_flows(load_json(BASELINE_PATH)["edge_flows"])
    optimizer = parse_flows(load_json(OPTIMIZER_PATH)["baseline"]["final_edge_flows"])

    def utilization(flows, u, v) -> float:
        capacity = graph[u][v]["capacity"]
        return flows.get((u, v), 0.0) / capacity

    baseline_max = max(utilization(baseline, u, v) for u, v in graph.edges())
    optimizer_max = max(utilization(optimizer, u, v) for u, v in graph.edges())
    vmax = max(baseline_max, optimizer_max, 1e-9)
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = cm.YlOrRd

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.2))
    panels = [
        (axes[0], baseline, f"Baseline\nmax congestion = {baseline_max:.3f}"),
        (
            axes[1],
            optimizer,
            "Final local-search assignment\n(Phase 10 via optimizer wrapper)\n"
            f"max congestion = {optimizer_max:.3f}",
        ),
    ]
    for ax, flows, title in panels:
        draw_directed_edges(
            ax,
            graph,
            positions,
            {},
            color_fn=lambda u, v, f=flows: cmap(norm(utilization(f, u, v))),
            width_fn=lambda u, v, f=flows: 1.0 + 2.8 * utilization(f, u, v) / vmax,
        )
        draw_nodes(ax, positions)
        style_grid_axis(ax, title)
    fig.suptitle("Edge congestion: baseline vs final local-search assignment", fontsize=13, y=1.02)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("utilization = flow / capacity")
    save_figure(fig, path)


def plot_progress(path: Path) -> None:
    history = load_json(OPTIMIZER_PATH)["baseline"]["iteration_history"]
    if not history:
        raise ValueError("optimizer.json has no baseline iteration history")
    iterations = [0]
    means = [history[0]["metrics_before"]["mean_travel_time"]]
    p95s = [history[0]["metrics_before"]["p95_travel_time"]]
    maxc = [history[0]["metrics_before"]["max_congestion_ratio"]]
    for record in history:
        iterations.append(int(record["iteration"]))
        metrics = record["metrics_after"]
        means.append(metrics["mean_travel_time"])
        p95s.append(metrics["p95_travel_time"])
        maxc.append(metrics["max_congestion_ratio"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 7.2), sharex=True)
    ax1.plot(iterations, means, marker="o", color="#1f4e79", label="Mean travel time")
    ax1.plot(iterations, p95s, marker="s", color="#e67e22", label="p95 travel time")
    ax1.set_ylabel("Travel time")
    ax1.set_title("Baseline-start local-search progress (Phase 10 / optimizer wrapper)")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.45)
    ax2.plot(iterations, maxc, marker="D", color="#c0392b", label="Max congestion ratio")
    ax2.set_xlabel("Iteration (0 = initial assignment)")
    ax2.set_ylabel("Max congestion ratio")
    ax2.grid(True, linestyle=":", alpha=0.45)
    ax2.legend(loc="upper right")
    fig.text(
        0.5,
        0.01,
        "Stop reason: max_iterations_reached. Locally improved assignment, not a global optimum.",
        ha="center",
        fontsize=8,
    )
    save_figure(fig, path)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_network_before(FIGURE_DIR / "network_before.png")
    plot_network_after(FIGURE_DIR / "network_after.png")
    plot_congestion(FIGURE_DIR / "congestion_before_after.png")
    plot_progress(FIGURE_DIR / "optimization_progress.png")
    print("Phase 13: visualization")
    print("network_before: PASS")
    print("network_after: PASS")
    print("congestion_before_after: PASS")
    print("optimization_progress: PASS")
    print("wrote:", FIGURE_DIR.as_posix())


if __name__ == "__main__":
    main()
