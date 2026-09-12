"""Copy validated CRASH_CODE outputs into the dashboard public data directory."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(__file__).resolve().parent / "public" / "data"
sys.path.insert(0, str(ROOT))

from env import (
    CAPACITY,
    DEFAULT_NUM_TRIPS,
    DEFAULT_SEED,
    FREE_FLOW_TIME,
    GRID_SIZE,
    REMOVED_COORD_PAIRS,
    node_id,
)


def copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(f"Missing benchmark output: {src}")
    shutil.copy2(src, dest)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    copies = {
        ROOT / "comparison.csv": DEST / "comparison.csv",
        ROOT / "output" / "baseline.json": DEST / "baseline.json",
        ROOT / "output" / "greedy.json": DEST / "greedy.json",
        ROOT / "output" / "bottlenecks.json": DEST / "bottlenecks.json",
        ROOT / "output" / "impact.json": DEST / "impact.json",
        ROOT / "output" / "counterfactual.json": DEST / "counterfactual.json",
        ROOT / "output" / "optimizer.json": DEST / "optimizer.json",
        ROOT / "output" / "local_search.json": DEST / "local_search.json",
        ROOT / "output" / "explanations.json": DEST / "explanations.json",
        ROOT / "output" / "robustness" / "results.json": DEST / "robustness_results.json",
        ROOT / "output" / "robustness" / "summary.csv": DEST / "robustness_summary.csv",
        ROOT / "output" / "figures" / "network_before.png": DEST / "figures" / "network_before.png",
        ROOT / "output" / "figures" / "network_after.png": DEST / "figures" / "network_after.png",
        ROOT / "output" / "figures" / "congestion_before_after.png": DEST / "figures" / "congestion_before_after.png",
        ROOT / "output" / "figures" / "optimization_progress.png": DEST / "figures" / "optimization_progress.png",
    }
    for src, dest in copies.items():
        copy_file(src, dest)

    meta = {
        "grid_size": GRID_SIZE,
        "seed": DEFAULT_SEED,
        "trips": DEFAULT_NUM_TRIPS,
        "nodes": GRID_SIZE * GRID_SIZE,
        "capacity": CAPACITY,
        "free_flow_time": FREE_FLOW_TIME,
        "disrupted_edges": [
            f"{node_id(x1, y1)}->{node_id(x2, y2)}"
            for (x1, y1), (x2, y2) in REMOVED_COORD_PAIRS
        ],
        "disrupted_coords": [[[x1, y1], [x2, y2]] for (x1, y1), (x2, y2) in REMOVED_COORD_PAIRS],
        "travel_time_formula": "t(f) = t0 * (1 + 0.15 * (f / capacity)^4)",
        "note": "Copied from env.py. Dashboard does not redefine the benchmark.",
    }
    (DEST / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("synced", len(copies), "files plus meta.json ->", DEST)


if __name__ == "__main__":
    main()
