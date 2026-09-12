"""End-to-end runner for the official CRASH_CODE(AI-02) benchmark.

Default command:

    python3 run.py

runs the official seed-20260911 pipeline only. Phase 15 alternative-seed
robustness is optional and must be requested with --robustness.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

OFFICIAL_SEED = 20260911
OFFICIAL_NODES = 25
OFFICIAL_EDGES = 78
OFFICIAL_TRIPS = 120

OFFICIAL_OUTPUTS = [
    ROOT / "output" / "baseline.json",
    ROOT / "output" / "greedy.json",
    ROOT / "output" / "candidates.json",
    ROOT / "output" / "bottlenecks.json",
    ROOT / "output" / "impact.json",
    ROOT / "output" / "counterfactual.json",
    ROOT / "output" / "local_search.json",
    ROOT / "output" / "optimizer.json",
    ROOT / "comparison.csv",
]
VISUALIZATION_OUTPUTS = [
    ROOT / "output" / "figures" / "network_before.png",
    ROOT / "output" / "figures" / "network_after.png",
    ROOT / "output" / "figures" / "congestion_before_after.png",
    ROOT / "output" / "figures" / "optimization_progress.png",
]
EXPLANATION_OUTPUTS = [
    ROOT / "output" / "explanations.json",
    ROOT / "output" / "explanations.txt",
]
ROBUSTNESS_OUTPUTS = [
    ROOT / "output" / "robustness" / "results.json",
    ROOT / "output" / "robustness" / "summary.csv",
]


class PhaseError(Exception):
    """A named pipeline phase failed."""

    def __init__(self, phase: str, message: str) -> None:
        super().__init__(f"{phase}: {message}")
        self.phase = phase
        self.message = message


def log(tag: str, text: str) -> None:
    print(f"[{tag}] {text}", flush=True)


def isolate_argv(fn, extra: list[str] | None = None):
    saved = sys.argv[:]
    sys.argv = [saved[0], *(extra or [])]
    try:
        return fn()
    finally:
        sys.argv = saved


def require_files(phase: str, paths: list[Path]) -> None:
    missing = [path.as_posix() for path in paths if not path.is_file()]
    if missing:
        raise PhaseError(phase, "missing expected output(s): " + ", ".join(missing))


def load_dashboard_sync():
    path = ROOT / "dashboard" / "sync_data.py"
    spec = importlib.util.spec_from_file_location("dashboard_sync_data", path)
    if spec is None or spec.loader is None:
        raise PhaseError("dashboard", f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_official_environment() -> None:
    from env import DEFAULT_SEED, create_environment

    if DEFAULT_SEED != OFFICIAL_SEED:
        raise PhaseError(
            "environment",
            f"env.DEFAULT_SEED is {DEFAULT_SEED}, expected {OFFICIAL_SEED}",
        )
    env = create_environment()
    if env.seed != OFFICIAL_SEED:
        raise PhaseError(
            "environment",
            f"create_environment() returned seed {env.seed}, expected {OFFICIAL_SEED}",
        )
    nodes = env.graph.number_of_nodes()
    edges = env.graph.number_of_edges()
    trips = len(env.trips)
    if nodes != OFFICIAL_NODES:
        raise PhaseError("environment", f"node count is {nodes}, expected {OFFICIAL_NODES}")
    if edges != OFFICIAL_EDGES:
        raise PhaseError("environment", f"directed edge count is {edges}, expected {OFFICIAL_EDGES}")
    if trips != OFFICIAL_TRIPS:
        raise PhaseError("environment", f"trip count is {trips}, expected {OFFICIAL_TRIPS}")
    log(
        "PASS",
        f"official environment seed={env.seed} nodes={nodes} edges={edges} trips={trips}",
    )


def run_verify() -> None:
    import verify

    code = verify.main()
    if code != 0:
        raise PhaseError("verification", "Phase 3 verification gate failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "CRASH_CODE(AI-02) official benchmark runner.\n\n"
            "This project is a deterministic traffic-routing optimization benchmark\n"
            "on a disrupted 5x5 directed grid (seed 20260911, 120 trips, 78 directed\n"
            "edges after the official disruption). The default command regenerates\n"
            "official algorithm outputs, analysis artifacts, visualizations,\n"
            "explanations, runs the Phase 3 verification gate, and syncs dashboard data.\n\n"
            "Outputs are written to output/ and comparison.csv. The dashboard copies\n"
            "those files into dashboard/public/data/; it does not solve routes.\n\n"
            "Phase 15 robustness (seeds 20260912-20260914) is a small-sample extra\n"
            "experiment. It is NOT part of the default official run and does not\n"
            "replace seed 20260911."
        ),
        epilog=(
            "Examples:\n"
            "  python3 run.py\n"
            "  python3 run.py --skip-dashboard\n"
            "  python3 run.py --robustness\n"
            "  python3 run.py --skip-visualizations --skip-explanations\n"
        ),
    )
    parser.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Skip copying official outputs into dashboard/public/data/.",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help=(
            "After the official pipeline, run the optional Phase 15 alternative-seed "
            "analysis. Writes only to output/robustness/; does not overwrite official "
            "seed-20260911 outputs."
        ),
    )
    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip Phase 13 figure generation.",
    )
    parser.add_argument(
        "--skip-explanations",
        action="store_true",
        help="Skip Phase 14 explanation generation.",
    )
    return parser.parse_args()


def main() -> int:
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    args = parse_args()
    started = time.perf_counter()
    status = {
        "official": "FAIL",
        "verification": "FAIL",
        "algorithms": "FAIL",
        "analysis": "FAIL",
        "visualizations": "SKIP" if args.skip_visualizations else "FAIL",
        "explanations": "SKIP" if args.skip_explanations else "FAIL",
        "dashboard": "SKIP" if args.skip_dashboard else "FAIL",
        "robustness": "SKIP" if not args.robustness else "FAIL",
    }

    log("RUN", "CRASH_CODE(AI-02) official benchmark pipeline")
    log("RUN", f"working directory: {ROOT}")
    log("RUN", "robustness is optional and is not part of the default official run")

    try:
        log("RUN", "environment / official seed")
        validate_official_environment()

        log("RUN", "baseline")
        from algorithms.baseline import main as baseline_main

        isolate_argv(baseline_main)
        require_files("baseline", [ROOT / "output" / "baseline.json"])
        log("PASS", "baseline")

        log("RUN", "greedy")
        from algorithms.greedy import main as greedy_main

        isolate_argv(greedy_main)
        require_files("greedy", [ROOT / "output" / "greedy.json"])
        log("PASS", "greedy")

        log("RUN", "K-shortest candidates")
        import candidates

        isolate_argv(candidates.main)
        require_files("candidates", [ROOT / "output" / "candidates.json"])
        log("PASS", "candidates")
        status["algorithms"] = "PASS"

        log("RUN", "bottlenecks")
        import bottlenecks

        isolate_argv(bottlenecks.main)
        require_files("bottlenecks", [ROOT / "output" / "bottlenecks.json"])
        log("PASS", "bottlenecks")

        log("RUN", "trip impact")
        import impact

        isolate_argv(impact.main)
        require_files("impact", [ROOT / "output" / "impact.json"])
        log("PASS", "impact")

        log("RUN", "counterfactual")
        import counterfactual

        isolate_argv(counterfactual.main)
        require_files("counterfactual", [ROOT / "output" / "counterfactual.json"])
        log("PASS", "counterfactual")

        log("RUN", "local search")
        from algorithms.local_search import main as local_search_main

        isolate_argv(local_search_main)
        require_files("local_search", [ROOT / "output" / "local_search.json"])
        log("PASS", "local search")

        log("RUN", "optimizer")
        from algorithms.optimizer import main as optimizer_main

        isolate_argv(optimizer_main)
        require_files("optimizer", [ROOT / "output" / "optimizer.json"])
        log("PASS", "optimizer")

        log("RUN", "comparison")
        import comparison

        isolate_argv(comparison.main)
        require_files("comparison", [ROOT / "comparison.csv"])
        log("PASS", "comparison")
        status["analysis"] = "PASS"

        if args.skip_visualizations:
            log("SKIP", "visualizations")
        else:
            log("RUN", "visualizations")
            import visualize

            isolate_argv(visualize.main)
            require_files("visualizations", VISUALIZATION_OUTPUTS)
            log("PASS", "visualizations")
            status["visualizations"] = "PASS"

        if args.skip_explanations:
            log("SKIP", "explanations")
        else:
            log("RUN", "explanations")
            import explain

            isolate_argv(explain.main)
            require_files("explanations", EXPLANATION_OUTPUTS)
            log("PASS", "explanations")
            status["explanations"] = "PASS"

        require_files("official outputs", OFFICIAL_OUTPUTS)
        status["official"] = "PASS"

        log("RUN", "Phase 3 verification")
        run_verify()
        log("PASS", "Phase 3 verification")
        status["verification"] = "PASS"

        if args.robustness:
            log("RUN", "optional Phase 15 robustness (does not overwrite official outputs)")
            import robustness

            isolate_argv(robustness.main)
            require_files("robustness", ROBUSTNESS_OUTPUTS)
            log("PASS", "robustness")
            status["robustness"] = "PASS"
        else:
            log("SKIP", "robustness (pass --robustness to run seeds 20260912-20260914)")

        if args.skip_dashboard:
            log("SKIP", "dashboard sync")
        else:
            log("RUN", "dashboard data sync")
            load_dashboard_sync().main()
            log("PASS", "dashboard data sync")
            status["dashboard"] = "PASS"

    except PhaseError as exc:
        log("FAIL", f"{exc.phase}: {exc.message}")
        _print_summary(status, time.perf_counter() - started)
        return 1
    except Exception as exc:
        log("FAIL", f"{type(exc).__name__}: {exc}")
        _print_summary(status, time.perf_counter() - started)
        return 1

    elapsed = time.perf_counter() - started
    log("PASS", f"official pipeline finished in {elapsed:.2f}s (runtime is informational)")
    _print_summary(status, elapsed)
    if any(value == "FAIL" for value in status.values()):
        return 1
    return 0


def _print_summary(status: dict[str, str], elapsed: float) -> None:
    print()
    print("CRASH_CODE(AI-02)")
    print("-----------------")
    print(f"Official benchmark: {status['official']}")
    print(f"Phase 3 verification: {status['verification']}")
    print(f"Algorithms: {status['algorithms']}")
    print(f"Analysis: {status['analysis']}")
    print(f"Visualizations: {status['visualizations']}")
    print(f"Explanations: {status['explanations']}")
    print(f"Dashboard data: {status['dashboard']}")
    print(f"Robustness: {status['robustness']}")
    print(f"Wall time (informational): {elapsed:.2f}s")
    print()


if __name__ == "__main__":
    sys.exit(main())
