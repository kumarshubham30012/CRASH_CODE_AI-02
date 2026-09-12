# CRASH_CODE(AI-02)

Deterministic traffic-routing optimization benchmark after a network disruption.

This repository evaluates congestion-aware routing and a local-search optimizer on a synthetic 5×5 directed grid with a fixed official disruption, fixed demand, and a fixed travel-time model. Results are reproducible for the official seed. The project does **not** claim a global optimum or statistical significance.

## What the project does

A 5×5 bidirectional grid loses one directed edge pair. One hundred twenty origin–destination trips are sampled once, then several routing methods assign each trip a simple path. Congestion is applied after assignment through the official polynomial travel-time function. Mean travel time, 95th-percentile travel time, and maximum edge utilization (`flow / capacity`) are the evaluation metrics.

A React dashboard presents the validated outputs. It does not re-solve routes.

## Benchmark

Official instance (defined in `env.py`, not duplicated as independent constants in the runner):

| Item | Value |
| --- | --- |
| Network | 5×5 directed grid |
| Nodes | 25 |
| Official disruption | directed pair `(2,2)↔(3,2)` removed (node IDs `12↔13`) |
| Directed edges after disruption | 78 |
| Demand | 120 trips, distinct OD, shortest-path distance ≥ 4 |
| Seed | `20260911` |
| RNG | NumPy PCG64 |
| Free-flow time | 1 per edge |
| Capacity | 8 |
| Travel time | `t(f) = t0 * (1 + 0.15 * (f / capacity)^4)` |

Evaluation metrics:

- mean congested travel time
- p95 congested travel time (`numpy.percentile`, linear)
- maximum congestion ratio `flow / capacity`

## Algorithms / analysis

**Algorithms (produce route assignments):**

- **Baseline** — independent free-flow shortest path for every trip.
- **Greedy** — sequential congestion-aware Dijkstra using live edge costs from current flow.
- **Local search** — deterministic one-trip reroutes from candidate paths, using bottleneck/impact ranking and a lexicographic improvement rule.
- **Optimizer** — orchestration around the Phase 10 local-search method. Final official metrics may match local search because this layer does not introduce a second solver.

**Supporting generators / analysis (not separate congestion solvers):**

- **K-shortest candidates** — Yen-style simple paths; candidate #1 (index 0) is the free-flow shortest path and matches baseline in the comparison table.
- **Marginal social cost** — `MC(f) = 1 + 0.75 * (f/8)^4`, used as a routing heuristic in greedy, not as the evaluation travel time.
- **Bottlenecks** — rank edges by utilization and related costs.
- **Trip impact** — rank trips by contribution to congested edges.
- **Counterfactual analysis** — ask what happens if one high-impact trip uses a different candidate. An improving swap is not a proof of global optimality.
- **Robustness** — optional extra run on seeds `20260912`, `20260913`, `20260914`. Small-sample check only.

## Quick start

Python 3.11+ recommended. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 run.py
```

`python3 run.py` is the official end-to-end command. It:

1. Validates the official environment (seed `20260911`)
2. Runs baseline, greedy, candidates, bottleneck/impact/counterfactual analysis, local search, and optimizer
3. Writes `comparison.csv`
4. Generates visualizations and explanations
5. Runs the Phase 3 verification gate
6. Syncs dashboard data

It does **not** run the optional robustness seeds.

```bash
python3 run.py --help
python3 run.py --skip-dashboard
python3 run.py --skip-visualizations
python3 run.py --skip-explanations
python3 run.py --robustness
```

Outputs:

- `output/*.json` and `output/explanations.txt`
- `output/figures/*.png`
- `comparison.csv`
- `output/robustness/` only when `--robustness` is used (or from a previous explicit run)
- `dashboard/public/data/` copies (generated, gitignored)

## Dashboard

Presentation layer over existing files. After `python3 run.py` (or `python3 dashboard/sync_data.py`):

```bash
cd dashboard
npm install
npm run dev
```

Production:

```bash
cd dashboard
npm run build
npm run preview
```

`npm run dev` and `npm run build` also run `python3 sync_data.py`. Open the Vite URL (development default `http://localhost:5173`).

## Verification

```bash
python3 verify.py
```

This is the Phase 3 gate for the official environment and baseline reconstruction. `python3 run.py` runs the same gate after regenerating official outputs and exits non-zero if it fails.

Runtime fields may differ between machines. Deterministic metrics, routes, and rankings must not.

## Robustness

```bash
python3 run.py --robustness
```

or:

```bash
python3 robustness.py
```

This is a **small-sample robustness analysis** on three additional deterministic seeds. It writes `output/robustness/` and is required to leave official `output/*.json` and `comparison.csv` unchanged. It does not establish statistical significance or universal superiority.

## Project structure

```
env.py                 Official network, seed, demand
flow.py                Flows and official travel-time function
verify.py              Phase 3 verification gate
run.py                 End-to-end official runner
candidates.py          K-shortest candidate routes
bottlenecks.py         Edge bottleneck ranking
impact.py              Trip impact ranking
counterfactual.py      One-trip reroute probes
comparison.py          Algorithm comparison table
visualize.py           Network / congestion / progress figures
explain.py             Accepted-move explanations
robustness.py          Optional alternative-seed experiment
algorithms/            Baseline, greedy, MSC helper, local search, optimizer
output/                Validated official artifacts
dashboard/             React/Vite presentation UI
requirements.txt       Python dependencies
```

## Reproducibility

- Official seed `20260911` and PCG64 are the only default demand generator.
- `env.py` is the source of benchmark constants.
- `verify.py` reconstructs baseline flows and metrics from saved routes.
- `run.py` refuses to continue if `env.DEFAULT_SEED` is not `20260911`.
- Re-running the official pipeline may change `runtime_seconds` only among permitted fields.

## Limitations

- Local search / optimizer results are neighborhood improvements under a fixed move rule, not a proof of global optimality.
- The robustness experiment uses three extra seeds only.
- The dashboard displays stored outputs; it does not compute new routes.
- The benchmark is this synthetic grid and this demand process, not a city network.
- K-shortest in `comparison.csv` is the free-flow shortest candidate, not a congestion optimizer.

## License / notes

Hackathon project: System-Optimal Transit Rerouting After a Network Disruption.
