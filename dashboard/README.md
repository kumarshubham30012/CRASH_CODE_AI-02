# CRASH_CODE(AI-02) Dashboard

Presentation and analysis UI for the deterministic traffic-routing benchmark.

The dashboard does **not** re-run routing algorithms. It loads validated project outputs and displays them.

## Data sources

Copied into `public/data/` by `sync_data.py` from the repository root:

- `comparison.csv`
- `output/baseline.json`
- `output/greedy.json`
- `output/bottlenecks.json`
- `output/impact.json`
- `output/counterfactual.json`
- `output/optimizer.json`
- `output/local_search.json`
- `output/explanations.json`
- `output/robustness/results.json`
- `output/robustness/summary.csv`
- `output/figures/*.png`
- `meta.json` generated from `env.py` (seed, grid, disruption, formula)

Re-run the sync step after regenerating benchmark outputs.

## Install

```bash
cd dashboard
python3 sync_data.py
npm install
```

## Development

```bash
cd dashboard
npm run dev
```

Opens Vite on port 5173. `npm run dev` syncs data first.

## Production build

```bash
cd dashboard
npm run build
npm run preview
```

## Notes

- Best reported metrics are the minima in `comparison.csv`, not a global optimum.
- K-shortest is candidate #1 / free-flow shortest, not a congestion optimizer.
- Optimizer wraps Phase 10 local search.
- Robustness is a 3-seed sample, not statistical significance.
