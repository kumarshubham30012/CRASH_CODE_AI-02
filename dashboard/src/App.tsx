import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  GitBranch,
  Info,
  Layers,
  Map as MapIcon,
  Route,
  Shield,
  Table2,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import NetworkMap from "./components/NetworkMap";
import {
  bestReported,
  fetchJson,
  fetchText,
  formatNum,
  parseComparison,
  parseCsv,
  toNumber,
  utilizationClass,
  type ComparisonRow,
  type Meta,
} from "./data/load";

const NAV = [
  ["overview", "Overview", Layers],
  ["algorithms", "Algorithms", Table2],
  ["network", "Network", MapIcon],
  ["bottlenecks", "Bottlenecks", Activity],
  ["trips", "Trips", Route],
  ["optimization", "Optimization", GitBranch],
  ["robustness", "Robustness", Shield],
  ["methodology", "Methodology", Info],
] as const;

type AnyJson = Record<string, unknown>;

function asRecord(value: unknown): AnyJson {
  return value && typeof value === "object" ? (value as AnyJson) : {};
}

function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <div className="card">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
    </div>
  );
}

function Badge({ util }: { util: number }) {
  const kind = utilizationClass(util);
  const label = kind === "over" ? "over capacity" : kind === "near" ? "near capacity" : "below capacity";
  return <span className={`badge ${kind}`}>{label}</span>;
}

export default function App() {
  const [error, setError] = useState<string | null>(null);
  const [section, setSection] = useState("overview");
  const [meta, setMeta] = useState<Meta | null>(null);
  const [comparison, setComparison] = useState<ComparisonRow[]>([]);
  const [baseline, setBaseline] = useState<AnyJson | null>(null);
  const [greedy, setGreedy] = useState<AnyJson | null>(null);
  const [bottlenecks, setBottlenecks] = useState<AnyJson | null>(null);
  const [impact, setImpact] = useState<AnyJson | null>(null);
  const [counterfactual, setCounterfactual] = useState<AnyJson | null>(null);
  const [optimizer, setOptimizer] = useState<AnyJson | null>(null);
  const [explanations, setExplanations] = useState<AnyJson | null>(null);
  const [robustness, setRobustness] = useState<AnyJson | null>(null);
  const [robustCsv, setRobustCsv] = useState<Record<string, string>[]>([]);
  const [netAlgo, setNetAlgo] = useState("baseline");
  const [bnAlgo, setBnAlgo] = useState("baseline");
  const [tripQuery, setTripQuery] = useState("");
  const [selectedTrip, setSelectedTrip] = useState<AnyJson | null>(null);
  const [cfTrip, setCfTrip] = useState<AnyJson | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [
          metaJson,
          comparisonText,
          baselineJson,
          greedyJson,
          bottlenecksJson,
          impactJson,
          counterfactualJson,
          optimizerJson,
          explanationsJson,
          robustnessJson,
          robustText,
        ] = await Promise.all([
          fetchJson<Meta>("/data/meta.json"),
          fetchText("/data/comparison.csv"),
          fetchJson<AnyJson>("/data/baseline.json"),
          fetchJson<AnyJson>("/data/greedy.json"),
          fetchJson<AnyJson>("/data/bottlenecks.json"),
          fetchJson<AnyJson>("/data/impact.json"),
          fetchJson<AnyJson>("/data/counterfactual.json"),
          fetchJson<AnyJson>("/data/optimizer.json"),
          fetchJson<AnyJson>("/data/explanations.json"),
          fetchJson<AnyJson>("/data/robustness_results.json"),
          fetchText("/data/robustness_summary.csv"),
        ]);
        if (cancelled) return;
        setMeta(metaJson);
        setComparison(parseComparison(comparisonText));
        setBaseline(baselineJson);
        setGreedy(greedyJson);
        setBottlenecks(bottlenecksJson);
        setImpact(impactJson);
        setCounterfactual(counterfactualJson);
        setOptimizer(optimizerJson);
        setExplanations(explanationsJson);
        setRobustness(robustnessJson);
        setRobustCsv(parseCsv(robustText));
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load benchmark data");
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  const bestMean = bestReported(comparison, "mean_travel_time");
  const bestP95 = bestReported(comparison, "p95_travel_time");
  const bestMax = bestReported(comparison, "max_congestion_ratio");
  const optBaseline = asRecord(optimizer?.baseline);
  const history = (optBaseline.iteration_history as AnyJson[] | undefined) ?? [];
  const progress = useMemo(() => {
    if (!history.length) return [];
    const first = asRecord(history[0]);
    const before = asRecord(first.metrics_before);
    const points = [
      {
        iteration: 0,
        mean: toNumber(before.mean_travel_time as number),
        p95: toNumber(before.p95_travel_time as number),
        maxc: toNumber(before.max_congestion_ratio as number),
      },
    ];
    for (const record of history) {
      const rec = asRecord(record);
      const after = asRecord(rec.metrics_after);
      points.push({
        iteration: Number(rec.iteration),
        mean: toNumber(after.mean_travel_time as number),
        p95: toNumber(after.p95_travel_time as number),
        maxc: toNumber(after.max_congestion_ratio as number),
      });
    }
    return points;
  }, [history]);

  const flows = useMemo(() => {
    if (netAlgo === "greedy") return (greedy?.edge_flows as Record<string, number>) ?? {};
    if (netAlgo === "optimizer") return (asRecord(optBaseline.final_edge_flows) as Record<string, number>) ?? {};
    return (baseline?.edge_flows as Record<string, number>) ?? {};
  }, [netAlgo, greedy, baseline, optBaseline]);

  const bnEdges = ((asRecord(bottlenecks?.[bnAlgo]).edges as AnyJson[]) ?? []);
  const impactTrips = ((asRecord(impact?.baseline).trips as AnyJson[]) ?? []).filter((trip) => {
    if (!tripQuery) return true;
    return JSON.stringify(trip).includes(tripQuery);
  });
  const cfTrips = (asRecord(counterfactual?.baseline).trip_results as AnyJson[]) ?? [];
  const explanationItems = (explanations?.explanations as AnyJson[]) ?? [];

  if (error) {
    return (
      <div className="main">
        <div className="error">
          <strong>Benchmark data could not be loaded.</strong>
          <p>{error}</p>
          <p>Run `python3 dashboard/sync_data.py` then restart the dashboard. Missing values are not replaced with estimates.</p>
        </div>
      </div>
    );
  }

  if (!meta || !comparison.length) {
    return <div className="main note">Loading validated benchmark outputs…</div>;
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          CRASH_CODE(AI-02)
          <small>Deterministic benchmark</small>
        </div>
        {NAV.map(([id, label, Icon]) => (
          <button key={id} className={`nav-btn ${section === id ? "active" : ""}`} onClick={() => setSection(id)}>
            <Icon size={14} style={{ marginRight: 8 }} />
            {label}
          </button>
        ))}
      </aside>
      <main className="main">
        {section === "overview" && (
          <>
            <div className="hero">
              <h1>CRASH_CODE(AI-02)</h1>
              <p>Deterministic Traffic Routing Optimization Benchmark</p>
              <p>Benchmarking congestion-aware routing and deterministic optimization under network disruption.</p>
              <div className="pills">
                <span className="pill">5×5 Grid</span>
                <span className="pill">{meta.trips} Trips</span>
                <span className="pill">Seed: {meta.seed}</span>
                <span className="pill">{asRecord(baseline).number_of_directed_edges as number} Directed Edges</span>
                <span className="pill">Official disruption: {meta.disrupted_edges.join(", ")}</span>
              </div>
            </div>
            <div className="grid kpis">
              <Kpi label="Nodes" value={String(meta.nodes)} />
              <Kpi label="Directed edges after disruption" value={String(asRecord(baseline).number_of_directed_edges ?? "Data unavailable")} />
              <Kpi label="Trips" value={String(asRecord(baseline).number_of_trips ?? meta.trips)} />
              <Kpi label="Benchmark seed" value={String(asRecord(baseline).seed ?? meta.seed)} />
              <Kpi label="Disrupted edge pair" value={meta.disrupted_edges.join(" / ")} />
              <Kpi label="Best reported mean travel time" value={formatNum(bestMean)} />
              <Kpi label="Best reported p95 travel time" value={formatNum(bestP95)} />
              <Kpi label="Lowest reported max congestion" value={formatNum(bestMax, 3)} />
            </div>
            <p className="note">
              “Best reported benchmark result” is the minimum value among algorithms in comparison.csv.
              It is not a global optimum and is not a statistical claim.
            </p>
          </>
        )}

        {section === "algorithms" && (
          <div className="section">
            <h2>Algorithm comparison</h2>
            <p className="note">
              Source: comparison.csv. K-shortest is candidate #1 / free-flow shortest-path, not a congestion optimizer.
              Optimizer and local search can match because Phase 11 only orchestrates Phase 10 local search.
            </p>
            <div className="card">
              <table>
                <thead>
                  <tr>
                    <th>Algorithm</th>
                    <th>Trips</th>
                    <th>Mean</th>
                    <th>p95</th>
                    <th>Max congestion</th>
                    <th>Runtime (s)</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.map((row) => (
                    <tr key={row.algorithm}>
                      <td>{row.algorithm}</td>
                      <td>{row.trips}</td>
                      <td className={row.mean_travel_time === bestMean ? "best" : ""}>{formatNum(row.mean_travel_time)}</td>
                      <td className={row.p95_travel_time === bestP95 ? "best" : ""}>{formatNum(row.p95_travel_time)}</td>
                      <td className={row.max_congestion_ratio === bestMax ? "best" : ""}>{formatNum(row.max_congestion_ratio, 3)}</td>
                      <td>{row.runtime_seconds === null ? "—" : formatNum(row.runtime_seconds, 6)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="grid" style={{ gridTemplateColumns: "1fr", marginTop: 16 }}>
              {[
                ["Mean travel time", "mean_travel_time"],
                ["P95 travel time", "p95_travel_time"],
                ["Maximum congestion", "max_congestion_ratio"],
              ].map(([title, key]) => (
                <div className="card" key={key}>
                  <h3>{title}</h3>
                  <div className="chart">
                    <ResponsiveContainer>
                      <BarChart data={comparison}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#2a3548" />
                        <XAxis dataKey="algorithm" stroke="#93a0b5" />
                        <YAxis stroke="#93a0b5" />
                        <Tooltip />
                        <Bar dataKey={key} fill="#5b8def" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {section === "network" && (
          <div className="section">
            <h2>Network view</h2>
            <p className="note">
              Interactive 5×5 directed grid from env.py geometry. Dashed red edges are the official disruption.
              Color is utilization = flow / capacity from the selected assignment. Capacity comes from env.py via meta.json.
            </p>
            <div className="row">
              <select className="select" value={netAlgo} onChange={(e) => setNetAlgo(e.target.value)}>
                <option value="baseline">baseline</option>
                <option value="greedy">greedy</option>
                <option value="optimizer">local_search / optimizer</option>
              </select>
              <div className="legend">
                <span><span className="dot" style={{ background: "#3ecf8e" }} />below capacity</span>
                <span><span className="dot" style={{ background: "#e6b84d" }} />near capacity (≥0.8)</span>
                <span><span className="dot" style={{ background: "#e85d5d" }} />over capacity (&gt;1)</span>
                <span><span className="dot" style={{ background: "#e85d5d" }} />disrupted (dashed)</span>
              </div>
            </div>
            <div className="card">
              <NetworkMap flows={flows} capacity={meta.capacity} disrupted={meta.disrupted_edges} />
            </div>
            <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", marginTop: 12 }}>
              <img className="figure" src="/data/figures/network_before.png" alt="Network before disruption" />
              <img className="figure" src="/data/figures/network_after.png" alt="Network after disruption" />
              <img className="figure" src="/data/figures/congestion_before_after.png" alt="Congestion before and after" />
              <img className="figure" src="/data/figures/optimization_progress.png" alt="Optimization progress" />
            </div>
          </div>
        )}

        {section === "bottlenecks" && (
          <div className="section">
            <h2>Bottleneck analysis</h2>
            <p className="note">Source: bottlenecks.json. Ranking is unchanged from Phase 7. Utilization &gt; 1 means flow exceeds nominal capacity.</p>
            <select className="select" value={bnAlgo} onChange={(e) => setBnAlgo(e.target.value)}>
              <option value="baseline">baseline</option>
              <option value="greedy">greedy</option>
            </select>
            <div className="card" style={{ marginTop: 12, overflowX: "auto" }}>
              <table>
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Edge</th>
                    <th>Flow</th>
                    <th>Capacity</th>
                    <th>Utilization</th>
                    <th>Travel time</th>
                    <th>Marginal social cost</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {bnEdges.slice(0, 25).map((edge) => {
                    const util = Number(edge.utilization);
                    return (
                      <tr key={String(edge.edge)}>
                        <td>{String(edge.rank)}</td>
                        <td>{String(edge.source)} → {String(edge.destination)}</td>
                        <td>{String(edge.flow)}</td>
                        <td>{String(edge.capacity)}</td>
                        <td>{formatNum(util, 3)}</td>
                        <td>{formatNum(Number(edge.congested_travel_time))}</td>
                        <td>{edge.marginal_social_cost === undefined ? "Data unavailable" : formatNum(Number(edge.marginal_social_cost))}</td>
                        <td><Badge util={util} /></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {section === "trips" && (
          <div className="section">
            <h2>Trip impact and counterfactuals</h2>
            <p className="note">
              Impact scores come from impact.json. Counterfactuals answer: what happens if one high-impact trip uses a different Phase 5 candidate? A genuine improvement is not a global optimum.
            </p>
            <div className="row">
              <input className="select" placeholder="Filter trips" value={tripQuery} onChange={(e) => setTripQuery(e.target.value)} />
            </div>
            <div className="card" style={{ overflowX: "auto" }}>
              <table>
                <thead>
                  <tr>
                    <th>Trip</th>
                    <th>Source</th>
                    <th>Dest</th>
                    <th>Impact</th>
                    <th>Critical impact</th>
                    <th>Max util</th>
                  </tr>
                </thead>
                <tbody>
                  {impactTrips.slice(0, 20).map((trip) => (
                    <tr key={String(trip.trip_index)} onClick={() => setSelectedTrip(trip)} style={{ cursor: "pointer" }}>
                      <td>{String(trip.trip_index)}</td>
                      <td>{String(trip.source)}</td>
                      <td>{String(trip.destination)}</td>
                      <td>{formatNum(Number(trip.impact_score))}</td>
                      <td>{formatNum(Number(trip.critical_edge_impact))}</td>
                      <td>{formatNum(Number(trip.max_edge_utilization), 3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {selectedTrip && (
              <div className="detail" style={{ marginTop: 12 }}>
                <strong>Trip {String(selectedTrip.trip_index)}</strong>
                <div>Route: {Array.isArray(selectedTrip.route) ? (selectedTrip.route as number[]).join(" → ") : "Data unavailable"}</div>
                <div>Critical-edge count: {String(selectedTrip.critical_edge_count ?? "Data unavailable")}</div>
              </div>
            )}
            <h3 style={{ marginTop: 24 }}>Counterfactual evaluations</h3>
            <select
              className="select"
              onChange={(e) => {
                const trip = cfTrips.find((item) => String(item.trip_index) === e.target.value) ?? null;
                setCfTrip(trip);
              }}
            >
              <option value="">Select a Phase 9 trip</option>
              {cfTrips.map((trip) => (
                <option key={String(trip.trip_index)} value={String(trip.trip_index)}>
                  trip {String(trip.trip_index)}
                </option>
              ))}
            </select>
            {cfTrip && (
              <div className="card" style={{ marginTop: 12, overflowX: "auto" }}>
                <p className="note">Current route: {Array.isArray(cfTrip.current_route) ? (cfTrip.current_route as number[]).join(" → ") : "Data unavailable"}</p>
                <table>
                  <thead>
                    <tr>
                      <th>Candidate</th>
                      <th>Δ mean</th>
                      <th>Δ p95</th>
                      <th>Δ max cong</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {((cfTrip.candidates as AnyJson[]) ?? []).map((cand) => {
                      const current = Boolean(cand.is_current_route);
                      const better = Boolean(cand.candidate_is_better);
                      return (
                        <tr key={String(cand.candidate_index)}>
                          <td>{String(cand.candidate_index)}</td>
                          <td>{formatNum(Number(cand.delta_mean_travel_time))}</td>
                          <td>{formatNum(Number(cand.delta_p95_travel_time))}</td>
                          <td>{formatNum(Number(cand.delta_max_congestion_ratio), 3)}</td>
                          <td>
                            <span className={`badge ${current ? "neutral" : better ? "improve" : "worse"}`}>
                              {current ? "current route" : better ? "genuine improvement" : "no improvement"}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {section === "optimization" && (
          <div className="section">
            <h2>Optimization progress and explanations</h2>
            <p className="note">
              Phase 11 wraps Phase 10 local search. Baseline-start stop reason: {String(optBaseline.stop_reason ?? "Data unavailable")}.
              Converged: {String(optBaseline.converged ?? "Data unavailable")}. This is not a global optimum.
            </p>
            <div className="grid kpis">
              <Kpi label="Initial mean" value={formatNum(toNumber(asRecord(optBaseline.initial_metrics).mean_travel_time as number))} />
              <Kpi label="Final mean" value={formatNum(toNumber(asRecord(optBaseline.final_metrics).mean_travel_time as number))} />
              <Kpi label="Accepted moves" value={String(optBaseline.accepted_move_count ?? "Data unavailable")} />
              <Kpi label="Stop reason" value={String(optBaseline.stop_reason ?? "Data unavailable")} />
            </div>
            <div className="card">
              <h3>Iteration history (optimizer.json)</h3>
              {progress.length ? (
                <div className="chart">
                  <ResponsiveContainer>
                    <LineChart data={progress}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#2a3548" />
                      <XAxis dataKey="iteration" stroke="#93a0b5" />
                      <YAxis stroke="#93a0b5" />
                      <Tooltip />
                      <Line type="linear" dataKey="mean" stroke="#5b8def" dot={false} name="mean" />
                      <Line type="linear" dataKey="p95" stroke="#e6b84d" dot={false} name="p95" />
                      <Line type="linear" dataKey="maxc" stroke="#e85d5d" dot={false} name="max congestion" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <p className="note">Data unavailable</p>
              )}
            </div>
            <h3 style={{ marginTop: 20 }}>Accepted-move explanations</h3>
            {explanationItems.map((item) => (
              <details key={`${item.iteration}-${item.trip_index}`} className="accordion">
                <summary>
                  Iteration {String(item.iteration)} · trip {String(item.trip_index)} · {String(item.origin)} → {String(item.destination)} · candidate {String(item.candidate_index)}
                </summary>
                <div className="body">
                  <p>{String(item.reason ?? "Data unavailable")}</p>
                  <p><strong>Change:</strong> {(item.old_route as number[] | undefined)?.join(" → ")} → {(item.new_route as number[] | undefined)?.join(" → ")}</p>
                  <p><strong>Bottlenecks:</strong> {String(item.bottleneck_summary ?? "Data unavailable")}</p>
                  <p><strong>Impact:</strong> {String(item.impact_summary ?? "Data unavailable")}</p>
                  <p><strong>Counterfactual:</strong> {String(item.counterfactual_summary ?? "Data unavailable")}</p>
                  <p><strong>Effect:</strong> {String(item.effect ?? "Data unavailable")}</p>
                  <p><strong>Decision:</strong> {String(item.decision ?? "Data unavailable")}</p>
                </div>
              </details>
            ))}
          </div>
        )}

        {section === "robustness" && (
          <div className="section">
            <h2>Small-sample robustness analysis</h2>
            <p className="note">
              Additional deterministic seeds {((robustness?.robustness_seeds as number[]) ?? []).join(", ") || "Data unavailable"}.
              Official seed remains {String(robustness?.official_seed ?? meta.seed)}. This is not statistical significance or universal superiority.
            </p>
            <div className="card" style={{ overflowX: "auto" }}>
              <table>
                <thead>
                  <tr>
                    <th>Seed</th>
                    <th>Algorithm</th>
                    <th>Mean</th>
                    <th>p95</th>
                    <th>Max congestion</th>
                    <th>Accepted moves</th>
                    <th>Stop reason</th>
                  </tr>
                </thead>
                <tbody>
                  {robustCsv.map((row, index) => (
                    <tr key={`${row.seed}-${row.algorithm}-${index}`}>
                      <td>{row.seed}</td>
                      <td>{row.algorithm}</td>
                      <td>{formatNum(toNumber(row.mean_travel_time))}</td>
                      <td>{formatNum(toNumber(row.p95_travel_time))}</td>
                      <td>{formatNum(toNumber(row.max_congestion_ratio), 3)}</td>
                      <td>{row.accepted_moves || "N/A"}</td>
                      <td>{row.stop_reason || "N/A"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="card" style={{ marginTop: 12 }}>
              <h3>Mean travel time across robustness seeds</h3>
              <div className="chart">
                <ResponsiveContainer>
                  <BarChart data={robustCsv.map((row) => ({ ...row, mean_travel_time: Number(row.mean_travel_time) }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#2a3548" />
                    <XAxis dataKey="algorithm" stroke="#93a0b5" />
                    <YAxis stroke="#93a0b5" />
                    <Tooltip />
                    <Bar dataKey="mean_travel_time" fill="#5b8def" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            <p className="note">{String(asRecord(robustness?.summary).conclusion ?? "Data unavailable")}</p>
          </div>
        )}

        {section === "methodology" && (
          <div className="section">
            <h2>Methodology</h2>
            <div className="card note">
              <ol>
                <li><strong>Network.</strong> Official {meta.grid_size}×{meta.grid_size} directed grid. Disruption removes {meta.disrupted_edges.join(" and ")}. {String(asRecord(baseline).number_of_directed_edges)} directed edges remain.</li>
                <li><strong>Demand.</strong> {meta.trips} deterministic trips, seed {meta.seed}, NumPy PCG64.</li>
                <li><strong>Congestion.</strong> {meta.travel_time_formula}. Utilization is flow/capacity.</li>
                <li><strong>Algorithms.</strong> Independent free-flow shortest paths (baseline); sequential congestion-aware greedy; K-shortest candidate generation (comparison uses candidate #1); deterministic local search; optimizer as Phase 10 orchestration.</li>
                <li><strong>Evaluation.</strong> Mean travel time, p95 travel time, maximum congestion ratio.</li>
                <li><strong>Reproducibility.</strong> Fixed seed, Phase 3 verification gate, regression against stored outputs. Dashboard values are loaded from those outputs and are not re-solved here.</li>
              </ol>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
