import { nodeCoord, utilizationClass } from "../data/load";

type Props = {
  flows: Record<string, number>;
  capacity: number;
  disrupted: string[];
};

function color(util: number | null): string {
  if (util === null) return "#4a5568";
  const kind = utilizationClass(util);
  if (kind === "over") return "#e85d5d";
  if (kind === "near") return "#e6b84d";
  return "#3ecf8e";
}

export default function NetworkMap({ flows, capacity, disrupted }: Props) {
  const grid = 5;
  const size = 420;
  const pad = 36;
  const step = (size - pad * 2) / (grid - 1);
  const pos = (id: number) => {
    const { x, y } = nodeCoord(id, grid);
    return { x: pad + x * step, y: pad + (grid - 1 - y) * step };
  };
  const edges: Array<[number, number]> = [];
  for (let y = 0; y < grid; y += 1) {
    for (let x = 0; x < grid; x += 1) {
      const u = grid * y + x;
      if (x < grid - 1) edges.push([u, u + 1], [u + 1, u]);
      if (y < grid - 1) edges.push([u, u + grid], [u + grid, u]);
    }
  }
  const disruptedSet = new Set(disrupted);

  return (
    <div className="network-wrap">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label="5 by 5 disrupted grid">
        {edges.map(([u, v]) => {
          const a = pos(u);
          const b = pos(v);
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const len = Math.hypot(dx, dy) || 1;
          const ox = (-dy / len) * 5;
          const oy = (dx / len) * 5;
          const key = `${u}->${v}`;
          const missing = disruptedSet.has(key);
          const util = missing || !(key in flows) ? null : flows[key] / capacity;
          return (
            <line
              key={key}
              x1={a.x + ox}
              y1={a.y + oy}
              x2={b.x + ox}
              y2={b.y + oy}
              stroke={missing ? "#e85d5d" : color(util)}
              strokeDasharray={missing ? "5 4" : undefined}
              strokeWidth={missing ? 2.2 : 1.4 + (util ?? 0)}
              opacity={0.95}
            />
          );
        })}
        {Array.from({ length: 25 }, (_, id) => {
          const p = pos(id);
          return (
            <g key={id}>
              <circle cx={p.x} cy={p.y} r={14} fill="#141b27" stroke="#5b8def" />
              <text x={p.x} y={p.y + 4} textAnchor="middle" fill="#e8eef8" fontSize="10">
                {id}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
