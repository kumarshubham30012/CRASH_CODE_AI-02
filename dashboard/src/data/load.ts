export type ComparisonRow = {
  algorithm: string;
  trips: number;
  mean_travel_time: number;
  p95_travel_time: number;
  max_congestion_ratio: number;
  runtime_seconds: number | null;
};

export type Meta = {
  grid_size: number;
  seed: number;
  trips: number;
  nodes: number;
  capacity: number;
  free_flow_time: number;
  disrupted_edges: string[];
  travel_time_formula: string;
};

export async function fetchText(path: string): Promise<string> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load ${path} (${response.status})`);
  }
  return response.text();
}

export async function fetchJson<T>(path: string): Promise<T> {
  const text = await fetchText(path);
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new Error(`Malformed JSON: ${path}`);
  }
}

export function parseCsv(text: string): Record<string, string>[] {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    const row: Record<string, string> = {};
    headers.forEach((header, index) => {
      row[header] = values[index] ?? "";
    });
    return row;
  });
}

export function toNumber(value: string | number | null | undefined): number | null {
  if (value === null || value === undefined || value === "") return null;
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : null;
}

export function formatNum(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "Data unavailable";
  return value.toFixed(digits);
}

export function parseComparison(text: string): ComparisonRow[] {
  return parseCsv(text).map((row) => ({
    algorithm: row.algorithm,
    trips: Number(row.trips),
    mean_travel_time: Number(row.mean_travel_time),
    p95_travel_time: Number(row.p95_travel_time),
    max_congestion_ratio: Number(row.max_congestion_ratio),
    runtime_seconds: toNumber(row.runtime_seconds),
  }));
}

export function bestReported(rows: ComparisonRow[], key: keyof ComparisonRow): number | null {
  const values = rows
    .map((row) => row[key])
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (!values.length) return null;
  return Math.min(...values);
}

export function utilizationClass(util: number): "ok" | "near" | "over" {
  if (util > 1) return "over";
  if (util >= 0.8) return "near";
  return "ok";
}

export function nodeCoord(id: number, grid = 5): { x: number; y: number } {
  return { x: id % grid, y: Math.floor(id / grid) };
}
