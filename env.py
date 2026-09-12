"""Phase 1 environment: disrupted 5x5 grid and deterministic demand."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

GRID_SIZE = 5
DEFAULT_SEED = 20260911
DEFAULT_NUM_TRIPS = 120
FREE_FLOW_TIME = 1
CAPACITY = 8
MIN_TRIP_DISTANCE = 4
REMOVED_COORD_PAIRS = (((2, 2), (3, 2)), ((3, 2), (2, 2)))


def node_id(x: int, y: int) -> int:
    return GRID_SIZE * y + x


def build_network() -> nx.DiGraph:
    """Build a 5x5 bidirectional grid with the official disruption applied."""
    graph = nx.DiGraph()

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            graph.add_node(node_id(x, y), x=x, y=y)

    def add_directed_edge(u: int, v: int) -> None:
        graph.add_edge(u, v, free_flow_time=FREE_FLOW_TIME, capacity=CAPACITY)

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            u = node_id(x, y)
            if x + 1 < GRID_SIZE:
                v = node_id(x + 1, y)
                add_directed_edge(u, v)
                add_directed_edge(v, u)
            if y + 1 < GRID_SIZE:
                v = node_id(x, y + 1)
                add_directed_edge(u, v)
                add_directed_edge(v, u)

    for (x1, y1), (x2, y2) in REMOVED_COORD_PAIRS:
        graph.remove_edge(node_id(x1, y1), node_id(x2, y2))

    return graph


def generate_trips(
    graph: nx.DiGraph,
    seed: int = DEFAULT_SEED,
    num_trips: int = DEFAULT_NUM_TRIPS,
) -> list[tuple[int, int]]:
    """Sample demand on the disrupted graph using NumPy PCG64."""
    rng = np.random.Generator(np.random.PCG64(seed))
    nodes = np.array(sorted(graph.nodes()), dtype=int)
    distances = dict(nx.all_pairs_shortest_path_length(graph))

    trips: list[tuple[int, int]] = []
    while len(trips) < num_trips:
        origin = int(rng.choice(nodes))
        destination = int(rng.choice(nodes))
        if origin == destination:
            continue
        if distances[origin][destination] >= MIN_TRIP_DISTANCE:
            trips.append((origin, destination))
    return trips


@dataclass(frozen=True)
class Environment:
    graph: nx.DiGraph
    trips: list[tuple[int, int]]
    seed: int


def create_environment(
    seed: int = DEFAULT_SEED,
    num_trips: int = DEFAULT_NUM_TRIPS,
) -> Environment:
    graph = build_network()
    trips = generate_trips(graph, seed=seed, num_trips=num_trips)
    return Environment(graph=graph, trips=trips, seed=seed)


def _trip_distances_ok(graph: nx.DiGraph, trips: list[tuple[int, int]]) -> bool:
    distances = dict(nx.all_pairs_shortest_path_length(graph))
    return all(
        origin != destination and distances[origin][destination] >= MIN_TRIP_DISTANCE
        for origin, destination in trips
    )


def _disrupted_edges_absent(graph: nx.DiGraph) -> bool:
    return all(
        not graph.has_edge(node_id(x1, y1), node_id(x2, y2))
        for (x1, y1), (x2, y2) in REMOVED_COORD_PAIRS
    )


if __name__ == "__main__":
    env = create_environment()
    print("nodes:", env.graph.number_of_nodes())
    print("directed_edges:", env.graph.number_of_edges())
    print("trips:", len(env.trips))
    print("seed:", env.seed)
    print("first_trips:", env.trips[:10])
    print("all_trip_distances_ge_4:", _trip_distances_ok(env.graph, env.trips))
    print("disrupted_edges_absent:", _disrupted_edges_absent(env.graph))
    print("removed_node_ids:", (node_id(2, 2), node_id(3, 2)))
