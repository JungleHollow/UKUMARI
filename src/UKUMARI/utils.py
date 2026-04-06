"""
This utils file should cover any miscellaneous functions that facilitate running the package across modules
"""

from __future__ import annotations

import random

import numpy as np
import rustworkx as rx
import rustworkx.generators as rx_gen  # Workaround for unrecognised module import in base rustworkx


def pygraph_to_pydigraph(input_graph: rx.PyGraph) -> rx.PyDiGraph:
    """
    Transform an arbitrary rx.PyGraph to an rx.PyDiGraph object where each unidirectional edge in the PyGraph becomes two opposing monodirectional edges with weight equal to the original.
    Mainly used exclusively to transform undirected watts_strogatz_graph() returns to directed ones for use in the model.

    :param input_graph: The undirected input graph to be transformed into a directed version.
    :return: A directed version of the input graph where each edge has been transformed into two edges pointing in opposing directions.
    """
    new_graph: rx.PyDiGraph = rx.PyDiGraph()
    for node in input_graph.nodes():
        new_graph.add_node(node)

    for edge in input_graph.weighted_edge_list():
        # 'edge' is a (node_a_index, node_b_index, weight) tuple
        # Given that multiple edges were not allowed at creation, each combination of a and b should be unique
        new_graph.add_edge(edge[0], edge[1], edge[2])  # Edge going from a -> b
        new_graph.add_edge(edge[1], edge[0], edge[2])  # Edge going from b -> a
    return new_graph


def watts_strogatz_graph(
    n: int, k: int, p: float, seed: int | np.random.RandomState | None = None
) -> rx.PyGraph:
    """
    Returns an undirected Watts-Strogatz small-world graph generated using rustworkx.
    An adapted version of watts_strogatz_graph() from the NetworkX library.

    :param n: The number of nodes in the graph.
    :param k: The number of nearest neighbours that each node is joined to initially.
    :param p: The probability of rewiring each edge of the original ring lattice.
    :param seed: The random seed to use for random generation
    """
    random_gen: random.Random | np.random.Generator
    match type(seed):
        case int():
            random_gen = random.Random(seed)
        case np.random.RandomState():
            random_gen = np.random.default_rng()
            np.random.set_state(seed)
        case _:
            random_gen = random.Random()

    if k >= n:
        # >= instead of == as this utility function does not care about accounting for complete graphs...
        raise ValueError(
            "k is larger than or equal to n; choose a smaller k or larger n."
        )

    G: rx.PyGraph = rx_gen.empty_graph(n)
    nodes: list[int] = list(range(n))  # nodes labeled 0 to n-1

    # Connect each node to k/2 neighbours
    for j in range(1, k // 2 + 1):
        targets: list[int] = (
            nodes[j:] + nodes[0:j]
        )  # first j nodes become last in the list
        G.add_edges_from(zip(nodes, targets, [0.0 for _ in range(len(nodes))]))

    # Rewire edges from each node
    # Loop over all nodes in order (label) and neighbours in order (distance)
    # No self loops or multiple edges allowed
    for j in range(1, k // 2 + 1):  # Outer loop is neighbours
        targets = nodes[j:] + nodes[0:j]
        # Inner loop in noder order
        for u, v in zip(nodes, targets):
            if random_gen.random() < p:
                w = random_gen.choice(nodes)
                # Enforce no self loops or multiple edges
                while w == u or G.has_edge(u, w):
                    w = random_gen.choice(nodes)
                    if G.degree(u) >= n - 1:
                        break  # Skip this rewiring
                else:
                    G.remove_edge(u, v)
                    G.add_edge(u, w, 0.0)
    return G


def connected_watts_strogatz_graph(
    n: int,
    k: int,
    p: float,
    tries: int = 100,
    seed: int | np.random.RandomState | None = None,
) -> rx.PyDiGraph:
    """
    Returns a connected, directed Watts-Strogatz small-world graph.
    An adapted version of connected_watts_strogatz_graph() from the NetworkX library.

    :param n: The number of nodes in the graph.
    :param k: The number of nearest neighbours that each node is joined to initially.
    :param p: The probability of rewiring each edge of the original ring lattice.
    :param tries: The number of times to try producing a connected graph after rewiring, before raising an exception.
    :param seed: The random seed to use for random generation.
    """
    for i in range(tries):
        G: rx.PyGraph = watts_strogatz_graph(n, k, p, seed=seed)
        if rx.is_connected(G):
            directed_graph: rx.PyDiGraph = pygraph_to_pydigraph(G)
            return directed_graph
    raise RuntimeError(
        "Exceeded maximum number of tries generating a connected Watts-Strogatz small-world graph..."
    )
