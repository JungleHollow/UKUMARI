"""
This utils file should cover any miscellaneous functions that facilitate running the package across modules
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import rustworkx as rx
import rustworkx.generators as rx_gen  # Workaround for unrecognised module import in base rustworkx
from scipy.stats import beta, truncnorm, gamma, levy, uniform, norm

# ========== Graph utils ========== #


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


# ========== Math utils ========== #


def beta_value_attenuation(input_value: float, a: float = 0.9, b: float = 0.9) -> Any:
    """
    Takes an input value and rescales it using a beta distribution. This function is intended to be used exclusively
    for the attenuation of indirect neighbouring opinions when an agent is estimating an opinion climate in a hierarchy.

    Importantly, the output values are not used for constructing the opinion climate, instead they serve as a threshold
    used to decide if the original opinion values are included or not.

    a and b should generally be equal to each other and less than 1.0 to approximate a binomial distribution whose PDF still has
    non-zero values in the range (0, 1). This is to enable modeling of the tendency for opinions to become polarised over time;
    with this distribution meaning that agents will place much higher importance on extreme observed opinions and less importance
    on moderate opinions.

    :param input_value: The value to be attenuated. Should always be in the range [-1, 1]
    :param a: The alpha parameter for the beta distribution.
    :param b: The beta parameter for the beta distribution.
    :return: The attenuated input value.
    """
    original_opinion: float = input_value

    # Shift the range of input_value from [-1, 1] to [0, 1]
    input_value = (input_value / 2.0) + 0.5

    # Constrain to (0, 1) to prevent the beta pdf from reaching infinity
    if input_value == 0.0:
        input_value = 0.001
    elif input_value == 1.0:
        input_value = 0.999

    # Define the beta function and find the upper bound of its PDF
    beta_func = beta(a, b)
    upper_bound: float = beta_func.pdf(0.001)

    # Calculate the beta PDF of the input value
    beta_value: float = beta_func.pdf(input_value)

    # Normalise the beta value using the upper bound of the PDF
    attenuation_factor: float = beta_value / upper_bound

    # Rescale the original value by the attenuation factor
    attenuated_opinion: float = original_opinion * attenuation_factor

    # Check if range has to be constrained (float operations)
    if attenuated_opinion < -1.0:
        attenuated_opinion = -1.0
    elif attenuated_opinion > 1.0:
        attenuated_opinion = 1.0

    return attenuated_opinion


# ========== Random Generation Utils ==========

def draw_random_value(distribution: str, parameters: dict | None = None) -> float:
    """
    Utility function that handles random value generation from multiple distributions in the same function.

    All values generated by this function will be in the range [0, 1], with any necessary scaling ocurring in
    the calling functions.

    Note, "scale" and "loc" parameters should still be included as their default values (0 for loc, and 1 for scale)
    when calling with parameters to prevent dictionary key errors.

    :param distribution: The name of the distribution to draw from.
    :param parameters: A dictionary that contains any relevant parameters to be specified for a given distribution.
    :return: A float value drawn from the random distribution.
    """
    drawn_value: float = 0.0

    match distribution:
        case "gaussian":
            if parameters:
                drawn_value = truncnorm.rvs(parameters["a"], parameters["b"], loc=parameters["loc"], scale=parameters["scale"])
            else:
                drawn_value = truncnorm.rvs(0.0, 1.0)
        case "beta":
            if parameters:
                drawn_value = beta.rvs(parameters["a"], parameters["b"], loc=parameters["loc"], scale=parameters["scale"])
            else:
                drawn_value = beta.rvs(1.0, 1.0)
        case "levy":
            if parameters:
                drawn_value = levy.rvs(loc=parameters["loc"], scale=parameters["scale"])
            else:
                drawn_value = levy.rvs()
        case "uniform":
            if parameters:
                drawn_value = uniform.rvs(loc=parameters["loc"], scale=parameters["scale"])
            else:
                drawn_value = uniform.rvs()
        case "gamma":
            if parameters:
                drawn_value = gamma.rvs(parameters["a"], loc=parameters["loc"], scale=parameters["scale"])
            else:
                drawn_value = gamma.rvs(1.0)
    return drawn_value

# ========== Random Walk Utils ==========

def value_rw_delta(input_value: float, mean: float, variance: float) -> float:
    """
    Draws a random walk delta from a normal distribution with the specified parameters, and then adds this delta
    to the input value before returning.

    :param input_value: The value from which the random walk begins.
    :param mean: The mean of the normal distribution from which the delta is drawn.
    :param variance: The variance of the normal distribution from which the delta is drawn.
    :return: The result of the random walk.
    """
    rw_delta: float = norm.rvs(loc=mean, scale=variance)
    rw_result: float = input_value + rw_delta
    return rw_result