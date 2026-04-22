from __future__ import annotations

import contextlib
import os
import warnings
import zipfile
from collections.abc import Iterable
from copy import deepcopy
from random import Random
from shutil import rmtree
from typing import Any, Iterator, override

import numpy as np
import polars as pl
import rustworkx as rx

from .agents import Agent
from .utils import (
    beta_value_attenuation,
    connected_watts_strogatz_graph,
    value_rw_delta,
)


class GraphNode:
    """
    A helper class that allows rustworx to more efficiently store information about Agents in the graph nodes
    """

    def __init__(self, agent: Agent) -> None:
        """
        :param agent: The Agent object that is being associated with this GraphNode
        """
        self.index: int
        self.agent: Agent = agent

    def set_index(self, idx: int) -> None:
        """
        A setter method to set the GraphNode's index value.

        :param idx: The index to set for this GraphNode.
        """
        self.index = idx
        return None

    @override
    def __str__(self) -> str:
        """
        An override of what calling 'print()' on a GraphNode object will output.
        """
        return f"Agent ({self.agent.id}) at graph node ({self.index})"


class GraphEdge:
    """
    A helper class that allows rustworx to more efficiently store information about Agent relationships in the graph edges.
    As the social hierarchies are assumed to be DiGraphs, each GraphEdge is directional, and the social weighting that Agent
    A places on Agent B will not necessarilly be equally reciprocated.
    """

    def __init__(
        self,
        hierarchy: str,
        from_node: int,
        to_node: int,
        weighting: float = 0.0,
    ) -> None:
        """
        :param hierarchy: The name of the social hierarchy that this edge belongs to
        :param from_node: The index of the starting node
        :param to_node: The index of the destination node
        :param weighting: The opinion weighting that is being assigned
        """
        self.index: int
        self.weighting: float = weighting
        self.from_node: int = from_node
        self.to_node: int = to_node
        self.hierarchy: str = hierarchy

    def set_index(self, idx: int) -> None:
        """
        A setter function that changes this GraphEdge's index value.

        :param idx: The index to store for this GraphEdge.
        """
        self.index = idx
        return None

    def set_weighting(self, value: float) -> None:
        """
        A setter function that changes this GraphEdge's weighting value.

        :param value: The new weighting to store for this GraphEdge.
        """
        self.weighting = value
        return None

    def update_from_node(self, idx: int) -> None:
        """
        A setter function that updates the from_node's index for this GraphEdge.

        :param idx: The from_node's new index value to update to.
        """
        self.from_node = idx
        return None

    def update_to_node(self, idx: int) -> None:
        """
        A setter function that updates the to_node's index for this GraphEdge.

        :param idx: The to_node's new index value to update to.
        """
        self.to_node = idx
        return None

    @override
    def __str__(self) -> str:
        """
        An override of what calling 'print()' on a GraphEdge object will output.
        """
        return f"GraphEdge of weight ({self.weighting}) from node ({self.from_node}) to node ({self.to_node}) in the {self.hierarchy} social layer"


class Graph:
    """
    A graph class that defines a single agent-based model layer.
    This corresponds to the agents' attitudes towards one another
    with respect to different social hierarchies.
    """

    def __init__(self, name: str, rw_params: tuple[float, float]) -> None:
        """
        :param name: The name of the social hierarchy that this Graph object will be representing
        :param rw_params: The (mean, variance) of the normal distribution used for the dynamic relationships random walk
        """
        # Defined as DiGraph as it is common in social networks for relationships to be unidirectional or unbalanced
        self.graph: rx.PyDiGraph = rx.PyDiGraph()
        self.node_count: int = 0
        self.edge_count: int = 0
        self.name: str = name
        self.rw_params: tuple[float, float] = rw_params
        self.generation_params: dict[
            str, Any
        ] = {  # Used for random graph generation, can be manually set by the user if desired
            "p": 0.4,
            "m": 3,
            "sbm_sizes": 10,
        }

    def change_generation_params(self, **params) -> None:
        """
        Setter function which outlines the existing generation parameters used in generate_graph()
        and allows the user to alter them.

        :param p: The probability of edge rewiring (small-world) or edge creation (random).
        :param m: The number of nearest neighbours that each node is connected to initially (scale-free).
        :param: sbm_sizes: The size of generated blocks (blockmodel).
        """
        for key, value in params:
            if (
                key not in self.generation_params.keys()
            ):  # Skip any invalid parameters which have been passed
                warnings.warn(
                    f"WARNING: Invalid graph generation parameter ({key}) specified when trying to modify parameter values.",
                    category=UserWarning,
                )
                continue
            elif (
                type(self.generation_params[key]) is not type(value)
            ):  # Skip altering any parameters which have been assigned invalid data types
                warnings.warn(
                    f"WARNING: Invalid data type detected for the value when modifying parameter {key}.",
                    category=UserWarning,
                )
                continue
            self.generation_params[key] = value
        return None

    def load_graph(
        self, path: str, name: str, rw_params: tuple[float, float] | None = None
    ) -> None:
        """
        Loads a Graph object stored in the GraphML format from the given path.
        The social hierarchy name must be explicitly passed with this call.

        :param path: Path to a stored graph file.
        :param name: The name of the hierarchy that the stored Graph belongs to.
        :param rw_params: The mean and variance of the Graph's random walk distribution (optional).
        """
        graph: list[Any] = rx.read_graphml(path)
        self.graph = graph[0]
        self.node_count = len(self.graph.nodes())
        self.edge_count = len(self.graph.edges())
        self.name = name

        if rw_params:
            self.rw_params = rw_params

        return None

    def save_graph(self, path: str) -> None:
        """
        Saves the existing Graph object in the GraphML format to the given path.

        :param path: Path to which the Graph will be saved.
        """
        rx.write_graphml(self.graph, path)

        if not os.path.exists(path):
            raise EnvironmentError(f"Failed to write graph {self.name} to path: {path}")

        return None

    def get_node(self, node_index: int) -> Any:
        """
        A getter function to access GraphNode objects.

        :param node_index: The index of the node to access.
        :return: The GraphNode object if the index was valid, or None otherwise.
        """
        try:
            return self.graph.nodes()[node_index]
        except IndexError:
            warnings.warn(
                f"WARNING: Node with index {node_index} is out of bounds for graph {self.name} with {self.node_count} total nodes.",
                category=RuntimeWarning,
            )
            return None

    def get_edge(self, edge_index: int) -> Any:
        """
        A getter function to access GraphEdge objects.

        :param edge_index: The index of the edge to access.
        :return: The GraphEdge object if the index was valid, or None otherwise.
        """
        try:
            return self.graph.edges()[edge_index]
        except IndexError:
            warnings.warn(
                f"WARNING: Edge with index {edge_index} is out of bounds for graph {self.name} with {self.edge_count} total edges.",
                category=RuntimeWarning,
            )
            return None

    def update_node_indices(self) -> None:
        """
        Iterates over all the existing nodes in the graph and updates their stored indices to reflect the current graph state.
        Will also update the graph node_count attribute
        """
        for index in self.graph.node_indices():
            self.graph[index].set_index(index)
        self.update_edge_indices()
        self.node_count = len(self.graph.nodes())
        return None

    def add_nodes(self, agents: Iterable[Agent]) -> None:
        """
        Creates appropriate GraphNodes from the given Agents, and then adds these to the graph.

        :param agents: The Agent objects that will be converted to GraphNodes and added to the graph
        """
        nodes: list[GraphNode] = []
        for agent in agents:
            agent_node = GraphNode(agent)
            nodes.append(agent_node)

        self.graph.add_nodes_from(nodes)
        self.update_node_indices()
        return None

    def update_edge_indices(self) -> None:
        """
        Iterates over all the existing edges in the graph and updates their stored indices to reflect the current graph state.
        Will also update the graph edge_count attribute
        """
        for idx, data in self.graph.edge_index_map().items():
            graph_edge: GraphEdge = deepcopy(data[2])
            if (
                type(graph_edge) is list
            ):  # Workaround for unknown error where a list of a single GraphEdge is added to the base graph at some point
                graph_edge = graph_edge[0]
            graph_edge.set_index(idx)
            self.graph.update_edge_by_index(idx, graph_edge)
        self.edge_count = len(self.graph.edges())
        return None

    def add_edges(self, edges: dict) -> None:
        """
        Creates appropriate GraphEdges from the given dictionary and then adds these to the graph.

        :param edges: A dictionary of key-list pairs where each key corresponds to (from_node, to_node, [optional] weighting, [optional] name)
        """
        graph_edges: list[tuple[int, int, GraphEdge]] = []
        from_nodes: list[int] = edges["from_node"]
        to_nodes: list[int] = edges["to_node"]
        weightings: list[float] | None = None
        if "weighting" in edges.keys():
            weightings = edges["weighting"]

        # Used in case that explicit hierarchy names are set per edge (in the case of the mixed-hierarchy base graph in the model for example)
        names: list[str] | None = None
        if "name" in edges.keys():
            names = edges["name"]

        # Declare the data type of 'edge'
        edge: GraphEdge

        if weightings:
            if names:
                for i in range(len(from_nodes)):
                    edge = GraphEdge(
                        names[i], from_nodes[i], to_nodes[i], weightings[i]
                    )
                    graph_edges.append((from_nodes[i], to_nodes[i], deepcopy(edge)))
            else:
                for i in range(len(from_nodes)):
                    edge = GraphEdge(
                        self.name, from_nodes[i], to_nodes[i], weightings[i]
                    )
                    graph_edges.append((from_nodes[i], to_nodes[i], deepcopy(edge)))
        else:
            if names:
                for i in range(len(from_nodes)):
                    edge = GraphEdge(names[i], from_nodes[i], to_nodes[i])
                    graph_edges.append((from_nodes[i], to_nodes[i], deepcopy(edge)))
            else:
                for i in range(len(from_nodes)):
                    edge = GraphEdge(self.name, from_nodes[i], to_nodes[i])
                    graph_edges.append((from_nodes[i], to_nodes[i], deepcopy(edge)))

        self.graph.add_edges_from(graph_edges)
        self.update_edge_indices()
        return None

    def generate_graph(
        self,
        agents: list[Agent],
        method: str = "small-world",
        relationship_range: tuple[float, float] = (-1.0, 1.0),
    ) -> Any:
        """
        Randomly generate edges between existing Graph nodes and add them to the graph.

        :param agents: The subset of Agents in the base model that are being used as the nodes for this graph.
        :param method: The random generation method to use. Possible choices include: 'small-world', 'scale-free', 'random', 'blockmodel'; Defaults to 'small-world'.
        :param relationship_range: The valid range of generated relationship strengths (at most, constrained to [-1.0, 1.0]).
        :return: A reference to this Graph object.
        """
        if len(agents) <= 0:
            raise ValueError(
                f"Attempting to generate random graph for hierarchy '{self.name}' without passing any valid Agents."
            )

        n: int = len(agents)
        generated_graph: rx.PyDiGraph = rx.PyDiGraph()  # Initialise an empty graph for predictable behaviour in case of assignation errors
        random_gen: Random = (
            Random()
        )  # Initialise a random generator instance for this function

        match method:
            case "small-world":
                # Watts-Strogatz
                k: int = int(
                    np.ceil(np.log(n))
                )  # The smallest integer which is larger than log(n) to guarantee graph connectivity
                generated_graph = connected_watts_strogatz_graph(
                    n, k, self.generation_params["p"]
                )
            case "scale-free":
                # Barbasi-Albert
                generated_graph = rx.directed_barabasi_albert_graph(
                    n, self.generation_params["m"]
                )
            case "random":
                # Erdos-Renyi
                generated_graph = rx.directed_gnp_random_graph(
                    n, self.generation_params["p"]
                )
            case "blockmodel":
                # Holland et al.
                sbm_remainder: int = (
                    n % self.generation_params["sbm_sizes"]
                )  # Determine if there will be any remainder with the specified block size
                sbm_n_blocks: int = (
                    len(agents) // self.generation_params["sbm_sizes"]
                )  # Determine how many blocks will be created
                sbm_sizes: list[int] = [
                    self.generation_params["sbm_sizes"] for _ in range(sbm_n_blocks)
                ]
                sbm_sizes[-1] += (
                    sbm_remainder  # If any agents are left over, add them all to the last block
                )

                sbm_probabilities: np.ndarray = np.zeros(
                    (sbm_n_blocks, sbm_n_blocks), dtype=np.float64
                )  # Initialise a BxB array to hold the probabilities for inter-block connections
                for i in range(sbm_probabilities.shape[0]):
                    for j in range(sbm_probabilities.shape[1]):
                        sbm_probabilities[i, j] = (
                            random_gen.random()
                        )  # Set a random probability for edge connectivity from block i to block j (directed, asymmetrical)

                generated_graph = rx.directed_sbm_random_graph(
                    sbm_sizes, sbm_probabilities, False
                )  # "False" to disallow existence of self loops in the graph
            case _:
                raise ValueError(
                    f"Attempting to generate random graph with a non-supported method ({method}).\n\nUse one of the supported methods: 'small-world', 'scale-free', 'random', or 'blockmodel'..."
                )

        graph_nodes: list[GraphNode] = []
        for index, node in enumerate(generated_graph.nodes()):
            graph_node: GraphNode = GraphNode(deepcopy(agents[index]))
            graph_node.set_index(index)
            graph_nodes.append(graph_node)
        for idx, graph_node in enumerate(graph_nodes):
            generated_graph[idx] = (
                graph_node  # Update all the graph nodes with the new GraphNode data objects
            )

        self.graph = generated_graph  # Store the generated graph as the object's "graph" attribute (with 0.0 relationship weights currently)

        for index, edge in generated_graph.edge_index_map().items():
            generated_value = random_gen.uniform(
                relationship_range[0], relationship_range[1]
            )  # Generate a random value in the specified range (default is [-1.0, 1.0])

            graph_edge: GraphEdge = GraphEdge(
                self.name, edge[0], edge[1], weighting=generated_value
            )

            self.graph.update_edge_by_index(
                index, deepcopy(graph_edge)
            )  # Update the edge with a GraphEdge object

        # Update the node and edge counts manually as no call to update_x_indices() have been made
        self.node_count = len(self.graph.nodes())
        self.edge_count = len(self.graph.edges())

        return self

    def relationship_exists(self, from_node: int, to_node: int) -> int | None:
        """
        Checks for the existence of a relationship (weighted edge) between two Agents (nodes).

        :param from_node: the node index of the parent node.
        :param from_node: the node index of the child node.
        :return: The index of the edge if the relationship exists, or None otherwise
        """
        for edge in self.graph.edges():
            if edge.from_node == from_node and edge.to_node == to_node:
                return edge.index
        return None

    def get_relationships(
        self, node_1: int, node_2: int
    ) -> dict[tuple[int, int], float] | None:
        """
        Retrieves and reports the bidirectional relationship weightings between two nodes in the Graph.

        :param node_1: the node index of Agent 1.
        :param node_2: the node index of Agent 2.
        :return: Dictionary with the bidirectional edge weightings between two nodes (if they exist).
        """
        if not self.relationship_exists(node_1, node_2):
            return None

        relationships_dict: dict[tuple[int, int], float] = {}

        with contextlib.suppress(KeyError):
            relationships_dict[(node_2, node_1)] = self.graph.adj_direction(
                node_1, True
            )[node_2]

        with contextlib.suppress(KeyError):
            relationships_dict[(node_1, node_2)] = self.graph.adj_direction(
                node_1, False
            )[node_2]

        return relationships_dict

    def get_relationship(self, from_node: Agent, to_node: Agent) -> float:
        """
        Return a directed relationship from one node to another.

        :param from_node: The node that the relationship originates from.
        :param to_node: The node that the relationship points to.
        :return: The weighting of the directed relationship (from_node -> to_node).
        """
        relationship_dict: dict[int, Any] = self.graph.adj_direction(
            self.get_agent_index(from_node), False
        )
        graph_edge: GraphEdge = relationship_dict[self.get_agent_index(to_node)]
        return graph_edge.weighting

    def change_weights(self, node_1: int, node_2: int, value: float) -> None:
        """
        Updates the weight of the relationship between two agents in the graph.
        If no relationship previously exists, a new one is created.

        :param node_1: The index of some Agent in the graph.
        :param node_2: The index  of some other Agent in the graph.
        :param value: The new weight to assign.
        """
        edge_index: int | None = self.relationship_exists(node_1, node_2)
        updated_edge: list[Any] = [GraphEdge(self.name, node_1, node_2, value)]
        if edge_index is not None:
            self.graph.update_edge_by_index(edge_index, updated_edge)
        else:
            self.graph.add_edges_from(updated_edge)
        self.update_edge_indices()
        return None

    def remove_node(self, node: int) -> None:
        """
        Removes a node from the graph, along with any relationships involving it.

        :param node: The node index to remove from the graph.
        """
        self.graph.remove_node(node)

        edges_to_remove = []
        for edge in self.graph.edges():
            if edge.from_node == node or edge.to_node == node:
                edges_to_remove.append((edge.from_node, edge.to_node))

        for edge in edges_to_remove:
            self.remove_edge(edge[0], edge[1])
        # No need to update indices, as rustworkx will automatically add new nodes/edges into the largest empty index
        return None

    def remove_edge(self, from_node: int, to_node: int) -> None:
        """
        Removes a single edge from the graph.
        Throws a warning if the edge did not exist in the first place.

        :param from_node: the parent node in the edge.
        :param to_node: the child node in the edge.
        """
        edge_exists: int | None = self.relationship_exists(from_node, to_node)
        if edge_exists:
            self.graph.remove_edge(from_node, to_node)
        else:
            warnings.warn(
                f"WARNING: Attempted to remove edge ({from_node} -> {to_node}) which does not exist in the graph.",
                category=UserWarning,
            )
        return None

    def agent_in_graph(self, agent: Agent) -> bool:
        """
        A simple function that checks wether an Agent exists within a Graph.

        :param agent: the Agent whose existence in the Graph is being checked for.
        :return: A boolean indicating if the Agent exists in the Graph.
        """
        for node in self.graph.nodes():
            if agent.id == node.agent.id:
                return True
        return False

    def agent_previous_opinion(self, agent: Agent) -> None:
        """
        Set the specified Agent's previous opinion to be equal to the current opinion (before the current opinion changes in the current iteration).

        :param agent: The Agent whose previous opinion is being set.
        """
        agent_node: Any = self.node_from_agent(agent)
        agent_node.agent.store_previous_opinion()
        return None

    def agent_opinion_change(self, agent: Agent, change_delta: float) -> None:
        """
        Changes the specified Agent's current opinion by the given delta.

        :param agent: The Agent whose current opinion is being changed.
        :param change_delta: The value by which to change the Agent's current opinion.
        """
        agent_node: Any = self.node_from_agent(agent)
        agent_node.agent.change_opinion(change_delta)
        return None

    def agent_radicalisation_change(self, agent: Agent, radicalisation: bool) -> None:
        """
        Change the specified Agent's radicalisation status.

        :param agent: The Agent whose radicalisation status is being changed.
        :param radicalisation: The boolean radicalisation status.
        """
        agent_node: Any = self.node_from_agent(agent)
        agent_node.agent.change_radicalisation(radicalisation)
        return None

    def node_from_agent(self, agent: Agent) -> Any:
        """
        Returns the GraphNode object corresponding to the given Agent object.

        :param agent: The Agent object being searched for in the GraphNodes.
        :return: The GraphNode object corresponding to the input Agent.
        """
        agent_index: int = self.get_agent_index(agent)
        agent_node: Any = self.get_node(agent_index)
        return agent_node

    def get_agent_index(self, agent: Agent) -> int:
        """
        Searches for the node index in the Graph which corresponds to the input Agent object.

        :param agent: The Agent object whose index is being searched for.
        :return: The Agent's node index within the social hierarchy Graph.
        """
        for idx, node in enumerate(self.graph.nodes()):
            if agent.id == node.agent.id:
                return idx
        return 0

    def get_neighbours(self, agent: Agent) -> list[GraphNode]:
        """
        Finds all the nodes in the graph with direct relationships to the specified Agent.

        :param agent: The Agent for which the neighbours are being examined.
        :return: A list of the GraphNode objects belonging to the direct neighbours of the agent.
        """
        neighbour_nodes: list[GraphNode] = []
        agent_index: int = self.get_agent_index(agent)
        neighbour_indices: rx.rustworkx.NodeIndices = self.graph.neighbors(agent_index)
        for index in neighbour_indices:
            neighbour_node: GraphNode = self.get_node(index)
            neighbour_nodes.append(neighbour_node)
        return neighbour_nodes

    def step(self) -> None:
        """
        Step the individual Graph object:
            1. Handle dynamic relationships within the graph.
        """
        self.dynamic_relationships()
        return None

    def neighbour_influences(self, agent: Agent) -> float:
        """
        Looks at all the neighbours for an Agent and uses the neighbours' own opinions plus the
        weight of the relationship between Agents to return a final value by which the given
        Agent's opinion value will increment or decrement.

        :param agent: The Agent for which the strength of opinion change is being determined.
        :return: The final change in the Agent's opinion caused by their neighbours in this hierarchy.
        """
        agent_hierarchy_weighting: float = agent.social_weightings[self.name]
        agent_index: int = self.get_agent_index(agent)
        neighbour_indices: rx.NodeIndices = self.graph.neighbors(agent_index)

        final_change: float = 0.0
        for neighbour_index in neighbour_indices:
            relationship_strength: float = self.get_relationship(
                agent, self.get_node(neighbour_index).agent
            )
            neighbour_node: GraphNode = self.get_node(neighbour_index)

            average_opinion: float = (
                agent.opinion + neighbour_node.agent.opinion
            ) / 2.0  # Simple average of own and neighbour opinions
            distance_from_avg: float = (
                average_opinion - agent.opinion
            )  # The delta that must be applied to own opinion to reach the average
            weighted_delta: float = (
                distance_from_avg * agent_hierarchy_weighting * relationship_strength
            )  # The final opinion change

            final_change += weighted_delta
        return final_change

    def dynamic_relationships(self) -> None:
        """
        Uses the (mean, variance) passed at initialisation to draw random walk values by which each edge (relationship)
        in the hierarchy will be shifted. Aims to simulate dynamic relationships between agents across timesteps.
        """
        for edge in self.graph.edges():
            new_weighting: float = value_rw_delta(
                edge.weighting, self.rw_params[0], self.rw_params[1]
            )

            # Constrain the weighting back to [-1.0, 1.0] as needed
            if new_weighting < -1.0:
                new_weighting = -1.0
            elif new_weighting > 1.0:
                new_weighting = 1.0

            edge.set_weighting(new_weighting)
        return None

    def estimate_neighbour_opinions(self, agent: Agent) -> dict[str, float]:
        """
        Return the individual opinion climate values perceived by the Agent for each other Agent within this social hierarchy.

        :param agent: The Agent object which is estimating its neighbours' opinions.
        :return: A list of the Agent's perceived opinion values held by each of its hierarchy neighbours.
        """
        observed_opinions: dict[str, float] = {}

        direct_neighbours: list[GraphNode] = self.get_neighbours(agent)
        for direct_neighbour in direct_neighbours:
            observed_opinion: float = deepcopy(direct_neighbour.agent.opinion)
            observed_opinions[direct_neighbour.agent.id] = observed_opinion

        for node in self.graph.nodes():
            if node.agent.id == agent.id or node in direct_neighbours:
                # Only look at indirect neighbours
                continue

            raw_observed_opinion: float = deepcopy(node.agent.opinion)
            attenuated_opinion: float = beta_value_attenuation(raw_observed_opinion)

            if -0.5 > attenuated_opinion > 0.5:
                observed_opinions[node.agent.id] = raw_observed_opinion

        return observed_opinions

    def estimate_opinion_climate(self, agent: Agent) -> float:
        """
        Return the unique opinion climate perceived by the Agent within this social hierarchy.

        :param agent: The Agent object which is estimating the opinion climate.
        :return: The Agent's perceived `aggregated opinion' of this whole social hierarchy.
        """
        observed_opinions: list[
            float
        ] = []  # The observed opinions of the agent's direct neighbours and the relevant observed opinions of indirect neighbours

        direct_neighbours: list[GraphNode] = self.get_neighbours(agent)
        for direct_neighbour in direct_neighbours:
            observed_opinion: float = deepcopy(direct_neighbour.agent.opinion)
            observed_opinions.append(observed_opinion)

        for node in self.graph.nodes():
            if node.agent.id == agent.id or node in direct_neighbours:
                # Only look at indirect neighbours
                continue

            raw_observed_opinion: float = deepcopy(node.agent.opinion)
            attenuated_opinion: float = beta_value_attenuation(raw_observed_opinion)

            if (
                -0.5 > attenuated_opinion > 0.5
            ):  # Only take values which are still relevant after attenuation (i.e. values stronger than an absolute 0.5 after attenuation)
                observed_opinions.append(
                    raw_observed_opinion
                )  # NOT the attenuated opinion, as that would funamentally alter the nature of the opinion climate

        summed_opinions: float = sum(
            observed_opinions
        )  # Sum all of the observed opinions

        if len(observed_opinions) >= 1:
            opinion_climate: float = summed_opinions / float(
                len(observed_opinions)
            )  # Find the average of the aggregated, relevant opinions
            return opinion_climate
        else:
            return 0.0

    def calculate_polarisation(self) -> float:
        r"""
        Calculates the level of opinion polarisation in this Graph based on the equation:

        ..math::

            \pi(k) = \frac{1}{|K|(|K| - 1)}\sum_{i \neq j}^{i \in K, j \in K}(d_{ij} - y)^{2}

        where :math:`K` is the set of agents within this Graph, :math:`d_{ij}` is the distance between
        the opinions of agents :math:`i` and :math:`j`, and :math:`y` is the mean opinion distance among
        all agents in this Graph.

        :return: The measure of opinion radicalisation in this social hierarchy.
        """
        K: int = self.node_count
        opinion_distances: dict[str, float] = {}

        for i in self.graph.nodes():
            for j in self.graph.nodes():
                if i.agent.id == j.agent.id:
                    continue
                opinion_distance: float = abs(i.agent.opinion - j.agent.opinion)
                opinion_distances[f"{i.index},{j.index}"] = opinion_distance

        y: float = sum(opinion_distances.values()) / len(opinion_distances.values())

        summation: float = 0.0
        for distance in opinion_distances.values():
            square_distance: float = (distance - y) ** 2
            summation += square_distance

        radicalisation_measure: float = summation / (K * (K - 1))
        return radicalisation_measure

    def __in__(self, iterable: Iterable[Graph]) -> bool:
        """
        Determine if the Graph is contained within the Iterable of Graphs.

        :param iterable: The iterable of Graph objects in which membership is being determined.
        :return: A boolean indicating if this Graph is contained within the iterable.
        """
        for graph in iterable:
            if self.name == graph.name:
                return True
        return False

    @override
    def __str__(self) -> str:
        """
        An override of the Graph string representation when calling print().

        :return: A string outlining the name and graph properties of the specific social hierarchy.
        """
        return f"Graph representing the {self.name} social hierarchy with {self.node_count} nodes and {self.edge_count} edges"


class GraphSet:
    """
    A class that will collect all of the different social hierarchy graphs in the same structure
    and provide utilities using this collection.
    """

    def __init__(self, model: Any, graphs: list[Graph] | None = None) -> None:
        """
        :param model: The parent ABModel object that this GraphSet is being attached to.
        :param graphs: An optional iterable containing already created Graph objects.
        """
        self.parent_model: Any = model
        self.graphs: list[Graph] = []
        if graphs:
            self.graphs = graphs

    def save_graphset(self, directory_path: str) -> None:
        """
        Save all of the graphs contained within this graphset into a compressed subdirectory representing
        the saved GraphSet.

        :param directory_path: The path to the directory where the graphset subdirectory should be created.
        """
        # Assume that the passed directory path is to the base save path, not directly to the graphset subdirectory
        subdirectory_path: str = f"{directory_path}/_graphset"

        if os.path.isdir(subdirectory_path):
            # Remove the existing directory to allow for a new overwrite
            rmtree(subdirectory_path)

        # Create the _graphset subdirectory
        os.mkdir(subdirectory_path)

        graph_save_paths: list[str] = []

        for graph in self.graphs:
            # Save path for the specific hierarchy graph
            graph_save_path: str = f"{subdirectory_path}/graph_{graph.name}.graphml"
            graph.save_graph(graph_save_path)
            graph_save_paths.append(graph_save_path)

        zip_path: str = f"{subdirectory_path}.zip"

        if os.path.exists(zip_path):
            # Remove the existing zip file to allow for a new overwrite
            os.remove(zip_path)

        # Compress the subdirectory to minimise storage, and encapsulate all graphs into a single object
        with zipfile.ZipFile(
            zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as subdir_zip:
            for graph_path in graph_save_paths:
                subdir_zip.write(graph_path, arcname=f"{os.path.basename(graph_path)}")

        # Remove the uncompressed subdirectory if compression was successful
        if os.path.exists(zip_path):
            rmtree(subdirectory_path)

        return None

    def load_graphset(
        self, load_path: str, rw_params: dict[str, tuple[float, float]]
    ) -> None:
        """
        Loads a GraphSet that has been saved following the same process as in the save_graphset() function.

        :param load_path: The path to the model's overall save directory.
        :param rw_params: A <name : rw_params> dictionary containing the relevant external information for each graph.
        """
        zip_load_path: str = f"{load_path}/_graphset.zip"

        if not os.path.exists(zip_load_path):
            raise FileNotFoundError(
                f"No saved GraphSet was found at the path: {zip_load_path}"
            )

        # The path to the uncompressed subdirectory
        subdirectory_path: str = f"{load_path}/_graphset"

        # Remove any existing subdirectory with the same name to replace it with the newly loaded one
        if os.path.isdir(subdirectory_path):
            rmtree(subdirectory_path)

        # Create the uncompressed subdirectory
        os.mkdir(subdirectory_path)

        # Extract all the graphml files to the uncompressed subdirectory
        with zipfile.ZipFile(
            zip_load_path, mode="r", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as subdir_zip:
            subdir_zip.extractall(path=subdirectory_path)

        # Load each graphml file and add it to the GraphSet
        for graphml_name in os.listdir(subdirectory_path):
            print(graphml_name)
            graphml_path: str = f"{subdirectory_path}/{graphml_name}"

            # Extracts the name of the hierarchy without the "graph_" prefix or file type suffix
            graph_name: str = graphml_path[6:-8]

            graph_object: Graph = Graph("", (0.0, 0.0))
            graph_object.load_graph(graphml_path, graph_name, rw_params[graph_name])

            # Add the Graph object to the GraphSet
            self.add_graph(graph_object)

        return None

    def add_graph(self, graph: Graph) -> None:
        """
        A setter function to add a new Graph object to the GraphSet.

        :param graph: The Graph object to add to the GraphSet.
        """
        self.graphs.append(graph)
        return None

    def graph_at_index(self, graph_index: int) -> Graph | None:
        """
        A getter function to return a Graph object stored at the given index in the GraphSet.

        :param graph_index: The index of the Graph to return.
        :return: The Graph object to return
        """
        try:
            return self.graphs[graph_index]
        except IndexError:
            print(
                f"Index {graph_index} is out of bounds for the GraphSet. Only {len(self.graphs)} social hierarchies have been created."
            )
            return None

    def get_hierarchy(self, hierarchy: str) -> Graph | None:
        """
        A getter function to return a Graph object with the given hierarchy name.

        :param hierarchy: The name of the social hierarchy represented by the Graph to return.
        :return: The Graph object of the specified hierarchy, or None if no matching hierarchy was found.
        """
        for graph in self.graphs:
            if graph.name == hierarchy:
                return graph

        print(f"No graph representing the social hierarchy '{hierarchy}' was found...")
        return None

    def get_index(self, hierarchy: str) -> int:
        """
        A getter function that returns the index of a given hierarchy within the GraphSet.

        :param hierarchy: The name of the hierarchy that is being searched for.
        :return: The index of the hierarchy within the GraphSet.
        """
        for idx, graph in enumerate(self.graphs):
            if graph.name == hierarchy:
                return idx

        raise KeyError(
            f"The social hierarchy '{hierarchy}' does not exist in the GraphSet -- cannot return an index."
        )

    def list_hierarchies(self, print_out: bool = False) -> list[str]:
        """
        A utility function that iterates over the GraphSet and prints out the names of all the social hierarchies that are present.

        :param print_out: A boolean which flags if the listed hierarchies should be printed to the terminal.
        :return: A list of the names of all social hierarchies present in the GraphSet.
        """
        social_hierarchies: list[str] = []
        for graph in self.graphs:
            social_hierarchies.append(graph.name)

        if print_out:
            print(
                f"\nSocial hierarchies present in the GraphSet:\n\t{social_hierarchies}\n\n"
            )

        return social_hierarchies

    def calculate_polarisation(self, hierarchy: str) -> float:
        """
        A wrapper that calls a specific hierarchy graph's calculate_polarisation function and returns its value.

        :param hierarchy: The name of the hierarchy for which polarisation is being calculated.
        :return: The hierarchy polarisation value.
        """
        hierarchy_graph: Any = self.get_hierarchy(hierarchy)
        return hierarchy_graph.calculate_polarisation()

    def agent_opinion_threshold(
        self, agent: Agent, threshold: float = 0.9
    ) -> Iterable[str]:
        """
        A utility function that iterates over the GraphSet and records for which social hierarchies a specific Agent's weighting
        of those hierarchies is above a certain threshold value.
        :param agent: the Agent for which to check the AgentSet for.
        :param threshold: the absolute threshold value over which the Agent's opinion is considered significant.

        :return: An iterable containing the names of the hierarchies for which the agent's weighting is above the threshold.
        """
        significant_hierarchies: Iterable[str] = []
        for hierarchy in self.graphs:
            if hierarchy.agent_in_graph(agent):
                social_weighting: float = agent.social_weightings[hierarchy.name]
                if abs(social_weighting) > threshold:
                    significant_hierarchies.append(hierarchy.name)
        return significant_hierarchies

    def __in__(self, graph: Graph) -> bool:
        """
        A method defining how a GraphSet checks for Graph membership.

        :param graph: The Graph object whose membership is being checked for.
        :return: A boolean indicating if the Graph object is contained in self.graphs.
        """
        return graph in self.graphs

    def __contains__(self, graph: Graph) -> bool:
        """
        A secondary method defining how a GraphSet checks for Graph membership.

        :param graph: The Graph object whose membership is being checked for.
        :return: A boolean indicating if the Graph object is contained in self.graphs.
        """
        return self.graphs.__contains__(graph)

    def __len__(self) -> int:
        """
        A method defining how a GraphSet checks its length.

        :return: An integer specifying the number of Graph objects contained within the GraphSet.
        """
        return len(self.graphs)

    def __iter__(self) -> Iterator[Graph]:
        """
        A method defining how a GraphSet iterates over the Graphs contained within it.

        :return: An iterator object that iterates over the Graphs in the GraphSet.
        """
        return self.graphs.__iter__()

    @override
    def __str__(self) -> str:
        """
        An override of what calling `print()` on this object will output.

        :return: A string listing the names of the hierarchies which are contained in the GraphSet.
        """
        return f"GraphSet containing the graphs of the following social hierarchies:\n\n{self.list_hierarchies()}"
