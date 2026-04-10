from __future__ import annotations

import contextlib
import warnings
from collections.abc import Iterable
from random import Random
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

    def load_graph(self, path: str, name: str) -> None:
        """
        Loads a Graph object stored in the GraphML format from the given path.
        The social hierarchy name must be explicitly passed with this call.

        :param path: Path to a stored graph file.
        """
        graph: list[Any] = rx.read_graphml(path)
        self.graph = graph[0]
        self.node_count = len(self.graph.nodes())
        self.edge_count = len(self.graph.edges())
        self.name = name

    def save_graph(self, path: str) -> None:
        """
        Saves the existing Graph object in the GraphML format to the given path.

        :param path: Path to which the Graph will be saved.
        """
        rx.write_graphml(self.graph, path)

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
            self.graph[index].index = index
        self.update_edge_indices()
        self.node_count = len(self.graph.nodes())

    def add_nodes(self, agents: Iterable[Agent]) -> None:
        """
        Creates appropriate GraphNodes from the given Agents, and then adds these to the graph.

        :param agents: The Agent objects that will be converted to GraphNodes and added to the graph
        """
        nodes = []
        for agent in agents:
            agent_node = GraphNode(agent)
            nodes.append(agent_node)

        self.graph.add_nodes_from(nodes)
        self.update_node_indices()

    def update_edge_indices(self) -> None:
        """
        Iterates over all the existing edges in the graph and updates their stored indices to reflect the current graph state.
        Will also update the graph edge_count attribute
        """
        for index, data in self.graph.edge_index_map().items():
            data[2].index = index
        self.edge_count = len(self.graph.edges())

    def add_edges(self, edges: dict) -> None:
        """
        Creates appropriate GraphEdges from the given dictionary and then adds these to the graph.

        :param edges: A dictionary of key-list pairs where each key corresponds to (from_node, to_node, [optional] weighting)
        """
        graph_edges = []
        from_nodes = edges["from_node"]
        to_nodes = edges["to_node"]
        weightings = None
        if "weighting" in edges.keys():
            weightings = edges["weighting"]

        if weightings:
            for i in range(len(from_nodes)):
                edge = GraphEdge(self.name, from_nodes[i], to_nodes[i], weightings[i])
                graph_edges.append((from_nodes[i], to_nodes[i], edge))
        else:
            for i in range(len(from_nodes)):
                edge = GraphEdge(self.name, from_nodes[i], to_nodes[i])
                graph_edges.append((from_nodes[i], to_nodes[i], edge))

        self.graph.add_edges_from(graph_edges)
        self.update_edge_indices()

    def generate_graph(self, agents: list[Agent], method: str = "small-world") -> None:
        """
        Randomly generate edges between existing Graph nodes and add them to the graph.

        :param agents: The subset of Agents in the base model that are being used as the nodes for this graph.
        :param method: The random generation method to use. Possible choices include: 'small-world', 'scale-free', 'random', 'blockmodel'; Defaults to 'small-world'.
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
                k: int = np.ceil(
                    np.log(n)
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
                    (sbm_n_blocks, sbm_n_blocks), dtype=np.float16
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
        for node in generated_graph.nodes():
            graph_node: GraphNode = GraphNode(agents[node])
            graph_node.index = node
            graph_nodes.append(graph_node)
        for idx, graph_node in enumerate(graph_nodes):
            generated_graph[idx] = (
                graph_node  # Update all the graph nodes with the new GraphNode data objects
            )

        self.graph = generated_graph  # Store the generated graph as the object's "graph" attribute (with 0.0 weights currently)

        for edge in generated_graph.weighted_edge_list():
            generated_value = random_gen.uniform(
                -1, 1
            )  # Generate a random value in the range [-1, 1]
            self.change_weights(
                edge[0], edge[1], generated_value
            )  # Update the edge weighting in the graph

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

        :param node_1: Some Agent in the graph.
        :param node_2: Some other Agent in the graph.
        :param value: The new weight to assign.
        """
        edge_index: int | None = self.relationship_exists(node_1, node_2)
        updated_edge: list[Any] = [GraphEdge(self.name, node_1, node_2, value)]
        if edge_index is not None:
            self.graph.update_edge_by_index(edge_index, updated_edge)
        else:
            self.graph.add_edges_from(updated_edge)
        self.update_edge_indices()

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

    def agent_in_graph(self, agent: Agent) -> bool:
        """
        A simple function that checks wether an Agent exists within a Graph.

        :param agent: the Agent whose existence in the Graph is being checked for.
        :return: A boolean indicating if the Agent exists in the Graph.
        """
        for node in self.graph.nodes():
            if agent == node.agent:
                return True
        return False

    def get_agent_index(self, agent: Agent) -> int:
        """
        Searches for the node index in the Graph which corresponds to the input Agent object.

        :param agent: The Agent object whose index is being searched for.
        :return: The Agent's node index within the social hierarchy Graph.
        """
        for idx, node in enumerate(self.graph.nodes()):
            if agent == node.agent:
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
        Step the individual Graph object.
        """
        # TODO: Implement this function
        pass

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
            rw_value: float = value_rw_delta(
                edge.weighting, self.rw_params[0], self.rw_params[1]
            )
            if (edge.weighting + rw_value < -1.0) or (edge.weighting + rw_value > 1.0):
                # Constrain the relationship weightings to [-1, 1]
                continue
            else:
                edge.weighting += rw_value

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
            observed_opinion: float = direct_neighbour.agent.opinion
            observed_opinions.append(observed_opinion)

        for node in self.graph.nodes():
            if node.agent == agent or node in direct_neighbours:
                # Only look at indirect neighbours
                continue

            raw_observed_opinion: float = node.agent.opinion
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
        opinion_climate: float = summed_opinions / float(
            len(observed_opinions)
        )  # Find the average of the aggregated, relevant opinions

        return opinion_climate

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
                if i == j:
                    continue
                opinion_distance: float = i.agent.opinion - j.agent.opinion
                opinion_distances[f"{i.index},{j.index}"] = opinion_distance

        y: float = sum(opinion_distances.values()) / len(opinion_distances.values())

        summation: float = 0.0
        for distance in opinion_distances.values():
            square_distance: float = (distance - y) ** 2
            summation += square_distance

        radicalisation_measure: float = (1 / (K * (K - 1))) * summation
        return radicalisation_measure

    def __in__(self, iterable: Iterable[Graph]) -> bool:
        """
        Determine if the Graph is contained within the Iterable of Graphs.

        :param iterable: The iterable of Graph objects in which membership is being determined.
        :return: A boolean indicating if this Graph is contained within the iterable.
        """
        for graph in iterable:
            if self == graph:
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

    def __init__(self, model: Any, graphs: list[Graph] = []) -> None:
        """
        :param model: The parent ABModel object that this GraphSet is being attached to.
        :param graphs: An optional iterable containing already created Graph objects.
        """
        self.parent_model: Any = model
        self.graphs: list[Graph] = graphs

    def add_graph(self, graph: Graph) -> None:
        """
        A setter function to add a new Graph object to the GraphSet.

        :param graph: The Graph object to add to the GraphSet.
        """
        self.graphs.append(graph)

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
