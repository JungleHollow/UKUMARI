from __future__ import annotations

from copy import deepcopy
from random import choices, randint
from typing import Any

import numpy as np
from rustworkx.rustworkx import NoEdgeBetweenNodes

from .agents import Agent, AgentSet
from .graphs import Graph, GraphEdge, GraphSet
from .logging import GATOHLogger
from .visualisation import ABVisualiser


class ABModel:
    """
    An agent-based model class that is capable of handling multiple layers that affect agent behaviour.
    """

    def __init__(
        self,
        hierarchy_names: list[str],
        hierarchy_rw_distributions: list[tuple[float, float]],
        iterations: int = 100,
        silencing_threshold: float = 0.8,
        negation_threshold: float = 0.99,
        radicalisation_threshold: float = 0.9,
        data_file: str = "",
        model_id: str = "",
    ) -> None:
        """
        :param hierarchy_names: A list of strings representing the names of all social hierachies that will exist in the model.
        :param hierarchy_rw_distributions: A list of (mean, variance) tuples defining the parameters of normal distributions used in random walks for their corresponding hierarchies.
        :param iterations: The number of iterations that the model will run for.
        :param silencing_threshold: A threshold that, when surpassed by Agents, will cause them to cease expressing their opinions in a given hierarchy.
        :param negation_threshold: A threshold that, when surpassed by Agents, will cause their opinion to become its additive inverse.
        :param radicalisation_threshold: A threshold that determined how strong of an absolute opinion an Agent must hold before they begin to consider becoming radicalised.
        :param data_file: The path to which the logger's data should be saved to after iterations are run.
        :param model_id: An optional field to give the created model object a referencable ID.
        """
        self.hierarchy_information: dict[str, tuple[float, float]] = {}
        for idx, hierarchy in enumerate(hierarchy_names):
            self.hierarchy_information[hierarchy] = hierarchy_rw_distributions[idx]

        self.agents: AgentSet = AgentSet(self)
        self.graphs: GraphSet = GraphSet(self)

        # A model-handled 'base' Graph that keeps track of all relationships across the social hierarchies
        # (Used to greatly simplify network-level graph calculations)
        self.base_graph: Graph = Graph("base", (0.0, 0.0))

        self.logger: GATOHLogger = GATOHLogger(self, iterations, hierarchy_names)
        self.visualiser: ABVisualiser = ABVisualiser(self)
        self.current_iteration: int = 0
        self.max_iterations: int = iterations

        self.silencing_threshold: float = silencing_threshold
        self.negation_threshold: float = negation_threshold
        self.radicalisation_threshold: float = radicalisation_threshold

        self.data_file: str = data_file
        self.model_id: str = model_id

    def add_graph(self, graph: Graph) -> GraphSet:
        """
        Add a new Graph to the Model's GraphSet. It is assumed that this is a generated Graph object which already has
        a name and rw_params assigned to it.

        :param graph: The Graph object to be added to the Model's GraphSet.
        :return: The model's newly updated GraphSet.
        """
        self.graphs.add_graph(graph)

        # Also add new edges to the model's base graph
        self.add_base_graph_edges(graph)
        return self.graphs

    def add_graphs(
        self, graphs: list[Any], names: list[str], rw_params: list[tuple[float, float]]
    ) -> GraphSet:
        """
        Add new Graphs to the Model's GraphSet.

        :param graphs: A list of Graph objects or filepaths to stored GraphML objects.
        :param names: A list of the corresponding social hierarchy names to give to the Graphs.
        :param rw_params: The (mean, variance) to assign to the hierarchy when determining normal distributions for random walk dynamic relationships.
        :return: The model's newly updated GraphSet.
        """
        if type(graphs[0]) is Graph:
            for graph in graphs:
                self.graphs.add_graph(graph)
        else:
            for idx, graph in enumerate(graphs):
                new_graph: Graph = Graph(names[idx], rw_params[idx])
                new_graph.load_graph(graph, names[idx])
                self.graphs.add_graph(new_graph)
        self.update_base_graph()
        return self.graphs

    def generate_graphs(
        self,
        hierarchies: list[str],
        agents: list[Any] | AgentSet,
        method: str = "small-world",
        agent_subsetting: bool = False,
        rw_params: list[tuple[float, float]] | None = None,
    ) -> None:
        """
        Randomly generates graphs for the given social hierarchy names using the specified method.
        Hierarchies will only contain the agents whose names are passed to the function.

        :param hierarchies: A list containing the names of the social hierarchy graphs to be created.
        :param agents: A list of Agent objects which determines who is included in the hierarchies.
        :param method: The social network graph generation method to use. Options include: 'small-world', 'scale-free', 'random', 'blockmodel'. Defaults to 'small-world'.
        :param agent_subsetting: A boolean indicating if the agents should be sampled into random subsets when generating each graph.
        :param rw_params: A list of (mean, variance) tuples containing the random-walk distributions for each of the generated graphs.
        """
        agent_array: np.ndarray = np.array(agents)
        agent_sample: list[Agent] = []

        for idx, hierarchy in enumerate(hierarchies):
            if agent_subsetting:
                random_k: int = randint(len(agents) // 4, len(agents))
                if type(agents) is list:
                    agent_sample = list(
                        np.random.choice(agent_array, size=random_k, replace=False)
                    )
                elif type(agents) is AgentSet:
                    agent_sample = agents.sample(random_k)

            hierarchy_rw_param: tuple[float, float] = (0.0, 0.1)
            if rw_params:
                hierarchy_rw_param = rw_params[idx]

            hierarchy_graph: Graph = Graph(hierarchy, hierarchy_rw_param)
            hierarchy_graph = hierarchy_graph.generate_graph(
                agent_sample, method=method
            )

            self.add_graph(hierarchy_graph)
        return None

    def add_agent(self, agent: Agent) -> int:
        """
        Add a single new Agent to the model's AgentSet, returning its index within the AgentSet.

        :param agent: The Agent object to add to the AgentSet.
        :return: The index of the newly added Agent in the AgentSet.
        """
        # Add the Agent object to the model-handled 'base' graph
        self.base_graph.add_nodes([agent])
        return self.agents.add(agent)

    def add_agents(self, agents: list[Agent]) -> AgentSet:
        """
        Add new Agents to the Model's AgentSet.

        :param agents: A list of Agent objects to be added to the AgentSet.
        :return: The model's newly updated AgentSet.
        """
        for agent in agents:
            self.agents.add(agent)

        # Add all the new Agent objects to the model-handled 'base' graph
        self.base_graph.add_nodes(agents)
        return self.agents

    def generate_agents(
        self,
        id_base: str,
        personality_probs: dict[str, float],
        distribution: str = "gaussian",
        parameters: dict | None = None,
        number: int = 100,
    ) -> None:
        """
        Randomly generates a number of Agent objects.

        :param id_base: a 4-character alphabetic string that serves as the base of the XXXXnnnn id for each Agent.
        :param personality_probs: A dictionary of <personality : probability> specifying the probability of an Agent having any given personality.
        :param distribution: The distribution from which any random values will be drawn.
        :param parameters: Any explicit parameters that the distribution should use when being created.
        :param number: Number of agents to be randomly created.
        """
        # Convert to separate lists for use in random.choices()
        personalities: list[str] = list(personality_probs.keys())
        probabilities: list[float] = list(personality_probs.values())

        # Extract the hierarchy names from the information dictionary
        hierarchies: list[str] = list(self.hierarchy_information.keys())

        for i in range(number):
            new_agent: Agent = Agent()
            agent_id: str = f"{id_base}{i:04}"
            agent_index: int = self.add_agent(new_agent)
            agent_personality: str = choices(personalities, weights=probabilities, k=1)[
                0
            ]
            new_agent.generate_agent(
                agent_id,
                agent_index,
                hierarchies,
                distribution=distribution,
                personality=agent_personality,
                parameters=parameters,
            )
        return None

    def iterate(self) -> None:
        """
        Handles the main model iteration loop
        """
        while self.current_iteration < self.max_iterations:
            # Initialise the logger state for the current iteration
            if self.current_iteration == 0:
                self.logger.new_iteration(init=True)
            else:
                self.logger.new_iteration()

            # First each agent looks at its neighbours to see how their opinion will evolve this iterations
            for agent in self.agents:
                agent.previous_opinion = agent.opinion
                for hierarchy in self.graphs:
                    # Update the previous opinion across all hierarchies
                    hierarchy.agent_previous_opinion(agent)

                collective_changes: list[float] = []
                for hierarchy in self.graphs:
                    collective_changes.append(hierarchy.neighbour_influences(agent))
                total_change: float = sum(collective_changes)

                if (agent.opinion + total_change < -1.0) or (
                    agent.opinion + total_change > 1.0
                ):
                    # Constrain the agent opinion to [-1, 1]
                    continue
                else:
                    agent.opinion += total_change
                    for hierarchy in self.graphs:
                        # Update the current opinion across all hierarchies
                        hierarchy.agent_opinion_change(agent, total_change)

                # After the opinion change, determine if the agent has become radicalised
                was_radicalised: bool = agent.radicalisation(
                    collective_changes,
                    list(self.hierarchy_information.keys()),
                    self.radicalisation_threshold,
                )

                for hierarchy in self.graphs:
                    # Update the radicalisation status of the agent across all hierarchies
                    hierarchy.agent_radicalisation_change(agent, was_radicalised)

                # Update the radicalisation count in the logger as needed
                self.logger.variables.increment_radicalised(was_radicalised)
            self.step()
            self.update()

            self.logger_iteration()  # Handle the logger's iteration() calculations and call its method

            # Get this iteration's print string (will be formatted appropriately based on the print interval)
            iteration_print_string: str = self.logger.iteration_print()
            print(iteration_print_string)

            self.current_iteration += 1
        # Call the logger's save_data function which handles data persistence appropriately
        data_saved: bool = self.logger.save_data(self.data_file)
        if data_saved:
            print(
                f"\n\nGATOH logger data was successfully written to the file at path: {self.data_file}\n\n"
            )
        return None

    def step(self) -> None:
        """
        Steps the model forward one iteration. This does not handle agent opinion changes,
        but rather dynamic agent relationships and hierarchy weightings.
        """
        for graph in self.graphs:
            graph.step()
        for agent in self.agents:
            agent.step(self.hierarchy_information)
        return None

    def update(self) -> None:
        """
        Updates the agents' internal states to match the model step. This mainly handles the construction of agents'
        perceived opinion climates within their hierarchies, and the simulation of opinion silencing behaviours depending
        on these climates.
        """
        for agent in self.agents:
            silenced: dict[str, bool] = {}
            was_silenced: bool = False
            negation: bool = False
            for graph in self.graphs:
                est_opinion_climate: float = graph.estimate_opinion_climate(agent)
                is_silenced: tuple[bool, float] = agent.opinion_silencing(
                    graph.name, est_opinion_climate
                )
                silenced[graph.name] = is_silenced[0]

                if is_silenced[0]:
                    was_silenced = True

                if not negation:
                    negation = agent.opinion_negation(
                        graph.name, is_silenced[1], self.negation_threshold
                    )

            # Update the logger variables as needed
            self.logger.variables.increment_silenced(was_silenced)
            self.logger.variables.increment_negated(negation)

            # Update the Agent object
            agent.update(silenced, negation)
        return None

    def logger_iteration(self) -> None:
        """
        Calculate any relevant aggregate statistics and then pass these to the logger's iteration() function to be stored.

        Statistics calculated currently:
            1. Aggregate network opinion
            2. Network radicalisation log odds
            3. Layer navigability for each hierarchy
            4. Layer interdependence for each hierarchy
        """
        aggregate_opinion: float = self.calculate_aggregate_opinion()
        radicalisation_logodds: float = self.calculate_radicalisation_logodds()
        layer_interdependences: dict[str, float] = {}
        for hierarchy in self.hierarchy_information.keys():
            layer_interdependences[hierarchy] = self.calculate_interdependence(
                self.graphs.get_index(hierarchy)
            )

        layers_polarisation: dict[str, float] = self.calculate_layers_polarisation()

        self.logger.iteration(
            aggregate_opinion,
            radicalisation_logodds,
            layer_interdependences,
            layers_polarisation,
        )
        return None

    def calculate_aggregate_opinion(self) -> float:
        """
        Calculates the aggregate network opinion by iterating over each Agent in the model.

        :return: The aggregate network opinion value.
        """
        all_opinions: list[float] = []
        for agent in self.agents:
            all_opinions.append(agent.opinion)

        opinion_sum: float = sum(all_opinions)
        average_opinion: float = opinion_sum / len(all_opinions)

        return average_opinion

    def calculate_radicalisation_logodds(self) -> float:
        """
        Calculates the log odds of an Agent being radicalised within the model.

        :return: The log odds of agent radicalisation.
        """
        radicalised_count: int = 0
        for agent in self.agents:
            if agent.radicalised:
                radicalised_count += 1

        radicalisation_p: float = radicalised_count / len(self.agents)
        if 1.0 - radicalisation_p != 0.0:
            log_odds: float = np.log1p(radicalisation_p / (1.0 - radicalisation_p))
            return log_odds
        return 0.0

    def calculate_layers_polarisation(self) -> dict[str, float]:
        r"""
        Calculate the polarisation of the opinion climate within each hierarchy by calling each graph's calculate_polarisation() method.

        :return: A <hierarchy : value> dictionary containing the polarisation value for each hierarchy.
        """
        layers_polarisation: dict[str, float] = {}

        for hierarchy in self.hierarchy_information.keys():
            layers_polarisation[hierarchy] = self.graphs.calculate_polarisation(
                hierarchy
            )

        return layers_polarisation

    def calculate_navigability(
        self, from_node: tuple[int, int], to_node: tuple[int, int]
    ) -> float:
        r"""
        Calculate the difficulty of navigating from an arbitrary node :math:`s` in some layer :math:`a` to another arbitrary
        node :math:`t` in some layer :math:`b`, where :math:`a` and :math:`b` may or may not be the same layer.

        The formulae for the navigatability are defined by:

        .. math::

            S(s \rightarrow t) = -\log_{2}\sum_{\{p(s,t)\}} P[p(s,t)]

            P[p(s,t)] = \frac{1}{k_{s}} \prod_{j \in p(s,t)} \frac{1}{k_{j} - 1}

        :param from_node: A tuple containing (agent_index, graph_index) for the starting node.
        :param to_node: A tuple containing (agent_index, graph_index) for the end node.
        :return: The navigability value for the specified path.
        """
        # TODO: Implement this function
        raise NotImplementedError(
            "Navigability measure calculation is not yet implemented..."
        )
        return 0.0

    def calculate_interdependence(self, layer: int) -> float:
        r"""
        Calculate the layer interdependence; a measure of how much impact a specific layer has in the overall
        social network.

        The general formula for layer interdependence is defined as:

        .. math::

            \lambda^{a} = \frac{\sum_{i}\sum_{j \neq j\i}\Psi^{a}_{ij}}{\sum_{i}\sum_{j \neq i}\Psi_{ij}}

        where :math:`\Psi^{a}_{ij}` describes the number of shortest paths between nodes :math:`i` and :math:`j`
        using two or more layers, where at least one of the layers passed through is :math:`a`.

        For the case of multilayer social contagion modeling, it has been defined here as:

        .. math::

            \lambda^{a} = \frac{\sum_{i}\sum_{j \neq i}|OC'_{i}(j)^{a}|}{\sum_{i}\sum_{j \neq i}|OC'_{i}(j)|}

        where :math:`|OC'_{i}(j)^{a}|` is Agent :math:`j`'s opinion climate value as perceived by Agent :math:`i`
        in the social hierarchy layer :math:`a`. Although, the absolute of this value should be taken, as this
        is representative of the real `strength' of a layer.

        :param layer: The index of the layer of interest.
        :return: The layer interdependence measure for the layer of interest.
        """
        # Update the base graph's edge weights before performing any calculations (possibility for future features requiring this)
        self.update_base_graph()

        layer_of_interest: str = self.graphs.graphs[layer].name
        observed_opinions_all: dict[str, dict[str, dict[str, float]]] = {}

        for hierarchy in self.graphs:
            observed_opinions_layer: dict[str, dict] = {}
            for agent_i in hierarchy.graph.nodes():
                agent_i_oc: dict[str, float] = hierarchy.estimate_neighbour_opinions(
                    agent_i.agent
                )
                observed_opinions_layer[agent_i.agent.id] = agent_i_oc
            observed_opinions_all[hierarchy.name] = observed_opinions_layer

        interdep_numerator: float = 0.0
        # Get the sum of all the estimated opinion values, only for the layer of interest (a)
        oc_a: dict[str, dict] = observed_opinions_all[layer_of_interest]
        for agent_a_i, oc_a_i in oc_a.items():
            for oc_val in oc_a_i.values():
                interdep_numerator += abs(oc_val)

        interdep_denominator: float = 0.0
        # Get the sum of all the estimated opinion values for all layers (k)
        for layer_k, oc_k in observed_opinions_all.items():
            for agent_k_i, oc_k_i in oc_k.items():
                for oc_val in oc_k_i.values():
                    interdep_denominator += abs(oc_val)

        # Calculate the interdependence value for the layer
        layer_interdependence: float = interdep_numerator / interdep_denominator
        return layer_interdependence

    def get_base_indices_from_edge(
        self, hierarchy_graph: Graph, edge: GraphEdge
    ) -> tuple[int, int]:
        """
        A helper function for the base graph that takes in a GraphEdge object from a hierarchy graph and transforms
        the node indices from hierarchy graph indices to the respective index of the Agent objects in the model's
        AgentSet.

        :param hierarchy_graph: The corresponding hierarchy Graph object that the GraphEdge belongs in.
        :param edge: A GraphEdge object from one of the model's hierarchy graphs in the GraphSet.
        :return: The index in the AgentSet of the parent and child nodes involved in the hierarchy graph's relationship.
        """
        # Actually GraphNode objects, but must be declared as "Any" for cases where a non-existent node index is passed to the function...
        from_node: Any = hierarchy_graph.get_node(edge.from_node)
        to_node: Any = hierarchy_graph.get_node(edge.to_node)

        from_index_base: int = self.agents.get_index(from_node.agent)
        to_index_base: int = self.agents.get_index(to_node.agent)

        return from_index_base, to_index_base

    def add_base_graph_edges(self, graph: Graph) -> None:
        """
        A function that takes a Graph object and adds all of its weighted edges to the model's base graph.

        :param graph: The new Graph object that is being added to self.graphs.
        """
        new_edges: dict = {"from_node": [], "to_node": [], "weighting": [], "name": []}

        for idx, edge in graph.graph.edge_index_map().items():
            graph_edge: GraphEdge = deepcopy(edge[2])

            # Get the index of the Agent objects within the model's AgentSet (not the graph's node set)
            base_from_idx, base_to_idx = self.get_base_indices_from_edge(
                graph, graph_edge
            )

            new_edges["name"].append(graph_edge.hierarchy)
            new_edges["from_node"].append(base_from_idx)
            new_edges["to_node"].append(base_to_idx)
            new_edges["weighting"].append(graph_edge.weighting)

        self.base_graph.add_edges(deepcopy(new_edges))

        # Manual garbage collection
        del new_edges

        return None

    def update_base_graph(self) -> None:
        """
        Iterates over all relationships in the base graph and checks the respective relationship within the relevant hierarchy,
        updating the relationship weight if needed.
        """
        for hierarchy in self.graphs:
            for idx, edge in hierarchy.graph.edge_index_map().items():
                graph_edge: GraphEdge = deepcopy(edge[2])

                # Get the index of the Agent objects within the model's AgentSet (not the graph's node set)
                base_from_idx, base_to_idx = self.get_base_indices_from_edge(
                    hierarchy, graph_edge
                )

                # Update the weigting in base graph if an edge exists and the weighting is different from the hierarchy's
                try:
                    base_edge: GraphEdge = self.base_graph.graph.get_edge_data(
                        base_from_idx, base_to_idx
                    )
                    if graph_edge.weighting != base_edge.weighting:
                        self.base_graph.change_weights(
                            base_from_idx, base_to_idx, graph_edge.weighting
                        )
                # If the edge does not exist in the base graph, create it and add it to the base graph
                except NoEdgeBetweenNodes:
                    new_edge: dict = {
                        "from_node": [base_from_idx],
                        "to_node": [base_to_idx],
                        "weighting": [graph_edge.weighting],
                        "name": [hierarchy.name],
                    }
                    self.base_graph.add_edges(deepcopy(new_edge))

                    # Manual garbage collection
                    del new_edge
        return None
