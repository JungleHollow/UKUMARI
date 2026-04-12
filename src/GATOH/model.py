from __future__ import annotations

from random import choices, randint
from typing import Any

import numpy as np

from .agents import Agent, AgentSet
from .graphs import Graph, GraphSet
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
        negation_threshold: float = 0.95,
        radicalisation_threshold: float = 0.90,
    ) -> None:
        """
        :param hierarchy_names: A list of strings representing the names of all social hierachies that will exist in the model
        :param hierarchy_rw_distributions: A list of (mean, variance) tuples defining the parameters of normal distributions used in random walks for their corresponding hierarchies.
        :param iterations: The number of iterations that the model will run for
        :param negation_threshold: A threshold that, when surpassed by Agents, will cause their opinion to become its additive inverse.
        :param radicalisation_threshold: A threshold that determined how strong of an absolute opinion an Agent must hold before they begin to consider becoming radicalised.
        """
        self.hierarchy_information: dict[str, tuple[float, float]] = {}
        for idx, hierarchy in enumerate(hierarchy_names):
            self.hierarchy_information[hierarchy] = hierarchy_rw_distributions[idx]

        self.graphs: GraphSet = GraphSet(self)
        self.agents: AgentSet = AgentSet(self)

        self.logger: GATOHLogger = GATOHLogger(self, iterations, hierarchy_names)
        self.visualiser: ABVisualiser = ABVisualiser(self)
        self.current_iteration: int = 0
        self.max_iterations: int = iterations

        self.negation_threshold: float = negation_threshold
        self.radicalisation_threshold: float = radicalisation_threshold

    def add_graph(self, graph: Graph) -> None:
        """
        Add a new Graph to the Model's GraphSet. It is assumed that this is a generated Graph object which already has
        a name and rw_params assigned to it.

        :param graph: The Graph object to be added to the Model's GraphSet.
        """
        self.graphs.add_graph(graph)

    def add_graphs(
        self, graphs: list[Any], names: list[str], rw_params: list[tuple[float, float]]
    ) -> None:
        """
        Add new Graphs to the Model's GraphSet.

        :param graphs: A list of Graph objects or filepaths to stored GraphML objects
        :param names: A list of the corresponding social hierarchy names to give to the Graphs
        :param rw_params: The (mean, variance) to assign to the hierarchy when determining normal distributions for random walk dynamic relationships
        """
        if type(graphs[0]) is Graph:
            for graph in graphs:
                self.graphs.add_graph(graph)
        else:
            for idx, graph in enumerate(graphs):
                new_graph: Graph = Graph(names[idx], rw_params[idx])
                new_graph.load_graph(graph, names[idx])
                self.graphs.add_graph(new_graph)

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
            hierarchy_graph.generate_graph(agent_sample, method=method)
            self.add_graph(hierarchy_graph)

    def add_agent(self, agent: Agent) -> int:
        """
        Add a single new Agent to the model's AgentSet, returning its index within the AgentSet.

        :param agent: The Agent object to add to the AgentSet.
        :return: The index of the newly added Agent in the AgentSet.
        """
        return self.agents.add(agent)

    def add_agents(self, agents: list[Agent]) -> None:
        """
        Add new Agents to the Model's AgentSet.

        :param agents: A list of Agent objects to be added to the AgentSet
        """
        for agent in agents:
            self.agents.add(agent)

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

                # After the opinion change, determine if the agent has become radicalised
                was_radicalised: bool = agent.radicalisation(
                    collective_changes,
                    list(self.hierarchy_information.keys()),
                    self.radicalisation_threshold,
                )

                # Update the radicalisation count in the logger as needed
                self.logger.variables.increment_radicalised(was_radicalised)
            self.step()
            self.update()

            self.logger_iteration()  # Handle the logger's iteration() calculations and call its method
            self.logger.iteration_print(
                self.current_iteration
            )  # Does nothing if not at the print interval
            self.current_iteration += 1

    def step(self) -> None:
        """
        Steps the model forward one iteration. This does not handle agent opinion changes,
        but rather dynamic agent relationships and hierarchy weightings.
        """
        for graph in self.graphs:
            graph.step()
        for agent in self.agents:
            agent.step(self.hierarchy_information)

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
        log_odds: float = np.log(radicalisation_p / (1.0 - radicalisation_p))
        return log_odds

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

        The formula for layer interdependence is defined as:

        .. math::

            \lambda^{a} = \frac{\sum_{i}\sum_{i \neq j}\Psi^{a}_{ij}}{\sum_{i}\sum_{i \neq j}\Psi_{ij}}

        where :math:`\Psi^{a}_{ij}` describes the number of shortest paths between nodes :math:`i` and :math:`j`
        using two or more layers, where at least one of the layers passed through is :math:`a`.

        :param layer: The index of the layer of interest.
        :return: The layer interdependence measure for the layer of interest.
        """
        # TODO: Implement this function
        raise NotImplementedError(
            "Layer interdependence calculation is not yet implemented..."
        )
        return 0.0
