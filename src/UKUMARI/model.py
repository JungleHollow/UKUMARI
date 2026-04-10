from __future__ import annotations

from random import choices
from typing import Any

from .agents import Agent, AgentSet
from .graphs import Graph, GraphSet
from .logging import UKUMARILogger
from .visualisation import ABVisualiser


class ABModel:
    """
    An agent-based model class that is capable of handling multiple layers that affect agent behaviour.
    """

    # TODO: Add functions to calculate network-level informational statistics (i.e. layer interdependence, radicalisation log odds, etc...)

    def __init__(
        self,
        hierarchy_names: list[str],
        hierarchy_rw_distributions: list[tuple[float, float]],
        iterations: int = 100,
        negation_threshold: float = 0.95,
    ) -> None:
        """
        :param hierarchy_names: A list of strings representing the names of all social hierachies that will exist in the model
        :param hierarchy_rw_distributions: A list of (mean, variance) tuples defining the parameters of normal distributions used in random walks for their corresponding hierarchies.
        :param iterations: The number of iterations that the model will run for
        :param negation_threshold: A threshold that, when surpassed by Agents, will cause their opinion to become its additive inverse.
        """
        self.hierarchy_information: dict[str, tuple[float, float]] = {}
        for idx, hierarchy in enumerate(hierarchy_names):
            self.hierarchy_information[hierarchy] = hierarchy_rw_distributions[idx]

        self.graphs: GraphSet = GraphSet(self)
        self.agents: AgentSet = AgentSet(self)

        self.logger: UKUMARILogger = UKUMARILogger(self)
        self.visualiser: ABVisualiser = ABVisualiser(self)
        self.current_iteration: int = 0
        self.max_iterations: int = iterations

        self.negation_threshold: float = negation_threshold

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

    def generate_graphs(self, agents: list[Any], method: str = "small-world") -> None:
        """
        Randomly generates graphs for the given social hierarchy names using the specified method.
        Hierarchies will only contain the agents whose names are passed to the function.

        :param agents: A list of agent IDs or Agent objects which determines who is included in the hierarchies
        :param method: The social network graph generation method to use. Options include: 'small-world', 'scale-free', 'full'. Defaults to 'small-world'
        """
        # TODO: Implement this function
        raise NotImplementedError(
            "Random graph generation function in ABModel not implemented yet."
        )

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
            # First each agent looks at its neighbours to see how their opinion will evolve this iterations
            for agent in self.agents.agents:
                agent.previous_opinion = agent.opinion
                collective_changes: list[float] = []
                for hierarchy in self.graphs.graphs:
                    collective_changes.append(hierarchy.neighbour_influences(agent))
                total_change: float = sum(collective_changes)

                if (agent.opinion + total_change < -1.0) or (
                    agent.opinion + total_change > 1.0
                ):
                    # Constrain the agent opinion to [-1, 1]
                    continue
                else:
                    agent.opinion += total_change
            self.step()
            self.update()
            self.logger.iteration()  # Store all relevant model variables and states for future analysis
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
            negation: bool = False
            for graph in self.graphs:
                est_opinion_climate: float = graph.estimate_opinion_climate(agent)
                is_silenced: tuple[bool, float] = agent.opinion_silencing(
                    graph.name, est_opinion_climate
                )
                silenced[graph.name] = is_silenced[0]
                if not negation:
                    negation = agent.opinion_negation(
                        graph.name, is_silenced[1], self.negation_threshold
                    )
            agent.update(silenced, negation)

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
