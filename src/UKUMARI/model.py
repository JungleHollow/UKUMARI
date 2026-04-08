from __future__ import annotations

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
        iterations: int = 100
    ) -> None:
        """
        :param iterations: The number of iterations that the model will run for
        :param xlims: A (lower, upper) tuple representing the limits of the model's x-axis
        :param ylims: A (lower, upper) tuple representing the limits of the model's y-axis
        """
        self.graphs: GraphSet = GraphSet(self)
        self.agents: AgentSet = AgentSet(self)

        self.logger: UKUMARILogger = UKUMARILogger(self)
        self.visualiser: ABVisualiser = ABVisualiser(self)
        self.current_iteration: int = 0
        self.max_iterations: int = iterations

    def add_graphs(self, graphs: list[Any], names: list[str], rw_params: list[tuple[float, float]]) -> None:
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
        self, agents: list[Any], names: list[str], method: str = "small-world"
    ) -> None:
        """
        Randomly generates graphs for the given social hierarchy names using the specified method.
        Hierarchies will only contain the agents whose names are passed to the function.

        :param agents: A list of agent IDs or Agent objects which determines who is included in the hierarchies
        :param names: The names of the social hierarchies for which graphs are being generated
        :param method: The social network graph generation method to use. Options include: 'small-world', 'scale-free', 'full'. Defaults to 'small-world'
        """
        # TODO: Implement this function
        raise NotImplementedError(
            "Random graph generation function in ABModel not implemented yet."
        )

    def add_agents(self, agents: list[Agent]) -> None:
        """
        Add new Agents to the Model's AgentSet.

        :param agents: A list of Agent objects to be added to the AgentSet
        """
        for agent in agents:
            self.agents.add(agent)

    def generate_agents(self, attributes: dict, number: int = 100) -> None:
        """
        Randomly generates a number of Agent objects using the given attribute dictionary.
        The dictionary items should be singular explciit values to assign for all agents, or tuples representing:
            (mean, standard deviation, distribution to use) for random generation

        :param attributes: A dictionary containing (attribute: tuple) pairs for Agent attribute setting
        :param number: Number of agents to be randomly created.
        """
        # TODO: Reimplement this function (will be done similar to generate_graphs, where main function is in agents.py and this is a wrapper...)
        for i in range(number):
            new_agent: Agent = Agent()
            for key, value in attributes.items():
                if len(value) == 1:
                    new_agent.add_attribute(key, value=value)
                else:
                    new_agent.add_attribute(
                        key, mean=value[0], sdev=value[1], distribution=value[2]
                    )
            self.agents.add(new_agent)

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

                if (agent.opinion + total_change < -1.0) or (agent.opinion + total_change > 1.0):
                    # Constrain the agent opinion to [-1, 1]
                    continue
                else:
                    agent.opinion += total_change
            self.step()
            self.update()
            self.logger.iteration_print(
                self.current_iteration
            )  # Does nothing if not at the print interval
            self.current_iteration += 1

    def step(self) -> None:
        """
        Steps the model forward one iteration. This does not handle agent opinion changes,
        but rather dynamic agent movement and relationship changes.
        """
        # TODO: Implement this function
        pass

    def update(self) -> None:
        """
        Updates the agents' internal states to match the model step.
        """
        # TODO: Implement this function
        pass
