from __future__ import annotations

import random as rd
from copy import deepcopy
from typing import Any

import numpy as np

import src.GATOH.agents as agt
import src.GATOH.graphs as gr
import src.GATOH.model as md

# The parameters set for the tester class itself
TEST_PARAMETERS: dict[str, Any] = {
    "n_agents": 40,
    "n_negative": 4,
}

# The model parameters used when creating the ABModel instances
MODEL_PARAMETERS: dict[str, Any] = {
    "iterations": 100,
    "silencing_thresh": 0.9,
    "radical_thresh": 0.98,
    "negation_thresh": 0.99,
}

# The social hierarchies that will exist in the models
HIERARCHY_NAMES: list[str] = [
    "family",
    "friends",
    "religion",
    "neighbours",
    "cultural",
]

# The random walk distribution parameters for each hierarchy
HIERARCHY_RW_DISTRIBUTIONS: list[tuple[float, float]] = [
    (0.0, 0.01),  # Family
    (0.0, 0.05),  # Friends
    (0.0, 0.15),  # Religion
    (0.0, 0.08),  # Neighbours
    (0.0, 0.2),  # Cultural
]

# The hierarchy weightings that will be given to each hierarchy (shared amongst all agents)
HIERARCHY_WEIGHTINGS: dict[str, float] = {
    "family": 0.9,
    "friends": 0.7,
    "religion": 0.5,
    "neighbours": 0.55,
    "cultural": 0.25,
}

# Defining the distributions of the characteristics for Agents that will be used in the experiment
AGENT_CHARACTERISTICS: dict[str, Any] = {
    "non_negative_opinion": (0.0, 0.8),
    "negative_opinion": (-1.0, -0.5),
    "non_influential_connectivity": 3,
    "relationship": (-0.4, 0.4),
    "influential_connectivity": 20,
    "influential_relationship": (-0.9, 0.9),
    "social_susceptibility": 0.5,
    "personality": "social",
    "personal_benefit": True,
}

# Define the save paths for each model's logged variables (must point to a .csv file)
LI_MODEL_DATAFILE: str = "./experiments/Base/li_model_variables.csv"
HI_MODEL_DATAFILE: str = "./experiments/Base/hi_model_variables.csv"


class InfluentialTester:
    """
    A test class that sets up and runs models for a low-influence (li) scenario and a high-influence (hi) scenario for
    an experimental comaprison.

    In both cases, the models are composed of 5 hierarchies each with unique random walk distributions for their dynamic
    relationships. Additionally, both models will have 40 Agents within them, and run for a total of 100 iterations.

    In the case of the li model, all 40 Agents are "low influence" Agents with somewhat weaker relationships, and with
    hierarchy graphs having significantly less density of relationships between agents.

    In the case of the hi model, 36 of the Agents are "low influence" whilst 4 are replaced with "high influence" Agents.
    These high influence Agents have somewhat stronger relationships with their neighbours, and a significantly higher
    density of relationships between them and other Agents in the hierarchies.
    """

    def __init__(self):
        # Store the class parameters within the instance
        self.n_agents: int = TEST_PARAMETERS["n_agents"]
        self.n_negative: int = TEST_PARAMETERS["n_negative"]

        self.li_agents: list[agt.Agent] = self.create_li_agents()
        self.li_graphs: list[gr.Graph] = self.create_li_graphs(
            deepcopy(HIERARCHY_NAMES),
            deepcopy(HIERARCHY_RW_DISTRIBUTIONS),
            self.li_agents,
        )

        self.hi_agents: list[agt.Agent] = self.create_hi_agents()
        self.hi_graphs: list[gr.Graph] = self.create_hi_graphs(
            deepcopy(HIERARCHY_NAMES),
            deepcopy(HIERARCHY_RW_DISTRIBUTIONS),
            self.hi_agents,
        )

        self.li_model: md.ABModel = md.ABModel(
            deepcopy(HIERARCHY_NAMES),
            deepcopy(HIERARCHY_RW_DISTRIBUTIONS),
            iterations=MODEL_PARAMETERS["iterations"],
            silencing_threshold=MODEL_PARAMETERS["silencing_thresh"],
            negation_threshold=MODEL_PARAMETERS["negation_thresh"],
            radicalisation_threshold=MODEL_PARAMETERS["radical_thresh"],
            data_file=LI_MODEL_DATAFILE,
        )

        self.hi_model: md.ABModel = md.ABModel(
            deepcopy(HIERARCHY_NAMES),
            deepcopy(HIERARCHY_RW_DISTRIBUTIONS),
            iterations=MODEL_PARAMETERS["iterations"],
            silencing_threshold=MODEL_PARAMETERS["silencing_thresh"],
            negation_threshold=MODEL_PARAMETERS["negation_thresh"],
            radicalisation_threshold=MODEL_PARAMETERS["radical_thresh"],
            data_file=HI_MODEL_DATAFILE,
        )

    def create_li_agents(self) -> list[agt.Agent]:
        """
        Creates the population of Agents to be used in the low influence scenario.

        :return: A list containing all the returned Agent objects.
        """
        created_li_agents: list[agt.Agent] = []

        nn_opinion_range: tuple[float, float] = AGENT_CHARACTERISTICS[
            "non_negative_opinion"
        ]

        n_opinion_range: tuple[float, float] = AGENT_CHARACTERISTICS["negative_opinion"]

        agent_behaviour: tuple[str, float] = (
            AGENT_CHARACTERISTICS["personality"],
            AGENT_CHARACTERISTICS["social_susceptibility"],
        )

        # Define data types but do not assign any values
        agent_id: str
        agent_opinion: float
        agent: agt.Agent

        created_count: int = 0
        while created_count < self.n_agents - self.n_negative:
            created_count += 1
            agent_id = f"NONN{created_count:04}"

            agent_opinion = rd.uniform(nn_opinion_range[0], nn_opinion_range[1])

            agent = agt.Agent(
                agent_id,
                deepcopy(HIERARCHY_WEIGHTINGS),
                agent_opinion,
                agent_behaviour,
                AGENT_CHARACTERISTICS["personal_benefit"],
            )

            created_li_agents.append(deepcopy(agent))

            # Manual garbage collection
            del agent

        created_count = 0
        while created_count < self.n_negative:
            created_count += 1
            agent_id = f"NGTV{created_count:04}"

            agent_opinion = rd.uniform(n_opinion_range[0], n_opinion_range[1])

            agent = agt.Agent(
                agent_id,
                deepcopy(HIERARCHY_WEIGHTINGS),
                agent_opinion,
                agent_behaviour,
                AGENT_CHARACTERISTICS["personal_benefit"],
            )

            created_li_agents.append(deepcopy(agent))

            # Manual garbage collection
            del agent

        return created_li_agents

    def create_hi_agents(self) -> list[agt.Agent]:
        """
        Creates the population of Agents to be used in the high influence scenario.

        :return: A list containing all the created Agent objects.
        """
        created_hi_agents: list[agt.Agent] = []

        nn_agents: int = self.n_agents - self.n_negative
        nn_opinion_range: tuple[float, float] = AGENT_CHARACTERISTICS[
            "non_negative_opinion"
        ]

        agent_behaviour: tuple[str, float] = (
            AGENT_CHARACTERISTICS["personality"],
            AGENT_CHARACTERISTICS["social_susceptibility"],
        )

        created_count: int = 0
        while created_count < nn_agents:
            created_count += 1
            nn_agent_id: str = f"NONN{created_count:04}"

            nn_agent_opinion: float = rd.uniform(
                nn_opinion_range[0], nn_opinion_range[1]
            )

            nn_agent: agt.Agent = agt.Agent(
                nn_agent_id,
                deepcopy(HIERARCHY_WEIGHTINGS),
                nn_agent_opinion,
                agent_behaviour,
                AGENT_CHARACTERISTICS["personal_benefit"],
            )
            created_hi_agents.append(deepcopy(nn_agent))

            # Manual garbage collection
            del nn_agent

        n_opinion_range: tuple[float, float] = AGENT_CHARACTERISTICS["negative_opinion"]

        created_count = 0
        while created_count < self.n_negative:
            created_count += 1
            n_agent_id: str = f"INFN{created_count:04}"

            n_agent_opinion: float = rd.uniform(n_opinion_range[0], n_opinion_range[1])

            n_agent: agt.Agent = agt.Agent(
                n_agent_id,
                deepcopy(HIERARCHY_WEIGHTINGS),
                n_agent_opinion,
                agent_behaviour,
                AGENT_CHARACTERISTICS["personal_benefit"],
            )
            created_hi_agents.append(deepcopy(n_agent))

            # Manual garbage collection
            del n_agent

        return created_hi_agents

    def create_li_graphs(
        self,
        hierarchies: list[str],
        rw_distributions: list[tuple[float, float]],
        agents: list[agt.Agent],
    ) -> list[gr.Graph]:
        """
        Creates the graphs for the low influence model.

        For the purposes of this experiment, each graph contains all Agents in the population,
        and the relationships are created following a customised scale-free methodology.

        :param hierarchies: A list of the names of the hierarchies that each graph should represent.
        :param rw_distributions: A dictionary defining the random walk distributions for each social hierarchy.
        :param agents: The population of Agents from which the graphs will be constructed.
        :return: A list containing all the created Graph objects.
        """
        created_li_graphs: list[gr.Graph] = []

        # Create a set of the agent indices to be used later in iteration
        agent_indices: set = {i for i in range(len(agents))}

        # The valid range of relationship strengths originating from noninfluential agents
        noninf_rel_range: tuple[float, float] = AGENT_CHARACTERISTICS["relationship"]

        for idx, hierarchy in enumerate(hierarchies):
            graph: gr.Graph = gr.Graph(hierarchy, rw_distributions[idx])

            # Initialise the graph nodes using the population of Agents
            graph.add_nodes(deepcopy(agents))

            new_edges: dict[str, list] = {
                "from_node": [],
                "to_node": [],
                "weighting": [],
            }

            for agent_node in graph.graph.nodes():
                # Return a filtered tuple including all indices except the one for the current node
                valid_indices: list[int] = list(agent_indices ^ {agent_node.index})

                selected_indices: list[int] = list(
                    np.random.choice(
                        valid_indices,
                        size=AGENT_CHARACTERISTICS["non_influential_connectivity"],
                        replace=False,
                    )
                )

                for selected_index in selected_indices:
                    edge_weighting: float = rd.uniform(
                        noninf_rel_range[0], noninf_rel_range[1]
                    )

                    # Flip the to_ and from_ indices to make the weighting representative of the impact that agent_node has on others.
                    new_edges["from_node"].append(selected_index)
                    new_edges["to_node"].append(agent_node.index)
                    new_edges["weighting"].append(edge_weighting)

            graph.add_edges(deepcopy(new_edges))
            created_li_graphs.append(deepcopy(graph))

            # Manual garbage collection
            del new_edges, graph

        return created_li_graphs

    def create_hi_graphs(
        self,
        hierarchies: list[str],
        rw_distributions: list[tuple[float, float]],
        agents: list[agt.Agent],
    ) -> list[gr.Graph]:
        """
        Creates the graphs for the high influence model.

        For the purposes of this experiment, each graph contains all Agents in the population,
        and the relationships are created following a customised scale-free methodology.

        :param hierarchies: A list of the names of the hierarchies that each graph should represent.
        :param rw_distributions: A dictionary defining the random walk distributions for each social hierarchy.
        :param agents: The population of Agents from which the graphs will be constructed.
        :return: A list containing all the created Graph objects.
        """
        created_hi_graphs: list[gr.Graph] = []

        # Calculate the number of agents that are not influential
        n_ni_agents: int = self.n_agents - self.n_negative

        # Create a set of the agent indices to be used later in iteration
        agent_indices: set = {i for i in range(len(agents))}

        noninf_rel_range: tuple[float, float] = AGENT_CHARACTERISTICS["relationship"]
        inf_rel_range: tuple[float, float] = AGENT_CHARACTERISTICS[
            "influential_relationship"
        ]

        for idx, hierarchy in enumerate(hierarchies):
            graph: gr.Graph = gr.Graph(hierarchy, rw_distributions[idx])

            # Initialise the graph nodes using the full Agent population
            graph.add_nodes(deepcopy(agents))

            new_edges: dict[str, list] = {
                "from_node": [],
                "to_node": [],
                "weighting": [],
            }

            for agent_node in graph.graph.nodes():
                # Return a filtered tuple including all indices except the one for the current node
                valid_indices: list[int] = list(agent_indices ^ {agent_node.index})

                # Declare data types but assign no values
                selected_indices: list[int]
                edge_weighting: float

                if agent_node.index < n_ni_agents:
                    # The Agent is non-influential
                    selected_indices = list(
                        np.random.choice(
                            valid_indices,
                            size=AGENT_CHARACTERISTICS["non_influential_connectivity"],
                            replace=False,
                        )
                    )

                    for selected_index in selected_indices:
                        edge_weighting = rd.uniform(
                            noninf_rel_range[0], noninf_rel_range[1]
                        )

                        # Flip the to_ and from_ indices to make the weighting representative of the impact that agent_node has on others.
                        new_edges["from_node"].append(selected_index)
                        new_edges["to_node"].append(agent_node.index)
                        new_edges["weighting"].append(edge_weighting)
                else:
                    # The Agent is influential
                    selected_indices = list(
                        np.random.choice(
                            valid_indices,
                            size=AGENT_CHARACTERISTICS["influential_connectivity"],
                            replace=False,
                        )
                    )

                    for selected_index in selected_indices:
                        edge_weighting = rd.uniform(inf_rel_range[0], inf_rel_range[1])

                        # Flip the to_ and from_ indices to make the weighting representative of the imapct that agent_node has on others.
                        new_edges["from_node"].append(selected_index)
                        new_edges["to_node"].append(agent_node.index)
                        new_edges["weighting"].append(edge_weighting)

            graph.add_edges(deepcopy(new_edges))
            created_hi_graphs.append(deepcopy(graph))

            # Manual garbage collection
            del new_edges, graph

        return created_hi_graphs

    def setup_model_li(self) -> None:
        """
        Adds the appropriate Agent and Graph objects to the li model.
        """
        self.li_model.add_agents(self.li_agents)
        self.li_model.add_graphs(
            self.li_graphs,
            deepcopy(HIERARCHY_NAMES),
            deepcopy(HIERARCHY_RW_DISTRIBUTIONS),
        )
        return None

    def setup_model_hi(self) -> None:
        """
        Adds the appropriate Agent and Graph objects to the hi model.
        """
        self.hi_model.add_agents(self.hi_agents)
        self.hi_model.add_graphs(
            self.hi_graphs,
            deepcopy(HIERARCHY_NAMES),
            deepcopy(HIERARCHY_RW_DISTRIBUTIONS),
        )
        return None

    def run_model_li(self) -> None:
        """
        Runs the low influence model.
        """
        self.li_model.iterate()
        return None

    def run_model_hi(self) -> None:
        """
        Runs the high influence model.
        """
        self.hi_model.iterate()
        return None


if __name__ == "__main__":
    tester: InfluentialTester = InfluentialTester()

    # Setup the models
    tester.setup_model_li()
    tester.setup_model_hi()

    # Run the low influence scenario
    tester.run_model_li()

    # Run the high influence scenario
    tester.run_model_hi()
