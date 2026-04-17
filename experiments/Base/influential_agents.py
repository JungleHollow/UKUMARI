from __future__ import annotations

import random as rd
from typing import Any

import src.GATOH.agents as agt
import src.GATOH.graphs as gr
import src.GATOH.logging as lg
import src.GATOH.model as md


class InfluentialTester:
    """
    A test class that sets up and runs models for a low-influence (li) scenario and a high-influence (hi) scenario for
    an experimental comaprison.

    In both cases, the models are composed of 5 hierarchies each with unique random walk distributions for their dynamic
    relationships. Additionally, both models will have 20 Agents within them, and run for a total of 100 iterations.

    In the case of the li model, all 20 Agents are "low influence" Agents with somewhat weaker relationships, and with
    hierarchy graphs having significantly less density of relationships between agents.

    In the case of the hi model, 18 of the Agents are "low influence" whilst 2 are replace with "high influence" Agents.
    These high influence Agents have somewhat stronger relationships with their neighbours, and a significantly higher
    density of relationships between them and other Agents in the hierarchies.
    """

    # The model parameters used when creating the ABModel instances
    MODEL_PARAMETERS: dict[str, Any] = {
        "iterations": 100,
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
        "influential_connectivity": 20,
        "social_susceptibility": 0.5,
        "personality": "social",
    }

    def __init__(self):
        self.li_agents: list[agt.Agent] = []
        self.create_li_agents()

        self.hi_agents: list[agt.Agent] = []
        self.create_hi_agents()

        self.li_graphs: list[gr.Graph] = []
        self.create_li_graphs(
            InfluentialTester.HIERARCHY_NAMES,
            InfluentialTester.HIERARCHY_RW_DISTRIBUTIONS,
            self.li_agents,
        )

        self.hi_graphs: list[gr.Graph] = []
        self.create_hi_graphs(
            InfluentialTester.HIERARCHY_NAMES,
            InfluentialTester.HIERARCHY_RW_DISTRIBUTIONS,
            self.hi_agents,
        )

        self.li_model: md.ABModel = md.ABModel(
            InfluentialTester.HIERARCHY_NAMES,
            InfluentialTester.HIERARCHY_RW_DISTRIBUTIONS,
            iterations=InfluentialTester.MODEL_PARAMETERS["iterations"],
            negation_threshold=InfluentialTester.MODEL_PARAMETERS["negation_thresh"],
            radicalisation_threshold=InfluentialTester.MODEL_PARAMETERS[
                "radical_thresh"
            ],
        )
        self.hi_model: md.ABModel = md.ABModel(
            InfluentialTester.HIERARCHY_NAMES,
            InfluentialTester.HIERARCHY_RW_DISTRIBUTIONS,
            iterations=InfluentialTester.MODEL_PARAMETERS["iterations"],
            negation_threshold=InfluentialTester.MODEL_PARAMETERS["negation_thresh"],
            radicalisation_threshold=InfluentialTester.MODEL_PARAMETERS[
                "radical_thresh"
            ],
        )

    def create_li_agents(self) -> list[agt.Agent]:
        """
        Creates the population of Agents to be used in the low influence scenario.

        :return: A list containing all the returned Agent objects.
        """
        created_agents: list[agt.Agent] = []

        return created_agents

    def create_hi_agents(self) -> list[agt.Agent]:
        """
        Creates the population of Agents to be used in the high influence scenario.

        :return: A list containing all the created Agent objects.
        """
        created_agents: list[agt.Agent] = []

        return created_agents

    def create_li_graphs(
        self,
        hierarchies: list[str],
        rw_distributions: list[tuple[float, float]],
        agents: list[agt.Agent],
    ) -> list[gr.Graph]:
        """
        Creates the graphs for the low influence model.

        :param hierarchies: A list of the names of the hierarchies that each graph should represent.
        :param rw_distributions: A dictionary defining the random walk distributions for each social hierarchy.
        :param agents: The population of Agents from which the graphs will be constructed.
        :return: A list containing all the created Graph objects.
        """
        created_graphs: list[gr.Graph] = []

        return created_graphs

    def create_hi_graphs(
        self,
        hierarchies: list[str],
        rw_distributions: list[tuple[float, float]],
        agents: list[agt.Agent],
    ) -> list[gr.Graph]:
        """
        Creates the graphs for the high influence model.

        :param hierarchies: A list of the names of the hierarchies that each graph should represent.
        :param rw_distributions: A dictionary defining the random walk distributions for each social hierarchy.
        :param agents: The population of Agents from which the graphs will be constructed.
        :return: A list containing all the created Graph objects.
        """
        created_graphs: list[gr.Graph] = []

        return created_graphs

    def run_model_li(self, model_parameters: dict[str, Any]) -> None:
        """
        Runs the low influence model.

        :param model_parameters: A dictionary containing the labelled values for the low influence model's parameters.
        """
        pass

    def run_model_hi(self, model_parameters: dict[str, Any]) -> None:
        """
        Runs the high influence model.

        :param model_parameters: A dictionary containing the labelled values for the high influence model's parameters.
        """
        pass


if __name__ == "__main__":
    tester: InfluentialTester = InfluentialTester()
