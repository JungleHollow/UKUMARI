from __future__ import annotations

import os
import random as rd
from copy import deepcopy
from typing import Any

import numpy as np

import src.GATOH.agents as agt
import src.GATOH.graphs as gr
import src.GATOH.model as md


class GraphAlgTester:
    """
    A test class that sets up and runs models for each of the random graph generation algorithms in the GATOH framework for
    an experimental comparison.
    """

    def __init__(self, existing: bool = False) -> None:
        """
        :param existing: A flag indicating if the tester is loading an existing experiment.
        """
        # Store the class parameters within the instance
        self.algorithms: list[str] = TEST_PARAMETERS["generation_algorithms"]

        self.existing: bool = existing

        # Define the data types without assigning values
        self.model_agents: list[agt.Agent]

        # Dynamic model space
        self.models: dict[str, md.ABModel] = {}

        # Create the model objects no matter what
        for algorithm in self.algorithms:
            algorithm_model: md.ABModel = md.ABModel(
                deepcopy(HIERARCHY_NAMES),
                deepcopy(HIERARCHY_RW_DISTRIBUTIONS),
                save_dir=MODEL_SAVEDIRS[algorithm],
                data_file=MODEL_DATAFILES[algorithm],
                model_id=algorithm.upper(),
            )
            self.models[algorithm] = algorithm_model

        if not self.existing:  # Objects are needed to define and run the models
            pass

    def create_agents(self) -> list[agt.Agent]:
        """
        Generates and returns the population of Agents that will be used across all models.

        For the purposes of this experiment, the populations will be identical across all models,
        and only the way that Agent relationships are generated will change.

        :return: A list containing all the returned Agent objects.
        """
        # TODO: IMPLEMENT THIS FUNCTION
        return []

    def create_graphs(
        self,
        algorithm: str,
        hierarchies: list[str],
        rw_distributions: list[tuple[float, float]],
        agents: list[agt.Agent],
    ) -> list[gr.Graph]:
        """
        Creates the graphs for an arbitrary model.

        For the purposes of this experiment, each graph contains all Agents in the population,
        and the relationships are created using the specified algorithm for every graph in a given model.

        :param algorithm: The graph generation algorithm that is being used.
        :param hierarchies: A list of the names of the hierarchies that each graph should represent.
        :param rw_distributions: A dictionary defining the random walk distributions for each social hierarchy.
        :param agents: The population of Agents from which the graphs will be constructed.
        :return: A list containing all the created Graph objects.
        """
        # TODO: IMPLEMENT THIS FUNCTION
        return []

    def load_models(self) -> None:
        """
        Loads the model objects that have been previously saved at their respective directories.
        """
        # TODO: IMPLEMENT THIS FUNCTION
        return None

    def setup_models(self) -> None:
        """
        Adds the appropriate Agent and Graph objects to both models.
        """
        # TODO: IMPLEMENT THIS FUNCTION
        return None

    def run_models(self) -> None:
        """
        Runs each model in the tester class.
        """
        # TODO: IMPLEMENT THIS FUNCTION
        return None


if __name__ == "__main__":
    # The parameters set for the tester class itself
    TEST_PARAMETERS: dict[str, Any] = {
        "generation_algorithms": [
            "small-world",
            "scale-free",
            "random",
            "blockmodel",
        ]
    }

    # Default model parameters will be used for all scenarios, no need to set explicitly

    # The social hierarchies that will exist in the models
    HIERARCHY_NAMES: list[str] = [
        "family",
        "friends",
        "neighbours",
        "religion",
        "cultural",
    ]

    # The random walk distribution parameters for each hierarchy
    HIERARCHY_RW_DISTRIBUTIONS: list[tuple[float, float]] = [
        (0.0, 0.01),  # Family
        (0.0, 0.05),  # Friends
        (0.0, 0.08),  # Neighbours
        (0.0, 0.15),  # Religion
        (0.0, 0.2),  # Cultural
    ]

    # The hierarchy weightings that will be given to each hierarchy (shared amongst all agents)
    HIERARCHY_WEIGHTINGS: dict[str, float] = {
        "family": 0.9,
        "friends": 0.7,
        "neighbours": 0.55,
        "religion": 0.5,
        "cultural": 0.25,
    }

    # Defining the distributions of the characteristics for Agents that will be used in the experiment
    AGENT_CHARACTERISTICS: dict[str, Any] = {
        "opinion": (-0.8, 0.8),
        "connectivity": 6,
        "relationship": (-0.2, 0.8),
        # Other characteristics will be fully stochastically determined
    }

    # Define the save paths for each model's logged variables (must point to a .csv file)
    MODEL_DATAFILES: dict[str, str] = {}
    for algorithm in TEST_PARAMETERS["generation_algorithms"]:
        MODEL_DATAFILES[algorithm] = (
            f"./experiments/Base/GraphAlgorithms/{algorithm}_model_variables.csv"
        )

    # Define the save directories for each model
    MODEL_SAVEDIRS: dict[str, str] = {}
    for algorithm in TEST_PARAMETERS["generation_algorithms"]:
        MODEL_SAVEDIRS[algorithm] = (
            f"./experiments/Base/GraphAlgorithms/GraphAlgorithms_{algorithm}"
        )

    tester: GraphAlgTester

    if len(list(os.walk("./experiments/Base/GraphAlgorithms"))) < len(
        TEST_PARAMETERS["generation_algorithms"]
    ):
        # At least one algorithm's save subdirectory does not exist
        tester = GraphAlgTester()
    else:  # Assume that all existing subdirectories include every algorithms's valid save subdirectory...
        tester = GraphAlgTester(existing=True)
