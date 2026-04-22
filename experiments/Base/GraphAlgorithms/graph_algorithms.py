from __future__ import annotations

import os
import random as rd
from copy import deepcopy
from hashlib import algorithms_available
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
        self.num_agents: int = TEST_PARAMETERS["num_agents"]

        self.existing: bool = existing

        # Define the data types without assigning values
        self.model_agents: list[agt.Agent]
        self.model_graphs: dict[str, list[gr.Graph]]

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

        # TODO: Optimise this statement to account for partial missing and existing saves
        if not self.existing:  # Objects are needed to define and run the models
            self.model_agents = self.create_agents(self.num_agents)

            self.model_graphs = {}
            for algorithm in self.algorithms:
                algorithm_graphs: list[gr.Graph] = self.create_graphs(
                    algorithm,
                    deepcopy(HIERARCHY_NAMES),
                    deepcopy(HIERARCHY_RW_DISTRIBUTIONS),
                    deepcopy(self.model_agents),
                )
                self.model_graphs[algorithm] = algorithm_graphs

    def create_agents(self, num_agents: int) -> list[agt.Agent]:
        """
        Generates and returns the population of Agents that will be used across all models.

        For the purposes of this experiment, the populations will be identical across all models,
        and only the way that Agent relationships are generated will change.

        :return: A list containing all the returned Agent objects.
        """
        created_agents: list[agt.Agent] = []

        opinion_range: tuple[float, float] = AGENT_CHARACTERISTICS["opinion"]

        # Define data types but do not assign any values
        agent_id: str
        agent_opinion: float
        agent: agt.Agent
        agent_behaviour: tuple[str, float]
        personal_benefit: bool

        created_count: int = 0
        while created_count < num_agents:
            created_count += 1
            agent_id = f"AGNT{created_count:04}"

            # Stochastically generate the important Agent attributes
            agent_opinion = rd.uniform(opinion_range[0], opinion_range[1])
            agent_behaviour = (agt._draw_personality(), rd.uniform(0.0, 1.0))
            personal_benefit = rd.choice([True, False])

            agent = agt.Agent(
                agent_id,
                deepcopy(HIERARCHY_WEIGHTINGS),
                agent_opinion,
                agent_behaviour,
                personal_benefit,
            )

            created_agents.append(deepcopy(agent))

            # Manual garbage collection
            del agent

        return deepcopy(created_agents)

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
        created_graphs: list[gr.Graph] = []

        rel_range: tuple[float, float] = AGENT_CHARACTERISTICS["relationship"]

        for idx, hierarchy in enumerate(hierarchies):
            graph: gr.Graph = gr.Graph(hierarchy, rw_distributions[idx])

            graph.generate_graph(
                deepcopy(agents), method=algorithm, relationship_range=rel_range
            )

            created_graphs.append(deepcopy(graph))

            # Manual garbage collection
            del graph

        return deepcopy(created_graphs)

    def load_models(self, existing_saves: list[str] | None = None) -> None:
        """
        Loads the model objects that have been previously saved at their respective directories.

        :param existing_saves: An optional partial list of the algorithms representing existing models that can be loaded.
        """
        if existing_saves:
            for existing_save in existing_saves:
                self.models[existing_save].load_model(MODEL_SAVEDIRS[existing_save])
            return None

        for algorithm in self.algorithms:
            self.models[algorithm].load_model(MODEL_SAVEDIRS[algorithm])
        return None

    def setup_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Adds the appropriate Agent and Graph objects to both models.

        :param missing_saves: An optional partial list of the algorithms representing models that should be setup.
        """
        if missing_saves:
            for missing_save in missing_saves:
                self.models[missing_save].add_agents(deepcopy(self.model_agents))
                self.models[missing_save].add_graphs(
                    deepcopy(self.model_graphs[missing_save]),
                    deepcopy(HIERARCHY_NAMES),
                    deepcopy(HIERARCHY_RW_DISTRIBUTIONS),
                )
            return None

        for algorithm in self.algorithms:
            self.models[algorithm].add_agents(deepcopy(self.model_agents))
            self.models[algorithm].add_graphs(
                deepcopy(self.model_graphs[algorithm]),
                deepcopy(HIERARCHY_NAMES),
                deepcopy(HIERARCHY_RW_DISTRIBUTIONS),
            )
        return None

    def run_models(self, missing_saves: list[str] | None = None) -> None:
        """
        Runs each model in the tester class.

        :param missing_saves: An optional partial list of the algorithms representing models that should be run.
        """
        if missing_saves:
            for missing_save in missing_saves:
                self.models[missing_save].iterate()
                self.models[missing_save].save_model()
            return None

        for algorithm in self.algorithms:
            self.models[algorithm].iterate()
            self.models[algorithm].save_model()

        return None


if __name__ == "__main__":
    # The parameters set for the tester class itself
    TEST_PARAMETERS: dict[str, Any] = {
        "generation_algorithms": [
            "small-world",
            "scale-free",
            "random",
            "blockmodel",
        ],
        "num_agents": 50,
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

    # Check for existing saved models and store the relevant information
    save_dirs = list(os.walk("./experiments/Base/GraphAlgorithms"))[0][1]

    directory_missing: bool = False
    existing_savedirs: list[str] = []
    missing_savedirs: list[str] = []

    for algorithm, save_dir in MODEL_SAVEDIRS.items():
        dir_name: str = deepcopy(save_dir).split("/")[-1]
        if dir_name in save_dirs:
            existing_savedirs.append(algorithm)
        else:
            directory_missing = True
            missing_savedirs.append(algorithm)

    if directory_missing:  # At least one algorithm's save subdirectory does not exist
        # Create the tester normally, setup the models, and begin iterations
        tester = GraphAlgTester()

        if len(existing_savedirs) > 0:  # At least one model exists
            tester.load_models(existing_saves=existing_savedirs)
            tester.setup_models(missing_saves=missing_savedirs)
            tester.run_models(missing_saves=missing_savedirs)
        else:  # Assume all models should be newly created and run
            tester.setup_models()
            tester.run_models()
    else:  # Assume that all existing subdirectories include every algorithms's valid save subdirectory...
        # Create the tester in "existing" mode, and examine the results
        tester = GraphAlgTester(existing=True)
        tester.load_models()

    # TODO: Add the graph visualisation functions here once those features are implemented...
