from __future__ import annotations

import os
import pickle
import random as rd
import warnings
import zipfile
from collections.abc import Iterable
from copy import deepcopy
from shutil import rmtree
from typing import Any, Iterator, override

import numpy as np
import polars as pl

from .utils import draw_random_value, random_coinflip, value_rw_delta

# Definition of all valid, existing Agent personality types
PERSONALITIES: list[str] = ["neutral", "rational", "erratic", "impulsive", "social"]


def _draw_personality() -> str:
    """
    An Agent utility function that randomly draws a valid Agent personality type.

    :return: The string representing the drawn personality type.
    """
    drawn_personality: str = rd.choice(PERSONALITIES)
    return drawn_personality


class Agent:
    """
    A class to define the Agent objects that will interact with each other in an agent-based model.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Supported positional arguments:
            - <string> to set the Agent's id
            - <dict> of {hierarchy_name : weight} for the personal value that this Agent assigns to each social hierarchy
            - <float> in the range [-1, 1] to set the Agent's initial opinion on the topic of interest
            - <bool> to define if the socially contagious belief will be of personal benefit to this Agent
            - (string, float) for the agent's defined personality and their social susceptibility

        :param args: positional arguments that can be passed to each Agent
        :param kwargs: keyword arguments that can be passed to each Agent
        """

        # Attributes declared but without initialisation will be defined by self.generate_agent() in a subsequent call if no args are passed
        self.id: str  # Can be any arbitrary string, but likely will follow the form XXXX0000 allowing for up to 9999 agents per community
        self.index: int  # The Agent's index within the AgentSet it belongs in

        self.social_weightings: dict[str, float] = {}
        self.is_silenced: dict[str, bool] = {}

        self.opinion: float  # Range always [-1, 1]
        self.previous_opinion: float = (
            0.0  # Used to handle updating during model iterations
        )

        self.personal_benefit: bool  # Whether the Agent is personally benefitted by the adoption of the 'social virus' that is being spread

        self.social_susceptibility: float  # Range always [0, 1]
        self.personality: str = "neutral"
        self.radicalised: bool = False

        # If no args have been passed, it is assumed that self.generate_agent() will be subsequently called
        if args:
            for arg in args:
                match arg:
                    case dict():
                        self.add_attribute("social_weightings", value=arg)
                        for hierarchy in self.social_weightings.keys():
                            self.is_silenced[hierarchy] = False
                    case float():
                        self.add_attribute("opinion", value=arg)
                    case tuple():
                        self.add_attribute("personality", value=arg[0])
                        self.add_attribute("social_susceptibility", value=arg[1])
                    case str():
                        self.id = arg
                    case bool():
                        self.personal_benefit = arg
        if kwargs:
            for key, value in kwargs.items():
                # No checking for duplicate keys; assume that explicitly added kwargs should override any args.
                self.add_attribute(key, value=value)

    def generate_agent(
        self,
        id: str,
        index: int,
        hierarchies: list[str],
        distribution: str = "gaussian",
        personality: str | None = None,
        parameters: dict | None = None,
        personal_benefit: bool | None = None,
    ) -> Agent:
        """
        Randomly generate an Agent object based on the input parameters.

        :param id: The id that has been assigned for this specific Agent object under the conditions of the model specifications.
        :param index: The index of the Agent object within the model's AgentSet.
        :param hierarchies: A list containing the names of all valid social hierarchies in the model.
        :param distribution: The distribution to use for relevant attribute generation (Valid distributions include: 'gaussian', 'beta')
        :param personality: A string defining what type of personality the agent will have (defaults to 'neutral' on Agent __init__)
        :param parameters: A dictionary containing the distribution parameters used to generate random values.
        :param personal_benefit: A boolean indicating if the Agent would be personally benefitted by the adoption of the 'social virus' being spread.
        :return: The generated Agent object.
        """
        # Begin by setting crucial information
        self.id = id
        self.index = index
        if personality:
            self.personality = personality

        if personal_benefit:
            self.personal_benefit = personal_benefit
        else:
            self.personal_benefit = random_coinflip("bool")

        # Generate a weighting for each hierarchy; initialise the is_silenced flag for that hierarchy
        for hierarchy in hierarchies:
            self.social_weightings[hierarchy] = draw_random_value(
                distribution, parameters=parameters
            )
            self.is_silenced[hierarchy] = False

        # Generate the Agent's initial opinion
        self.opinion = draw_random_value(distribution, parameters=parameters)

        # If the initial opinion is very strong, the Agent is initialised as radicalised
        if -0.9 >= self.opinion >= 0.9:
            self.radicalised = True

        # Generate the Agent's susceptibility to social contagion
        self.social_susceptibility = draw_random_value(
            distribution, parameters=parameters
        )

        return self

    def add_attribute(
        self,
        name: str,
        value: Any | None = None,
        parameters: dict | None = None,
        distribution: str | None = None,
        overwrite: bool = True,
    ) -> None:
        """
        Dynamically add an attribute to this Agent object. If "value" is passed, an explicit initial value is given;
        if "mean" and "sdev" are passed, a value is generated from a random distribution.
        Supported random distributions are:
            - "gaussian"
            - "beta"
            - "gamma"
            - "uniform"
            - "levy"

        :param name: The name of the attribute to be added.
        :param value: Initial value of the attribute.
        :param parameters: The distribution parameters that will be used with the specified distribution for parameter generation
        :param distribution: String to select which random distribution will be used to generate the value
        """
        if not value and not (distribution and parameters):
            raise ValueError(
                "Either explicit `value` or distribution and valid distribution parameters are expected when adding Agent attributes."
            )

        if not overwrite and name in self.__dict__.keys():
            # Raise a warning but do not change any attributes or crash the model if overwriting an existing attribute whilst explicitly passing overwrite = False
            warnings.warn(
                f"WARNING: Attempting to overwrite an existing Agent attribute ({name}) without meaning to.",
                category=UserWarning,
            )
        else:
            if value:
                # Assume a given explicit value always overrides (mean, sdev)
                self.__dict__[name] = value
            elif distribution and parameters:
                self.__dict__[name] = draw_random_value(
                    distribution, parameters=parameters
                )
        return None

    def get_attribute(self, name: str) -> Any:
        """
        Return any dynamically added attribute held by the Agent object.

        :param name: The name of the parameter to get.
        :return: The value stored for the parameter.
        """
        try:
            return self.__dict__[name]
        except KeyError:
            warnings.warn(
                f"WARNING: Attempting to get an Agent attribute ({name}) which doesn't exist.",
                category=UserWarning,
            )
            return None

    def store_previous_opinion(self) -> None:
        """
        A setter method that stores the Agent's current opinion into the previous opinion.
        """
        self.previous_opinion = self.opinion
        return None

    def change_opinion(self, opinion_delta: float) -> None:
        """
        A setter method that changes the Agent's current opinion by a given delta value.

        :param opinion_delta: The delta value by which to shift the Agent's current opinion.
        """
        self.opinion += opinion_delta

        # Constrain the opinion back to [-1.0, 1.0] as needed
        if self.opinion < -1.0:
            self.opinion = -1.0
        elif self.opinion > 1.0:
            self.opinion = 1.0
        return None

    def change_radicalisation(self, radicalisation: bool) -> None:
        """
        A setter method that changes the Agent's radicalisation value.

        :param radicalisation: The boolean radicalisation value to set.
        """
        self.radicalised = radicalisation
        return None

    def step(self, rw_distributions: dict[str, tuple[float, float]]) -> None:
        """
        Step the individual agent object:
            1. Handle dynamic social hierarchy weightings

        :param rw_distributions: A dictionary of <hierarchy name : (mean, variance)> defining the random walk distributions of each social
                                    hierarchy in the model
        """
        self.evolve_hierarchies(rw_distributions)
        return None

    def update(self, opinion_silenced: dict[str, bool], negation_ocurred: bool) -> None:
        """
        Updates the internal state of the agent after the model has stepped:
            1. Updates what social hierarchies the Agent's opinion is currently silenced in
            2. Inverts the Agent's current opinion if opinion negation ocurred

        :param opinion_silenced: A dictionary of <hierarchy : boolean> indicating which social hierarchies the Agent is silencing themselves in.
        :param negation_ocurred: A boolean indicating if opinion negation has ocurred in the current iteration.
        """
        self.is_silenced = opinion_silenced  # Update is_silenced
        if negation_ocurred:
            self.opinion *= -1.0  # Invert the Agent's current opinion
        return None

    def opinion_silencing(
        self,
        hierarchy: str,
        estimated_opinion_climate: float,
        silencing_threshold: float | None = None,
    ) -> tuple[bool, float]:
        """
        Determines if an agent will become silenced in a given social hierarchy based on their attributes.

        If no silencing threshold has been passed, each Agent's own social susceptibility is used as the threshold instead.

        :param hierarchy: The name of the social hierarchy that opinion silencing is being checked in.
        :param estimated_opinion_climate: The opinion climate perceived by the Agent in this hierarchy (not necessarily objectively 'accurate').
        :param silencing_threshold: A hierarchy or global silencing threshold that must be surpassed for silencing to occur.
        :return: A (boolean, absolute_difference) tuple indicating if silencing occurs, and the absolute difference between the perceived opinion climate and the Agent's own opinion.
        """
        # It is assumed that a radicalised Agent will never silence themselves regardless of the perceived opinion climate
        if self.radicalised:
            return False, 0.0

        threshold: float
        if silencing_threshold:
            threshold = silencing_threshold
        else:
            threshold = self.social_susceptibility

        absolute_difference: float = 0.0

        if self.personality in ["neutral", "rational", "erratic"]:
            # Cases where opinion silencing will be less influenced by the surrounding opinion climate.
            absolute_difference = abs(estimated_opinion_climate - self.opinion) * 0.8
        elif self.personality in ["impulsive", "social"]:
            # Cases where opinion silencing will be much more influenced by the surrounding opinion climate.
            absolute_difference = abs(estimated_opinion_climate - self.opinion)

        return absolute_difference > threshold, absolute_difference

    def opinion_negation(
        self, hierarchy: str, absolute_difference: float, threshold: float
    ) -> bool:
        """
        Checks if the Agent has experienced sufficiently 'overwhelming' social pressure in a hierarchy leading to a complete
        reversal of their opinion.

        :param hierarchy: The name of the social hierarchy where opinion negation is being checked for.
        :param absolute_difference: The absolute difference between the perceived opinion climate and the Agent's own opinion.
        :param threshold: A global model threshold that has been specified for this effect to occur.
        :return: A boolean indicating if the Agent's opinion experienced a total negation.
        """
        # It is assumed that a radicalised Agent will never experience a total opinion reversal regardless of the perceived opinion climate
        if self.radicalised:
            return False

        negation_strength: float = absolute_difference

        # Multiplication by (susceptibility * hierarchy weighting) will always decrease negation strength, whilst division will always increase it
        if self.personality in ["neutral", "rational"]:
            # Cases where opinion negation is less likely to occur
            negation_strength *= (
                self.social_susceptibility * self.social_weightings[hierarchy]
            )
        elif self.personality in ["erratic", "impulsive", "social"]:
            # Cases where opinion negation is more likely to occur
            negation_strength /= (
                self.social_susceptibility * self.social_weightings[hierarchy]
            )

        return negation_strength > threshold

    def radicalisation(
        self,
        hierarchy_changes: list[float],
        hierarchies: list[str],
        threshold: float,
    ) -> bool:
        """
        Uses the agent's own opinion as well as the neighbours' opinions to determine if
        the agent has become radicalised in their actions.

        :param hierarchy_changes: A list of the opinion changes caused in each social hierarchy by neighbours during this iteration.
        :param hierarchy_names: A list of hierarchy names in an order corresponding to the passed hierarchy changes list.
        :param threshold: The radicalisation threshold that has been defined at the global level in the model.
        :return: A boolean indicating if the Agent has become radicalised or not.
        """
        # If the Agent is already radicalised, always return False (as the Agent cannot become 'radicalised' again)
        if self.radicalised:
            return False

        # Absolute opinion declared here to reduce calls to abs() in the match statement
        absolute_opinion: float = abs(self.opinion)

        match self.personality:
            case "neutral":
                # This will mean that radicalisation is exclusively determined by the strength of the Agent's opinion
                if absolute_opinion >= threshold:
                    self.radicalised = True
                    return self.radicalised
            case "rational":
                # This will likely mean that the agent is more disposed towards considering tangible benefits and their own
                # opinions when determining radicalisation, rather than external influences
                if absolute_opinion >= threshold and self.personal_benefit:
                    self.radicalised = True
                    return self.radicalised
                elif absolute_opinion >= threshold and not self.personal_benefit:
                    # In the case where the threshold is met but there is no explicit personal benefit, radicalisation is treated as a coinflip
                    self.radicalised = random_coinflip("bool")
                    return self.radicalised
            case "erratic":
                # Radicalisation is influenced by personal opinion to some extent, but is largely stochastically determined
                random_threshold: float = rd.random()
                if (absolute_opinion * 0.9 >= threshold) and (random_threshold >= 0.5):
                    self.radicalised = True
                    return self.radicalised
            case "impulsive":
                # The agent places very strong consideration on tangible benefits over anything else
                if (
                    absolute_opinion >= threshold / 2
                ):  # (threshold / 2) as the Agent behaves impulsively and less is required for them to consider becoming radicalised
                    self.radicalised = self.personal_benefit
                    return self.radicalised
            case "social":
                # Radicalisation is strongly determined by the opinion climate and neighbour opinions rather than internal factors
                absolute_changes: float = 0.0

                for change in hierarchy_changes:
                    absolute_change: float = abs(change)
                    if absolute_change >= self.social_susceptibility:
                        # A strong opinion change was caused by some hierarchy
                        self.radicalised = True
                        return self.radicalised
                    else:
                        absolute_changes += absolute_change
                # If no changes were strong enough individually, check for the aggregate (with a relatively lower threshold)
                if absolute_changes >= self.social_susceptibility * (
                    len(hierarchy_changes) // 2
                ):
                    self.radicalised = True
                    return self.radicalised
        # If this is somehow reached, an error has ocurred (but False is returned just in case)
        return False

    def evolve_hierarchies(
        self, rw_distributions: dict[str, tuple[float, float]]
    ) -> None:
        """
        Experimental function that aims to model the constantly evolving 'intrinsic value' that Agents place on
        the social hierarchies that they belong in over time.

        :param rw_distributions: A dictionary specifying the global random walk distributions defined for each hierarchy in the model.
        """
        for key, value in rw_distributions.items():
            rw_result: float = value_rw_delta(
                self.social_weightings[key], value[0], value[1]
            )

            # Constrain the result back to [-1, 1] if necessary
            if rw_result < -1.0:
                self.social_weightings[key] = -1.0
            elif rw_result > 1.0:
                self.social_weightings[key] = 1.0
            else:
                self.social_weightings[key] = rw_result
        return None

    def life_events(self) -> None:
        """
        Experimental function that aims to model the ways in which Agent behaviours change according to major random life events over time
        """
        # TODO: Implement this function
        raise NotImplementedError(
            "Agent life events have not been implemented as a feature yet."
        )

    def __in__(self, iterable: Iterable[Agent]) -> bool:
        """
        Determine if the Agent is contained within an iterable of Agents

        :param iterable: The iterable of Agent objects in which membership is being determined.
        :return: A boolean indicating if this Agent is contained within the iterable.
        """
        for agent in iterable:
            if self == agent:
                return True
        return False

    @override
    def __str__(self) -> str:
        """
        An override to what calling `print()` on this object will output
        """
        return f"Agent {self.id} which {'is' if self.radicalised else 'is not'} radicalised with an opinion value of {self.opinion}"


class AgentSet:
    """
    An ordered collection of Agent objects that maintains consistency for the Model
    """

    def __init__(self, model: Any) -> None:
        """
        :param model: The parent ABModel object that this AgentSet is being attached to
        """
        self.parent_model = model
        self.agents: list = []
        self.random: rd.Random = rd.Random()

    def save_agentset(self, directory_path: str) -> None:
        """
        Save the Agent objects into a compressed subdirectory representing the the saved AgentSet.

        :param directory_path: The path to the directory where the agentset subdirectory should be created.
        """
        subdirectory_path: str = f"{directory_path}/_agentset"

        # Removes the subdirectory if it already exists to allow for a new overwrite
        if os.path.isdir(subdirectory_path):
            rmtree(subdirectory_path)

        # Create the _agentset subdirectory
        os.mkdir(subdirectory_path)

        agent_save_paths: list[str] = []

        for agent in self.agents:
            agent_save_path: str = f"{subdirectory_path}/_agent_{agent.id}.pkl"
            # Pickle the python Agent object
            with open(agent_save_path, "wb") as agent_file:
                pickle.dump(agent, agent_file)
            agent_save_paths.append(agent_save_path)

        zip_path: str = f"{subdirectory_path}.zip"

        # Removes the zip file if it already exists to allow for a new overwrite
        if os.path.exists(zip_path):
            os.remove(zip_path)

        # Compress the subdirectory to minimise storage and encapsulate all the Agents into a single object
        with zipfile.ZipFile(
            zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as subdir_zip:
            for graph_path in agent_save_paths:
                subdir_zip.write(graph_path)

        # Remove the uncompressed subdirectory if compression was successful
        if os.path.exists(zip_path):
            rmtree(subdirectory_path)

        return None

    def load_agentset(self, load_path: str) -> None:
        """
        Loads an AgentSet that has been saved following the same process as in the save_agentset() function.

        :param load_path: The path to the model's overall save directory.
        """
        zip_load_path: str = f"{load_path}/_agentset.zip"

        if not os.path.exists(zip_load_path):
            raise FileNotFoundError(
                f"No saved AgentSet was found at the path: {zip_load_path}"
            )

        # The path to the uncompressed agentset subdirectory
        subdirectory_path: str = f"{load_path}/_agentset"

        # Remove any existing subdirectory with the same name to replace it with the newly loaded one
        if os.path.isdir(subdirectory_path):
            os.rmdir(subdirectory_path)

        # Create the uncompressed directory
        os.mkdir(subdirectory_path)

        # Extract all the Agent pickles to the uncompressed directory
        with zipfile.ZipFile(
            zip_load_path, mode="r", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as subdir_zip:
            subdir_zip.extractall(path=subdirectory_path)

        # Unpickle each Agent object and add it to the AgentSet.
        for agent_pickle_path in os.listdir(subdirectory_path):
            with open(agent_pickle_path, "rb") as agent_pickle:
                agent_object: Agent = pickle.load(agent_pickle)
                self.add(agent_object)

        return None

    def __len__(self) -> int:
        """
        A method that defines how an AgentSet object checks its length.

        :return: the number of agents present in the AgentSet
        """
        return len(self.agents)

    def __iter__(self) -> Iterator[Agent]:
        """
        A method that defines how the AgentSet iterates over its Agents.

        :return: An Iterator object that iterates over all the Agents within the AgentSet.
        """
        return self.agents.__iter__()

    def __in__(self, agent: Agent) -> bool:
        """
        A method defining how an AgentSet checks for an Agent's membership.

        :param agent: The specific Agent object to check for.
        :return: A boolean indicating if the Agent object is in the AgentSet.
        """
        return agent in self.agents

    def __contains__(self, agent: Agent) -> bool:
        """
        A secondary method defining how an AgentSet checks for an Agent's membership.

        :param agent: The specific Agent object to check for.
        :return: A boolean indicating if the specified Agent object is in the AgentSet.
        """
        return self.agents.__contains__(agent)

    def __getitem__(self, item: int | slice) -> Any:
        """
        Retrieve an Agent or slice of Agents from the AgentSet.
        :param item: The index or slice for selecting the agents.
        :return: The selected agent or slice of agents based on the specified item.
        """
        return self.agents.__getitem__(item)

    def add(self, agent: Agent) -> int:
        """
        Add an Agent to the AgentSet.

        :param agent: The Agent object to be added.
        :return: The index of the newly added Agent.
        """
        self.agents.append(agent)
        self.agents[-1].index = len(self.agents)
        return self.agents[-1].index

    def update_indices(self) -> None:
        """
        Iterate over the AgentSet and update the current Agent object index values
        """
        for idx, agent in enumerate(self.agents):
            agent.index = idx
        return None

    def discard(self, agent: Agent) -> bool:
        """
        Removes an Agent from the AgentSet which matches the input Agent; does not return an error if the Agent does not exist.
        :param agent: The Agent object that should be removed from the set
        :return: A boolean to flag if the Agent was removed successfully or not
        """
        for idx, agnt in enumerate(self.agents):
            if agent == agnt:
                left_half: list[Agent] = deepcopy(self.agents[:idx])
                right_half: list[Agent] = deepcopy(self.agents[idx + 1 :])

                self.agents = deepcopy(left_half) + deepcopy(right_half)
                del left_half, right_half

                self.update_indices()
                return True
        return False

    def agent_at_index(self, index: int) -> Agent | None:
        """
        Returns the Agent object at the given index in the AgentSet.

        :param index: The index within the AgentSet to inspect.
        :return: The Agent object at the specified index.
        """
        try:
            return self.agents[index]
        except IndexError:
            print(
                f"Index {index} is out of bounds for the AgentSet. Only {len(self.agents)} Agents have been created."
            )
            return None

    def get_agent_by_id(self, id: str) -> Agent | None:
        """
        Searches the AgentSet for an Agent with the given id and returns its object if it exists.

        :param id: The id that was assigned to the Agent object at creation.
        :return: The Agent object with the specified id.
        """
        for agent in self.agents:
            if agent.id == id:
                return agent

        raise KeyError(
            f"The Agent with id '{id}' does not exist in the AgentSet -- unable to return an Agent object."
        )

    def get_index(self, agent: Agent) -> int:
        """
        Returns the index within the AgentSet of the input Agent object.

        :param agent: The Agent object whose index is being searched for.
        :return: The index of the Agent object within the AgentSet.
        """
        for idx, agt in enumerate(self.agents):
            if agent.id == agt.id:
                return idx

        raise KeyError(
            f"The Agent {agent.id} does not exist in the AgentSet -- unable to return an index."
        )

    def discard_index(self, index: int) -> bool:
        """
        Removes the Agent at the specified index in the AgentSet; does not return an error if the index is out of bounds.
        :param index: The index in the AgentSet which is to be removed
        :return: A boolean to flag if the Agent was removed successfully or not
        """
        if 0 < index < len(self.agents):
            left_half: list[Agent] = deepcopy(self.agents[:index])
            right_half: list[Agent] = deepcopy(self.agents[index + 1 :])

            self.agents = deepcopy(left_half) + deepcopy(right_half)
            del left_half, right_half

            self.update_indices()
            return True
        return False

    def remove(self, agent: Agent) -> bool:
        """
        Removes an Agent from the AgentSet which matches the input Agent; returning an error if such an Agent does not exist.
        :param agent: The Agent object that should be removed from the set
        :return: A boolean to flag that the Agent was removed successfully
        """
        for idx, agnt in enumerate(self.agents):
            if agent == agnt:
                left_half: list[Agent] = deepcopy(self.agents[:idx])
                right_half: list[Agent] = deepcopy(self.agents[idx + 1 :])

                self.agents = deepcopy(left_half) + deepcopy(right_half)
                del left_half, right_half

                self.update_indices()
                return True
        raise KeyError(
            f"Tried to remove an Agent with id {agent.id} that doesn't exist in the AgentSet"
        )

    def remove_index(self, index: int) -> bool:
        """
        Removes the Agent at the specified index in the AgentSet; returning an error if the index is out of bounds.
        :param index: The index in the AgentSet which is to be removed
        :return: A boolean to flag that the Agent was removed successfully
        """
        if 0 < index < len(self.agents):
            left_half: list[Agent] = deepcopy(self.agents[:index])
            right_half: list[Agent] = deepcopy(self.agents[index + 1 :])

            self.agents = deepcopy(left_half) + deepcopy(right_half)
            del left_half, right_half

            self.update_indices()
            return True
        raise IndexError(
            f"Tried to remove an Agent at out of bounds index {index} from the AgentSet"
        )

    def sample(self, n: int) -> list[Agent]:
        """
        Randomly draw n Agents from the AgentSet without replacement
        :param n: The number of agents to sample
        :return: A list of the agents sampled from the AgentSet
        """
        sampled_agents: list[Agent] = self.random.sample(self.agents, n)
        return deepcopy(sampled_agents)

    @override
    def __getstate__(self) -> dict:
        """
        Retrive the current state of the AgentSet for serialization.
        :return: a dictionary representing the current state of the AgentSet
        """
        return {"agents": self.agents, "random": self.random}
