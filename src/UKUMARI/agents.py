from __future__ import annotations

import warnings
from collections.abc import Callable, Generator, Iterable
from copy import deepcopy
from random import Random
from typing import Any, override

import numpy as np
import polars as pl


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
            - (string, float) for the agent's defined personality and their social susceptibility

        :param args: positional arguments that can be passed to each Agent
        :param kwargs: keyword arguments that can be passed to each Agent
        """

        self.id: str  # Can be any arbitrary string, but likely will follow the form XXXX0000 allowing for up to 9999 agents per community
        self.index: int  # The Agent's index within the AgentSet it belongs in

        self.social_weightings: dict[str, float] = {}
        self.opinion: float = 0.0  # Range always [-1, 1]

        self.previous_opinion: float = (
            0.0  # Used to handle updating during model iterations
        )
        self.social_susceptibility: float = 0.5  # Range always [0, 1]
        self.personality: str = "neutral"
        self.radicalised: bool = False

        if args:
            for arg in args:
                match arg:
                    case dict():
                        self.add_attribute("social_weightings", value=arg)
                    case float():
                        self.add_attribute("opinion", value=arg)
                    case tuple():
                        self.add_attribute("personality", value=arg[0])
                        self.add_attribute("social_susceptibility", value=arg[1])
                    case str():
                        self.id = arg

        if kwargs:
            for key, value in kwargs.items():
                # No checking for duplicate keys; assume that explicitly added kwargs should override any args.
                self.add_attribute(key, value=value)

    def add_attribute(
        self,
        name: str,
        value: Any | None = None,
        mean: float | None = None,
        sdev: float | None = None,
        distribution: str | None = None,
        overwrite: bool = True,
    ) -> None:
        # TODO: Add support for additional random distributions
        """
        Dynamically add an attribute to this Agent object. If "value" is passed, an explicit initial value is given;
        if "mean" and "sdev" are passed, a value is generated from a random distribution.
        Supported random distributions are:
            - "normal"
            - "uniform" -- In the case of uniform, `mean` will be treated as the median value, and `sdev` as the distance between the median and the boundaries

        :param name: The name of the attribute to be added.
        :param value: Optional initial value of the attribute.
        :param mean: Optional mean of the random distribution from which to generate the value
        :param sdev: Optional standard deviation of the random distribution from which to generate the value
        :param distribution: Optional string to select which random distribution will be used to generate the value
        """
        if not value and (not mean and not sdev):
            raise ValueError(
                "Either explicit `value` or distribution `mean` and `sdev` are expected when adding Agent attributes."
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
            elif mean and sdev:
                match distribution:
                    case "normal":
                        self.__dict__[name] = np.random.normal(loc=mean, scale=sdev)
                    case "uniform":
                        uniform_range = (mean - sdev, mean + sdev)
                        self.__dict__[name] = np.random.uniform(
                            low=uniform_range[0], high=uniform_range[1]
                        )
                    case None:
                        # Fall back on the normal distribution
                        self.__dict__[name] = np.random.normal(loc=mean, scale=sdev)

    def get_attribute(self, name: str) -> Any:
        try:
            return self.__dict__[name]
        except KeyError:
            warnings.warn(
                f"WARNING: Attempting to get an Agent attribute ({name}) which doesn't exist.",
                category=UserWarning,
            )
            return None

    def step(self):
        """
        Step the individual agent object
        """
        pass

    def update_state(self):
        """
        Updates the internal state of the agent after the model has stepped.
        """
        pass

    def radicalisation(self, neighbours: Iterable[Agent]) -> bool:
        """
        Uses the agent's own opinion as well as the neighbours' opinions to determine if
        the agent has become radicalised in their actions.

        :param neighbours: A list of all agents that "neighbour" this agent in any model layer.
        """
        # If the Agent is already radicalised, always return True
        if self.radicalised:
            return True

        match self.__getattribute__("personality"):
            case "rational":
                pass
            case "erratic":
                pass
            case "impulsive":
                pass
            case None:
                pass
        return False  # TODO: Finish this method (returning False to suppress typing warnings)

    def evolve_relationships(self):
        """
        Experimental function that aims to model the constantly evolving relationships between Agents over time
        """
        raise NotImplementedError(
            "Agent relationship evolution has not been implemented as a feature yet."
        )

    def life_events(self):
        """
        Experimental function that aims to model the ways in which Agent behaviours change according to major random life events over time
        """
        raise NotImplementedError(
            "Agent life events have not been implemented as a feature yet."
        )

    def __in__(self, iterable: Iterable[Agent]) -> bool:
        """
        Determine if the Agent is contained within an iterable of Agents

        :param iterable: The iterable of Agent objects in which membership is being determined
        """
        for agent in iterable:
            if self == agent:
                return True
        return False

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
        self.random: Random = Random()

    def __len__(self) -> int:
        """
        :return: the number of agents present in the AgentSet
        """
        return len(self.agents)

    def __contains__(self, agent: Agent) -> bool:
        """
        :param agent: the specific Agent object to check for
        :return: a boolean indicating if the specified Agent object is in the AgentSet
        """
        return self.agents.__contains__(agent)

    def __getitem__(self, item: int | slice) -> Any:
        """
        Retrieve an Agent or slice of Agents from the AgentSet.
        :param item: the index or slice for selecting the agents
        :return: the selected agent or slice of agents based on the specified item
        """
        return self.agents.__getitem__(item)

    def add(self, agent: Agent) -> int:
        """
        Add an Agent to the AgentSet.
        :param agent: the Agent object to be added
        """
        self.agents.append(agent)
        self.agents[-1].index = len(self.agents)
        return 1

    def update_indices(self) -> None:
        """
        Iterate over the AgentSet and update the current Agent object index values
        """
        for idx, agent in enumerate(self.agents):
            agent.index = idx

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

    def discard_index(self, index: int) -> bool:
        """
        Removes the Agent at the specified index in the AgentSet; does not return an error if the index is out of bounds.
        :param index: The index in the AgentSet which is to be removed
        :return: A boolean to flag if the Agent was removed successfully or not
        """
        if 0 < index < len(self.agents):
            left_half: list[Agent] = deepcopy(self.agents[:index])
            right_half: list[Agent] = deepcopy(self.agents[index + 1:])
            
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
        raise KeyError(f"Tried to remove an Agent with id {agent.id} that doesn't exist in the AgentSet")

    def remove_index(self, index: int) -> bool:
        """
        Removes the Agent at the specified index in the AgentSet; returning an error if the index is out of bounds.
        :param index: The index in the AgentSet which is to be removed
        :return: A boolean to flag that the Agent was removed successfully
        """
        if 0 < index < len(self.agents):
            left_half: list[Agent] = deepcopy(self.agents[:index])
            right_half: list[Agent] = deepcopy(self.agents[index + 1:])
            
            self.agents = deepcopy(left_half) + deepcopy(right_half)
            del left_half, right_half

            self.update_indices()
            return True
        raise IndexError(f"Tried to remove an Agent at out of bounds index {index} from the AgentSet")

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
