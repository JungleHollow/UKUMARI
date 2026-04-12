from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoggerVariables:
    """
    Dataclass that defines the simulation variables that are stored and tracked by the Logger.
    """

    # The maximum number of iterations that the simulation will run for
    max_iterations: int
    # The aggregated community opinion climate at each timestep
    aggregate_opinions: list[float] = field(default_factory=list)
    # The number of radicalised agents that exist in the model at each timestep
    radicalised_agents: list[int] = field(default_factory=list)
    # The total count of opinion silencing effects that have ocurred in the simulation over time
    silenced_agents: list[int] = field(default_factory=list)
    # The total count of opinion negation effects that have ocurred in the simulation over time
    negated_agents: list[int] = field(default_factory=list)
    # The calculated layer interdependence of each hierarchy at each timestep
    layer_interdependences: dict[str, list[float]] = field(default_factory=dict)
    # The calculated layer polarisation of each hierarchy at each timestep
    layers_polarisation: dict[str, list[float]] = field(default_factory=dict)
    # The log odds of radicalisation in the model at each timestep
    radicalisation_logodds: list[float] = field(default_factory=list)
    # The current iteration that the simulation is at
    current_iteration: int = 0

    def __init__(self, max_iterations: int, hierarchies: list[str]) -> None:
        """
        Store the number of max iterations and initialise all lists and dictionaries with the appropriate hierarchy names
        and sizes to match the number of iterations.

        :param max_iterations: The maximum number of iterations that the model will run its simulation for.
        :param hierarchies: A list containing the names of all social hierarchies present in the model.
        """
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.aggregate_opinions = [0.0 for _ in range(self.max_iterations)]
        self.radicalised_agents = [0 for _ in range(self.max_iterations)]
        self.silenced_agents = [0 for _ in range(self.max_iterations)]
        self.negated_agents = [0 for _ in range(self.max_iterations)]
        self.radicalisation_logodds = [0.0 for _ in range(self.max_iterations)]

        self.layer_interdependences = {}
        self.layers_polarisation = {}

        for hierarchy in hierarchies:
            self.layer_interdependences[hierarchy] = [
                0.0 for _ in range(self.max_iterations)
            ]
            self.layers_polarisation[hierarchy] = [
                0.0 for _ in range(self.max_iterations)
            ]

    def increment_radicalised(self, flag: bool) -> None:
        """
        A simple setter function that checks the input flag and updates the radicalisation count accordingly.

        :param flag: A boolean flag indicating if radicalisation ocurred.
        """
        if flag:
            self.radicalised_agents[self.current_iteration - 1] += 1

    def increment_silenced(self, flag: bool) -> None:
        """
        A simple setter function that checks the input flag and updates the opinion silencing events count accordingly.

        :param flag: A boolean flag indicating if opinion silencing ocurred.
        """
        if flag:
            self.silenced_agents[self.current_iteration - 1] += 1

    def increment_negated(self, flag: bool) -> None:
        """
        A simple setter function that checks the input flag and updates the opinion negation events count accordingly.

        :param flag: A boolean flag indicating if opinion negation ocurred.
        """
        if flag:
            self.negated_agents[self.current_iteration - 1] += 1

    def new_iteration(self, init: bool = False) -> None:
        """
        Increment the current_iteration counter and then copy all the values from the previous iteration to their respective list indexes for the new iteration.

        :param init: A boolean indicating if the call is being made during the first model iteration (no previous values to copy)
        """
        self.current_iteration += 1

        if init:
            return None

        # Variables defined to reduce repetition below
        t_now: int = self.current_iteration - 1
        t_last: int = self.current_iteration - 2
        # -1 and -2 indexes due to indexing logic for lists...

        # Only these 3 variables must be carried over, all others are calculated at the end of the timestep independently
        self.radicalised_agents[t_now] = self.radicalised_agents[t_last]
        self.silenced_agents[t_now] = self.silenced_agents[t_last]
        self.negated_agents[t_now] = self.negated_agents[t_last]

    def current_layers_repr(self) -> str:
        """
        Extract all the per-hierarchy variables for the current iteration and format it into a substring to be appended to the main iteration output.

        :return: A formatted substring containing all the per-hierarchy variables for the current model iteration.
        """
        output_string: str = (
            "\tHierarchy Name\tLayer Interdependence\tLayer Polarisation\n"
        )
        for hierarchy in self.layer_interdependences.keys():
            interdepence: float = self.layer_interdependences[hierarchy][
                self.current_iteration
            ]
            polarisation: float = self.layers_polarisation[hierarchy][
                self.current_iteration
            ]
            hierarchy_string: str = f"\t{hierarchy}\t{interdepence}\t{polarisation}\n"
            output_string += hierarchy_string
        return output_string

    def current_iteration_repr(self) -> str:
        """
        Extract all variable information for the current iteration and format it into a string to be printed to the terminal.

        :return: A formatted string containing all the variables for the current model iteration.
        """
        formatted_string: str = f"""\n\n==== GATOH model variables at iteration {self.current_iteration}/{self.max_iterations} ====\n\n
            Aggregate community opinion: {self.aggregate_opinions[self.current_iteration]}\n
            Number of radicalised agents in the community: {self.radicalised_agents[self.current_iteration]}\n
            Log odds of radicalisation ocurring: {self.radicalisation_logodds[self.current_iteration]}\n
            Number of opinion silencing events: {self.silenced_agents[self.current_iteration]}\n
            Number of opinion negation events: {self.negated_agents[self.current_iteration]}\n\n
            **** Layer statistics ****\n\n
            """ + self.current_layers_repr()
        return formatted_string


class GATOHLogger:
    """
    The logging module will contain all functions related to logging and/or printing model progress and information
    both during and after simulation.
    """

    # TODO: Expand on logging capabilities

    def __init__(
        self,
        model: Any,
        max_iterations: int,
        hierarchies: list[str],
        verbose: bool = False,
        print_interval: int = 10,
        write_file: bool = True,
    ) -> None:
        """
        :param model: The parent ABModel object that the GATOHLogger object is being attached to.
        :param max_iterations: The maximum number of iterations that the parent model is running its simulation for.
        :param hierarchies: A list containing the names of the social hierarchies present in the parent model.
        :param verbose: a flag to indicate if extended information should be printed during logging.
        :param print_interval: the number of model iterations to run in between each printed logging output.
        :param write_file: a flag to indicate if a log file should be written to disk at the end of logging.
        """
        self.parent_model: Any = model
        self.verbose: bool = verbose
        self.print_interval: int = print_interval
        self.write_file: bool = write_file
        self.variables: LoggerVariables = LoggerVariables(max_iterations, hierarchies)

    def new_iteration(self, init: bool = False) -> None:
        """
        A wrapper that calls LoggerVariables new_iteration().

        :param init: A flag indicating if this function is being called from the first iteration of the model.
        """
        self.variables.new_iteration(init=init)

    def iteration(
        self,
        aggregate_opinion: float,
        radicalisation_logodds: float,
        layer_interdependences: dict[str, float],
        layers_polarisation: dict[str, float],
    ) -> None:
        """
        Store all relevant model variables and states based on the level of logging that has been specified.

        :param aggregate_opinion: The aggregate network opinion that has been observed in the model at the end of this iteration.
        :param radicalisation_logodds: The log odds of an agent being radicalised in the model at the end of this iteration.
        :param layer_interdependences: A <hierarchy : value> dictionary containing the calculated layer interdependency for each hierarchy in the model at the end of this iteration.
        :param layers_polarisation: A <hierarchy : value> dictionary containing the calculated polarisation for each hierarchy in the model at the end of this iteration.
        """
        # TODO: Implement this function
        pass

    def iteration_print(self, current_iteration: int) -> None:
        """
        A method which prints out informative model statistics at the appropriate print_interval.

        :param current_iteration: The current iteration that the model is at when calling this method.
        """
        if current_iteration % self.print_interval != 0:
            return None
        else:
            # TODO: Finish this print block
            pass
        return None
