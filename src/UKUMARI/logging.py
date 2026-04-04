from __future__ import annotations

from typing import Any


class UKUMARILogger:
    """
    The logging module will contain all functions related to logging and/or printing model progress and information
    both during and after simulation
    """

    def __init__(
        self,
        model: Any,
        verbose: bool = False,
        print_interval: int = 10,
        write_file: bool = True,
    ) -> None:
        """
        :param model: The parent ABModel object that the ABMOSLogger object is being attached to
        :param verbose: a flag to indicate if extended information should be printed during logging
        :param print_interval: the number of model iterations to run in between each printed logging output
        :param write_file: a flag to indicate if a log file should be written to disk at the end of logging
        """
        self.parent_model: Any = model
        self.verbose: bool = verbose
        self.print_interval: int = print_interval
        self.write_file: bool = write_file

    def iteration_print(self, current_iteration: int) -> None:
        """
        A method which prints out informative model statistics at the appropriate print_interval

        :param current_iteration: The current iteration that the model is at when calling this method
        """
        if current_iteration % self.print_interval != 0:
            return None
        else:
            # TODO: Finish this print block
            pass
        return None
