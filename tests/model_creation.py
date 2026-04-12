from __future__ import annotations

import unittest as ut

import src.GATOH.model as md

HIERARCHY_NAMES: list[str] = ["Test_1", "Test_2"]
HIERARCHY_RW_DISTRIB: list[tuple[float, float]] = [(0, 0.01), (0, 0.2)]


class TestModelCreation(ut.TestCase):
    def setUp(self) -> None:
        """
        Initialise a model object with specific parameters for testing purposes.
        """
        self.model = md.ABModel(
            HIERARCHY_NAMES,
            HIERARCHY_RW_DISTRIB,
            iterations=99,
            negation_threshold=0.89,
            radicalisation_threshold=0.45,
        )

    def test_parameter_setting(self):
        self.assertEqual(
            self.model.current_iteration,
            0,
            "Current iteration not being initialised properly",
        )
        self.assertEqual(
            self.model.max_iterations, 99, "Max iterations not being stored correctly"
        )
        self.assertEqual(
            self.model.negation_threshold,
            0.89,
            "Negation threshold not being stored correctly",
        )
        self.assertEqual(
            self.model.radicalisation_threshold,
            0.45,
            "Radicalisation threshold not being stored correctly",
        )
        self.assertEqual(
            self.model.hierarchy_information,
            {
                HIERARCHY_NAMES[0]: HIERARCHY_RW_DISTRIB[0],
                HIERARCHY_NAMES[1]: HIERARCHY_RW_DISTRIB[1],
            },
            "Hierarchy information not being stored correctly as a dictionary",
        )

    def tearDown(self) -> None:
        """
        Reset the model object to run subsequent tests.
        """
        del self.model
