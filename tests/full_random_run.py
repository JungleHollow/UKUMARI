from __future__ import annotations

import random as rd
from copy import deepcopy

import rustworkx as rx

import src.GATOH.agents as agt
import src.GATOH.graphs as gr
import src.GATOH.model as md

# rd.seed(1312)
N_INDIVIDUALS: int = 80
AGENT_ID_BASE: str = "TEST"  # Define the Id base for the test case

HIERARCHY_NAMES: list[str] = [
    "close_hierarchy",
    "hierarchy_a",
    "hierarchy_b",
    "decaying_hierarchy",
]

HIERARCHY_RW_DISTRIB: list[tuple[float, float]] = [
    (0, 0.01),  # Close hierarchy
    (0, 0.1),
    (0, 0.2),
    (-0.05, 0.02),  # "decaying" hierarchy
]

AGENT_PERSONALITIES: dict[str, float] = {
    # Basic case using only 3 of the available personality types
    "neutral": 0.6,
    "social": 0.2,
    "rational": 0.2,
}

if __name__ == "__main__":
    model = md.ABModel(HIERARCHY_NAMES, HIERARCHY_RW_DISTRIB, 20)
    model.generate_agents(AGENT_ID_BASE, AGENT_PERSONALITIES, number=N_INDIVIDUALS)
    model.generate_graphs(
        HIERARCHY_NAMES, model.agents, agent_subsetting=True
    )  # Set agent_subsetting to True to have significant differences between hierarchies in this test
    model.iterate()
