from __future__ import annotations

import random as rd
from copy import deepcopy

import rustworkx as rx

import src.UKUMARI.agents as agt
import src.UKUMARI.graphs as gr
import src.UKUMARI.model as md

# rd.seed(1234)
N_INDIVIDUALS: int = 10

if __name__ == "__main__":
    individuals: list = []
    for i in range(N_INDIVIDUALS):
        tmp_weightings = {
            "base_graph": 0.5,
            "hierarchy_0": rd.random(),
            "hierarchy_1": rd.random(),
            "hierarchy_2": rd.random(),
            "hierarchy_3": rd.random(),
        }
        tmp_agent = agt.Agent(
            id=f"{i}", social_weightings=tmp_weightings, opinion=rd.uniform(-1, 1)
        )
        individuals.append(tmp_agent)

    layers: dict = {}

    # base_graph: rx.PyDiGraph = rx.PyDiGraph()
    # base_graph.add_nodes_from(individuals)
    # layers["base"] = base_graph

    base_graph: gr.Graph = gr.Graph("base_graph")
    base_graph.add_nodes(individuals)
    layers["base_graph"] = base_graph

    graph_density: list[float] = [0.3, 0.7, 0.9, 0.45]

    for j in range(4):
        tmp_graph: gr.Graph = deepcopy(base_graph)

        for k in range(int((N_INDIVIDUALS**2) * graph_density[j])):
            m: int = int(rd.choice(individuals).id)
            n: int = int(rd.choice(individuals).id)
            if m != n:
                tmp_rel: float = rd.random()
                tmp_edge: dict = {
                    "from_node": [m],
                    "to_node": [n],
                    "weighting": [tmp_rel],
                }
                tmp_graph.add_edges(tmp_edge)

        layers[f"hierarchy_{j}"] = tmp_graph

    model = md.ABModel(100)
    model.add_graphs(list(layers.values()), list(layers.keys()))
    model.add_agents(individuals)

    for agent in model.agents.agents:
        print(agent.opinion)

    model.iterate()
    for agent in model.agents.agents:
        print(agent.opinion)
