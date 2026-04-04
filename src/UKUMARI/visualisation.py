from __future__ import annotations

from typing import Any

import numpy as np
import rustworkx as rx
from matplotlib import pyplot as plt
from rustworkx.visualization import mpl_draw


class ABVisualiser:
    """
    The visualisations will include:
        - Graph structure plotting
        - Realtime model runtime display
        - Post-runtime summarisation graphs
        - Per-agent lifetime information
        - Per-graph lifetime information
    """

    def __init__(self, model: Any) -> None:
        self.parent_model: Any = model
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
