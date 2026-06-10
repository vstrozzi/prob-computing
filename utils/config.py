"""All tunable parameters of the maze EBM in one place.

State encoding: the canonical category order is (Path, NotPath, Wall),
matching the row/col order of EDGE_WEIGHTS. Bias vectors below are
expressed in this same order.
"""

import numpy as np

# --- State encoding ---------------------------------------------------------
PATH = 0
NOT_PATH = 1
WALL = 2
N_STATES = 3
STATE_NAMES = ("Path", "NotPath", "Wall")

# --- Node biases (order: Path, NotPath, Wall) -------------------------------
# Positive entries are favorable: node energy is -bias[state].
BIAS_WALL = np.array([0.0, 0.0, 5.0])    # wall cells prefer Wall
BIAS_GOAL = np.array([5.0, 0.0, 0.0])    # start and end cells prefer Path
BIAS_OTHER = np.array([0.5, 0.5, 0.0])   # corridor cells: Path/NotPath equally

# --- Edge function (rows/cols: Path, NotPath, Wall) --------------------------
# Applied once per edge (u, v) with u before v in row-major grid scan order
# (left-to-right, top-to-bottom). Edge energy is -EDGE_WEIGHTS[s_u, s_v].
EDGE_WEIGHTS = np.array([
    [1.0, 0.5, -1.0],
    [1.0, 0.6, -1.0],
    [0.0, 0.0,  0.0],
])

# Inverse temperature: scales the whole energy. Higher = sharper distribution.
BETA = 1.0

# --- Maze ---------------------------------------------------------------------
MAZE_WIDTH = 8    # corridor cells horizontally (grid is 2*W+1 wide)
MAZE_HEIGHT = 8   # corridor cells vertically (grid is 2*H+1 tall)
SEED = 7

# --- Sampling -----------------------------------------------------------------
N_WARMUP = 0           # Gibbs sweeps discarded before recording
N_SAMPLES = 200        # recorded samples per chain
STEPS_PER_SAMPLE = 1   # Gibbs sweeps between recorded samples
N_CHAINS = 32          # independent parallel chains (vmap)
