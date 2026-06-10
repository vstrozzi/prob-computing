"""All tunable parameters of the maze EBM in one place.

State encoding: the canonical category order is (Path, NotPath, Wall),
and bias vectors below are expressed in this same order.
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
BIAS_WALL = np.array([-8.0, -8.0, 12.0])  # wall cells strongly prefer Wall
BIAS_GOAL = np.array([5.0, -8.0, -8.0])   # start and end cells prefer Path
BIAS_OTHER = np.array([0.0, 0.0, -8.0])   # corridor cells reject Wall

# Inverse temperature: scales the whole energy. Higher = sharper distribution.
BETA = 2.0

# --- Degree-shell factor ------------------------------------------------------
DEGREE_PATH_REWARD = 0.5
DEGREE_OFF_REWARD = 0.55
DEGREE_PENALTY = 8.0

# --- Maze ---------------------------------------------------------------------
MAZE_WIDTH = 8    # corridor cells horizontally (grid is 2*W+1 wide)
MAZE_HEIGHT = 8   # corridor cells vertically (grid is 2*H+1 tall)
SEED = 7

# --- Sampling -----------------------------------------------------------------
N_WARMUP = 500         # Gibbs sweeps discarded before recording
N_SAMPLES = 200        # recorded samples per chain
STEPS_PER_SAMPLE = 5   # Gibbs sweeps between recorded samples
N_CHAINS = 32          # independent parallel chains (vmap)
