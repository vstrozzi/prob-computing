"""THRML sampling program for the maze EBM and analysis helpers.

Energy: E(s) = -beta * ( sum_i b_i[s_i] + sum_i D_i(s_i, s_N(i)) )
realized as a node-bias factor plus optional local degree-shell factors.
"""

from collections import deque
import itertools

import jax
import jax.numpy as jnp
import numpy as np
from thrml import Block, BlockGibbsSpec, FactorSamplingProgram, SamplingSchedule, sample_states
from thrml.models import CategoricalEBMFactor, CategoricalGibbsConditional

from utils import config


def _uses_degree_factor(degree_path_reward, degree_off_reward, degree_penalty):
    return degree_path_reward != 0.0 or degree_off_reward != 0.0 or degree_penalty != 0.0


def _degree_neighborhoods(mg):
    index = {coord: i for i, coord in enumerate(mg.coords)}
    grouped = {}
    for i, (r, c) in enumerate(mg.coords):
        if mg.grid[r, c] == 1 and (r, c) not in (mg.start, mg.end):
            continue
        neighbors = [
            index[nb]
            for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            if nb in index
        ]
        grouped.setdefault(len(neighbors), []).append((i, neighbors))
    return grouped


def _degree_weight_tensor(mg, rows, arity, path_reward, off_reward, penalty):
    weights = np.zeros((len(rows),) + (config.N_STATES,) * (arity + 1), dtype=np.float32)
    for row, (center_i, _) in enumerate(rows):
        coord = mg.coords[center_i]
        is_goal = coord in (mg.start, mg.end)
        target_degree = 1 if is_goal else 2
        for states in itertools.product(range(config.N_STATES), repeat=arity + 1):
            center_state = states[0]
            path_degree = sum(state == config.PATH for state in states[1:])
            if center_state == config.PATH:
                weights[(row,) + states] = path_reward if path_degree == target_degree else -penalty
            elif center_state == config.NOT_PATH and not is_goal:
                weights[(row,) + states] = off_reward if path_degree <= 1 else -penalty
            elif is_goal:
                weights[(row,) + states] = -penalty
    return weights


def _degree_factors(mg, beta, path_reward, off_reward, penalty):
    factors = []
    for arity, rows in _degree_neighborhoods(mg).items():
        centers = [mg.nodes[i] for i, _ in rows]
        neighbor_blocks = [
            Block([mg.nodes[neighbors[k]] for _, neighbors in rows])
            for k in range(arity)
        ]
        weights = beta * _degree_weight_tensor(mg, rows, arity, path_reward, off_reward, penalty)
        factors.append(CategoricalEBMFactor([Block(centers), *neighbor_blocks], jnp.asarray(weights)))
    return factors


def build_program(mg, beta=config.BETA,
                  degree_path_reward=config.DEGREE_PATH_REWARD,
                  degree_off_reward=config.DEGREE_OFF_REWARD,
                  degree_penalty=config.DEGREE_PENALTY):
    """Build the FactorSamplingProgram for a MazeGraph."""
    bias_factor = CategoricalEBMFactor([Block(mg.nodes)], jnp.asarray(beta * mg.biases))

    factors = [bias_factor]
    if _uses_degree_factor(degree_path_reward, degree_off_reward, degree_penalty):
        if len(mg.blocks) < 5:
            raise ValueError(
                "degree factors need maze_to_graph(..., block_scheme='two_hop') "
                "so every center-neighbor shell is split across Gibbs blocks"
            )
        factors.extend(_degree_factors(
            mg, beta, degree_path_reward, degree_off_reward, degree_penalty
        ))

    spec = BlockGibbsSpec(mg.blocks, [])
    samplers = [CategoricalGibbsConditional(config.N_STATES) for _ in mg.blocks]
    return FactorSamplingProgram(spec, samplers, factors, [])


def run_sampling(key, mg, program, schedule=None, n_chains=config.N_CHAINS, init_state=None):
    """Run n_chains parallel Gibbs chains. Returns (n_chains, n_samples, n_nodes) uint8."""
    if schedule is None:
        schedule = SamplingSchedule(
            n_warmup=config.N_WARMUP,
            n_samples=config.N_SAMPLES,
            steps_per_sample=config.STEPS_PER_SAMPLE,
        )

    init = []
    if init_state is None:
        for blk in mg.blocks:
            key, sub = jax.random.split(key)
            init.append(jax.random.randint(sub, (n_chains, len(blk.nodes)), 0, config.N_STATES,
                                           dtype=jnp.uint8))
    else:
        init_state = np.asarray(init_state, dtype=np.uint8)
        node_to_index = {node: i for i, node in enumerate(mg.nodes)}
        if init_state.shape == (len(mg.nodes),):
            for blk in mg.blocks:
                block_state = jnp.asarray([init_state[node_to_index[node]] for node in blk.nodes],
                                          dtype=jnp.uint8)
                init.append(jnp.broadcast_to(block_state, (n_chains, len(blk.nodes))))
        elif init_state.shape == (n_chains, len(mg.nodes)):
            for blk in mg.blocks:
                inds = [node_to_index[node] for node in blk.nodes]
                init.append(jnp.asarray(init_state[:, inds], dtype=jnp.uint8))
        else:
            raise ValueError(
                "init_state must have shape (n_nodes,) or (n_chains, n_nodes)"
            )

    run = jax.jit(jax.vmap(
        lambda i, k: sample_states(k, program, schedule, i, [], [Block(mg.nodes)])
    ))
    key, sub = jax.random.split(key)
    states = run(init, jax.random.split(sub, n_chains))
    return np.asarray(states[0])


def _degree_score(mg, states, path_reward, off_reward, penalty):
    score = np.zeros(states.shape[:-1], dtype=float)
    for rows in _degree_neighborhoods(mg).values():
        for center_i, neighbors in rows:
            center_state = states[..., center_i]
            path_degree = sum(states[..., nb] == config.PATH for nb in neighbors)
            coord = mg.coords[center_i]
            is_goal = coord in (mg.start, mg.end)
            target_degree = 1 if is_goal else 2

            score += np.where(
                center_state == config.PATH,
                np.where(path_degree == target_degree, path_reward, -penalty),
                0.0,
            )
            if not is_goal:
                score += np.where(
                    center_state == config.NOT_PATH,
                    np.where(path_degree <= 1, off_reward, -penalty),
                    0.0,
                )
            if is_goal:
                score += np.where(center_state == config.PATH, 0.0, -penalty)
    return score


def energy_of_states(mg, states, beta=config.BETA,
                     degree_path_reward=config.DEGREE_PATH_REWARD,
                     degree_off_reward=config.DEGREE_OFF_REWARD,
                     degree_penalty=config.DEGREE_PENALTY):
    """Energy of one or many states. states: (..., n_nodes) int -> (...) float."""
    s = np.asarray(states, dtype=int)
    bias_term = mg.biases[np.arange(s.shape[-1]), s].sum(axis=-1)
    degree_term = 0.0
    if _uses_degree_factor(degree_path_reward, degree_off_reward, degree_penalty):
        degree_term = _degree_score(
            mg, s, degree_path_reward, degree_off_reward, degree_penalty
        )
    return -beta * (bias_term + degree_term)


def shortest_path_mask(mg):
    """Boolean grid mask for the shortest open-cell path from start to end."""
    open_cells = mg.grid == 0
    parent = {mg.start: None}
    queue = deque([mg.start])
    while queue:
        r, c = queue.popleft()
        if (r, c) == mg.end:
            break
        for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (0 <= nb[0] < mg.grid.shape[0] and
                    0 <= nb[1] < mg.grid.shape[1] and
                    open_cells[nb] and nb not in parent):
                parent[nb] = (r, c)
                queue.append(nb)

    if mg.end not in parent:
        raise ValueError("maze end is not reachable from start")

    mask = np.zeros(mg.grid.shape, dtype=bool)
    coord = mg.end
    while coord is not None:
        mask[coord] = True
        coord = parent[coord]
    return mask


def is_shortest_path(mg, state):
    """True if Path cells are exactly the shortest open-cell route."""
    return np.array_equal(mg.state_to_grid(state) == config.PATH, shortest_path_mask(mg))


def is_solved(mg, state):
    """True if the Path-labeled cells 4-connect start to end."""
    g = mg.state_to_grid(state)
    path = g == config.PATH
    if not (path[mg.start] and path[mg.end]):
        return False
    n_rows, n_cols = path.shape
    seen = {mg.start}
    queue = deque([mg.start])
    while queue:
        r, c = queue.popleft()
        if (r, c) == mg.end:
            return True
        for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nb[0] < n_rows and 0 <= nb[1] < n_cols and path[nb] and nb not in seen:
                seen.add(nb)
                queue.append(nb)
    return False


def solved_matrix(mg, states):
    """Apply is_solved over (n_chains, n_samples, n_nodes) -> (n_chains, n_samples) bool."""
    states = np.asarray(states)
    return np.array([[is_solved(mg, s) for s in chain] for chain in states])
