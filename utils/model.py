"""THRML sampling program for the maze EBM and analysis helpers.

Energy: E(s) = -beta * ( sum_i b_i[s_i] + sum_(u,v) W[s_u, s_v] )
realized as two CategoricalEBMFactors (bias and pairwise), sampled by
block Gibbs over the two bipartite blocks.
"""

from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
from thrml import Block, BlockGibbsSpec, FactorSamplingProgram, SamplingSchedule, sample_states
from thrml.models import CategoricalEBMFactor, CategoricalGibbsConditional

from utils import config


def build_program(mg, beta=config.BETA):
    """Build the FactorSamplingProgram for a MazeGraph."""
    bias_factor = CategoricalEBMFactor([Block(mg.nodes)], jnp.asarray(beta * mg.biases))

    u_nodes = [mg.nodes[i] for i in mg.edges[:, 0]]
    v_nodes = [mg.nodes[i] for i in mg.edges[:, 1]]
    edge_w = beta * jnp.broadcast_to(
        jnp.asarray(mg.edge_weights)[None], (len(u_nodes), config.N_STATES, config.N_STATES)
    )
    edge_factor = CategoricalEBMFactor([Block(u_nodes), Block(v_nodes)], edge_w)

    spec = BlockGibbsSpec(mg.blocks, [])
    sampler = CategoricalGibbsConditional(config.N_STATES)
    return FactorSamplingProgram(spec, [sampler, sampler], [bias_factor, edge_factor], [])


def run_sampling(key, mg, program, schedule=None, n_chains=config.N_CHAINS):
    """Run n_chains parallel Gibbs chains. Returns (n_chains, n_samples, n_nodes) uint8."""
    if schedule is None:
        schedule = SamplingSchedule(
            n_warmup=config.N_WARMUP,
            n_samples=config.N_SAMPLES,
            steps_per_sample=config.STEPS_PER_SAMPLE,
        )

    init = []
    for blk in mg.blocks:
        key, sub = jax.random.split(key)
        init.append(jax.random.randint(sub, (n_chains, len(blk.nodes)), 0, config.N_STATES,
                                       dtype=jnp.uint8))

    run = jax.jit(jax.vmap(
        lambda i, k: sample_states(k, program, schedule, i, [], [Block(mg.nodes)])
    ))
    key, sub = jax.random.split(key)
    states = run(init, jax.random.split(sub, n_chains))
    return np.asarray(states[0])


def energy_of_states(mg, states, beta=config.BETA):
    """Energy of one or many states. states: (..., n_nodes) int -> (...) float."""
    s = np.asarray(states, dtype=int)
    bias_term = mg.biases[np.arange(s.shape[-1]), s].sum(axis=-1)
    edge_term = mg.edge_weights[s[..., mg.edges[:, 0]], s[..., mg.edges[:, 1]]].sum(axis=-1)
    return -beta * (bias_term + edge_term)


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
