"""Matplotlib helpers for mazes, sampled states, and sampling diagnostics."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from utils import config

# Cell colors in state order (Path, NotPath, Wall).
STATE_CMAP = ListedColormap(["#e76f51", "#f4f1de", "#264653"])
WALL_COLOR = "#264653"


def _mark_endpoints(ax, start, end):
    ax.scatter([start[1]], [start[0]], marker="o", s=80, c="#2a9d8f", label="start", zorder=3)
    ax.scatter([end[1]], [end[0]], marker="*", s=140, c="#e9c46a", label="end", zorder=3)
    ax.legend(loc="upper right", fontsize=8)


def plot_maze(grid, start, end, ax=None, title="Maze"):
    """Plot the raw 0/1 maze grid with start/end markers."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid, cmap=ListedColormap(["#f4f1de", WALL_COLOR]), vmin=0, vmax=1)
    _mark_endpoints(ax, start, end)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_state(mg, state, ax=None, title="Sampled state"):
    """Plot one sampled per-node state on the maze grid."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mg.state_to_grid(state), cmap=STATE_CMAP, vmin=0, vmax=config.N_STATES - 1)
    _mark_endpoints(ax, mg.start, mg.end)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_path_marginal(mg, states, ax=None, title="P(Path) marginal"):
    """Heatmap of the empirical P(state = Path) per cell, walls drawn solid."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))
    states = np.asarray(states).reshape(-1, len(mg.nodes))
    p_path = (states == config.PATH).mean(axis=0).reshape(mg.grid.shape)
    p_path = np.ma.masked_where(mg.grid == 1, p_path)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(WALL_COLOR)
    im = ax.imshow(p_path, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)
    _mark_endpoints(ax, mg.start, mg.end)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_energy_trace(energies, ax=None, title="Energy per Gibbs sample"):
    """Per-chain energy traces (thin) and their mean (bold). energies: (n_chains, n_samples)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    energies = np.atleast_2d(energies)
    for e in energies:
        ax.plot(e, color="gray", alpha=0.3, lw=0.8)
    ax.plot(energies.mean(axis=0), color="#e76f51", lw=2, label="mean over chains")
    ax.set_xlabel("sample index")
    ax.set_ylabel("energy")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_sampling_evolution(mg, chain_states, sample_steps, figsize=(16, 4)):
    """Snapshots of one chain at the given sample indices. chain_states: (n_samples, n_nodes)."""
    fig, axes = plt.subplots(1, len(sample_steps), figsize=figsize)
    for ax, t in zip(np.atleast_1d(axes), sample_steps):
        ax.imshow(mg.state_to_grid(chain_states[t]), cmap=STATE_CMAP,
                  vmin=0, vmax=config.N_STATES - 1)
        ax.set_title(f"sample {t}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Sampling evolution (one chain)")
    return fig


def plot_solved_fraction(solved, ax=None, title="Fraction of chains solved"):
    """Fraction of chains whose Path cells connect start to end, per sample step.

    solved: (n_chains, n_samples) bool from model.solved_matrix.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    solved = np.atleast_2d(solved)
    ax.plot(solved.mean(axis=0), color="#2a9d8f", lw=2)
    ax.set_xlabel("sample index")
    ax.set_ylabel("fraction solved")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    return ax
