# prob-computing

Solving a 2D maze with an energy-based model, sampled by block Gibbs with
[THRML](https://pypi.org/project/thrml/) (Extropic's probabilistic-computing simulator).

Each maze cell is a 3-state categorical variable (Path / NotPath / Wall). The maze is
encoded entirely in the energy: per-node biases pin walls and goals, and a local degree factor 
rewards valid path geometry, sampled via a 5-color two-hop block scheme.

## Layout

- `maze_solution_comparison.ipynb` — modeling the maze route using degree factors.
- `utils/config.py` — every tunable (state encoding, biases, edge matrix, beta, maze
  size, sampling schedule). All are also keyword overrides on the functions below.
- `utils/wrapper_maze.py` — maze generation (mazelib) and mapping to a THRML-ready graph.
- `utils/model.py` — factors, sampling program, energy computation, solved criterion.
- `utils/visualize.py` — matplotlib helpers.

## Setup

```bash
pip install thrml mazelib networkx matplotlib jupyter
jupyter lab maze_solution_comparison.ipynb
```
