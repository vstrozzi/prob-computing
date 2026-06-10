# Probabilistic Path Finder

The goal is to model a path finding problem in a way that can be solved with an energy-based model. From our knowledge there's no such a similar formulation, so we explored a way that could benefit from the Library, and indirectly from the HW implemention.

Solving a 2D maze with an energy-based model, sampled by block Gibbs with [THRML](https://pypi.org/project/thrml/) (Extropic's probabilistic-computing simulator).

(**IMPORTANT**: *A star version* (with heuristic on the bias) is available in the maze_solver_astar.ipynb. Not exhaustively tested, but works better thanks to faster convergence to low energy solutions thanks to the euristic on the bias based on the L2 distance from start and end for each node, pushing towards the path state P based on closeness.)

Each maze cell is a 3-state categorical variable (Path / NotPath / Wall). The maze is encoded entirely in the energy: per-node biases pin walls and goals, and a local degree factor rewards valid path geometry, sampled via a 5-color two-hop block scheme.

## Mathematical Formulation & THRML Integration

The maze routing problem is formally mapped into the THRML framework as follows:

1. **State Space Formalism**: The maze is a graph $G = (V, E)$, where each cell $i \in V$ holds a categorical random variable $x_i \in \{P, N, W\}$ (Path, NotPath, Wall).

2. **Target Distribution**: THRML samples from a Boltzmann distribution. The energy $E(x)$ of a state $x$ is defined as $-\beta S(x)$, where $S(x)$ is the score:

   $$S(x) = \sum_{i \in V} b_i[x_i] + \sum_{i \in V} D_i(x_i, x_{\mathcal{N}(i)})$$

3. **THRML Factors**:
   * **Bias Factor ($b_i$)**: A 1D tensor that enforces static boundaries:

```math
     b_i = \begin{cases} [-8,-8,12] & \text{if cell } i \text{ is a Wall} \\ [5,-8,-8] & \text{if } i \in \{\text{Start}, \text{End}\} \\ [0,0,-8] & \text{otherwise} \end{cases}
     $$
```

   * **Degree Factor ($D_i$)**: An N-dimensional tensor evaluating a cell and its neighbors ($\mathcal{N}(i)$) simultaneously. Let $d_i(x) = \sum_{j \in \mathcal{N}(i)} \mathbf{1}[x_j = P]$. The tensor assigns a score:

```math
D_i = \begin{cases} r_P & \text{if } x_i = P \text{ and } d_i = 1 \text{ (on Start/End goal)} \\ 
r_P & \text{if } x_i = P \text{ and } d_i = 2 \text{ (valid path segment)} \\ 
r_N & \text{if } x_i = N \text{ and } d_i \le 1 \text{ (valid empty space)} \\ 
-\lambda & \text{otherwise (penalizes dead-ends or branches)} 
\end{cases}
```


4. **Block Gibbs Execution**: Because the factor $D_i$ evaluates 5 dependent variables simultaneously, they cannot be updated at the same time. The grid is partitioned into 5 independent blocks, computed explicitly by the grouping assignment:

   $$\text{Color}(r, c) = (r + 2c) \pmod 5$$

   This "two-hop" scheme ensures cells within the same factor are never in the same block. THRML freezes 4 blocks and evaluates the multi-dimensional tensor lookups for the active block natively in parallel.

5. **Output**: We select the sampled state with the absolute lowest explicit energy $E(x)$, which traces the valid, minimal penalty route from Start to End.

## Layout

* `maze_solver.ipynb` — modeling the maze route using degree factors.
* `utils/config.py` — every tunable (state encoding, biases, edge matrix, beta, maze size, sampling schedule). All are also keyword overrides on the functions below.
* `utils/wrapper_maze.py` — maze generation (mazelib) and mapping to a THRML-ready graph.
* `utils/model.py` — factors, sampling program, energy computation, solved criterion.
* `utils/visualize.py` — matplotlib helpers.

## Setup

pip install thrml mazelib networkx matplotlib jupyter
jupyter lab maze_solution_comparison.ipynb
