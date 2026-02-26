# LRP Solver

A Python solver for the **Location-Routing Problem (LRP)** using the **Cuckoo Search** metaheuristic. Tested on classical OR benchmark instances.

## Problem Overview

The Location-Routing Problem combines two classical optimization challenges:

1. **Facility Location**: Which depots to open (minimising fixed costs)
2. **Vehicle Routing**: How to route vehicles through assigned customers (minimising travel distance)

**Objective**: Minimise total cost = sum of depot fixed costs + sum of vehicle route distances, subject to:
- Vehicle and depot capacity constraints
- All customer demand must be satisfied
- Each customer assigned to exactly one depot
- Each vehicle route begins and ends at a depot

## Algorithm

This solver uses **Cuckoo Search (CS)**, a nature-inspired metaheuristic that:
- Generates initial solutions via nearest-neighbour heuristic
- Applies Lévy flights for adaptive step-size control
- Uses global search (cross-depot customer transfers) and local search (route reordering)
- Implements probabilistic solution abandonment to escape local optima

**References**:
- Yang, X.-S., & Deb, S. (2009). "Cuckoo search via Lévy flights." *World Congress on Nature & Biologically Inspired Computing*, 210–214.

## Features

- Clean, modular architecture (separate `models/`, `algorithms/`, `io/`, `visualization/`)
- Type annotations and Google-style docstrings throughout
- Efficient memory use (shared distance objects, deepcopy optimization)
- No heavy numerical dependencies (uses `math.dist`, not `numpy`)
- Matplotlib visualisation of depot/customer locations and vehicle routes
- Configurable algorithm parameters via `CuckooConfig` dataclass

## Quick Start

### Installation

Requires Python 3.11+. Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Or with pip:

```bash
pip install -r <(uv pip compile pyproject.toml)
```

### Basic Usage

```bash
uv run python main.py
```

Solves the default benchmark (`Ch69Cli100x10`) with 10 initial solutions and 100 iterations per solution. Displays the best solution found and visualises routes.

### Command-Line Options

```bash
uv run python main.py --help
```

```
options:
  --customers PATH    Path to customer data file (default: DATALRP/DATALRP/Ch69Cli100x10)
  --depots PATH       Path to depot data file (default: DATALRP/DATALRP/Ch69Dep100x10)
  --solutions N       Number of initial solutions (default: 10)
  --iterations N      Cuckoo Search iterations per solution (default: 100)
  --no-plot          Skip route visualisation
```

### Examples

**Quick test (no plot)**:
```bash
uv run python main.py --solutions 5 --iterations 10 --no-plot
```

**Different benchmark**:
```bash
uv run python main.py --customers DATALRP/DATALRP/Gaskell67Cli21x5 \
                      --depots   DATALRP/DATALRP/Gaskell67Dep21x5 \
                      --solutions 10 --iterations 50 --no-plot
```

**Many iterations**:
```bash
uv run python main.py --solutions 20 --iterations 200
```

## Benchmark Datasets

The `DATALRP/DATALRP/` directory contains classical LRP benchmark instances from operations research literature:

| Instance | Customers | Depots | Source |
|---|---|---|---|
| Ch69Cli100x10 | 100 | 10 | Christofides (1969) |
| Daskin95Cli150x10 | 150 | 10 | Daskin (1995) |
| Gaskell67Cli21x5 | 21 | 5 | Gaskell (1967) |
| Min92Cli134x8 | 134 | 8 | Min (1992) |
| Or76Cli117x14 | 117 | 14 | Or (1976) |
| Perl83Cli55x10 | 55 | 10 | Perl (1983) |
| Srivastava86Cli8x2 | 8 | 2 | Srivastava (1986) |

**File format**:
- Customer files: `node_id x_coord y_coord demand`
- Depot files: `node_id x_coord y_coord capacity fixed_cost variable_cost`

All coordinates and distances are Euclidean. Costs are real-valued.

## Algorithm Parameters

Configure via CLI argument or by modifying `CuckooConfig`:

```python
from lrp.config import CuckooConfig

config = CuckooConfig(
    num_solutions=10,           # Population size
    num_iterations=100,         # Iterations per solution
    abandonment_prob=0.25,      # Probability of solution abandonment
    step_scale=0.01,            # Scaling factor for Lévy flight steps
    levy_beta=1.5,              # Lévy distribution shape (0 < beta <= 2)
    levy_averaging_steps=50,    # Samples for threshold estimation
)
```

## Project Structure

```
lrp/
├── __init__.py
├── config.py                    # Constants and algorithm config
├── models/
│   ├── node.py                  # Node, CustomerNode, DepotNode
│   ├── distance.py              # Distance class
│   ├── solution.py              # Solution container
│   └── vehicle_route.py         # VehicleRoute class
├── io/
│   └── data_loader.py           # load_customers(), load_depots()
├── algorithms/
│   ├── nearest_neighbor.py      # Heuristic for initial solutions
│   └── cuckoo_search.py         # CuckooSearch optimiser
└── visualization.py             # plot_routes()

main.py                          # CLI entry point
pyproject.toml                   # Project metadata
PLAN.md                          # Detailed improvement plan
README.md                        # This file
```

## Architecture Highlights

**Single Responsibility Principle**: Each module has one clear purpose.
- Models handle data structures
- `nearest_neighbor.py` handles heuristic construction
- `cuckoo_search.py` handles metaheuristic optimisation
- `data_loader.py` handles file I/O
- `visualization.py` handles plotting

**No Global State**: All algorithm parameters are passed via `CuckooConfig` dataclass.

**Memory Efficiency**:
- Distance objects are shared across solution copies (distances are immutable post-construction)
- Custom `__deepcopy__` in `CustomerNode` prevents exponential memory growth

**Type Safety**: Full type annotations throughout for IDE support and runtime clarity.

**Google Style**: All docstrings follow Google format with `Args`, `Returns`, `Raises` sections.

## Running from PyCharm

1. Open the project in PyCharm
2. Configure interpreter: **Settings → Project → Python Interpreter**
   - Point to `.venv/Scripts/python.exe`
3. Right-click `main.py` → **Run** or press **Shift+F10**

## Development Notes

### Adding a New Benchmark

Place customer and depot files in `DATALRP/DATALRP/` following the standard format, then:

```bash
uv run python main.py --customers DATALRP/DATALRP/YourDatasetCli... \
                      --depots   DATALRP/DATALRP/YourDatasetDep...
```

### Modifying Algorithm Parameters

Edit `CuckooConfig` defaults in `lrp/config.py` or pass `CuckooConfig` to `CuckooSearch`:

```python
from lrp.config import CuckooConfig
from lrp.algorithms.cuckoo_search import CuckooSearch

config = CuckooConfig(num_iterations=500, abandonment_prob=0.3)
searcher = CuckooSearch(config)
best = searcher.optimize(solutions)
```

### Performance Tuning

- **Smaller instances**: Increase `num_iterations` for convergence
- **Larger instances**: Increase `num_solutions` for population diversity
- **Speed**: Reduce `levy_averaging_steps` to lower threshold computation cost

## Output

The solver prints:
1. Data loading summary (customer/depot counts)
2. Solution building progress
3. Optimisation status
4. **Best solution**:
   - Total cost (opened depot fixed costs + vehicle distances)
   - List of active depots
   - Per-depot vehicle routes and distances
5. Route visualisation (Matplotlib window)

## Known Limitations

- Single-vehicle assignment per route (no vehicle splitting within a route)
- Euclidean distance metric only
- No time window constraints
- No heterogeneous fleet (all vehicles have same capacity)
- No load-dependent costs

## License

This project is provided as-is for educational and research purposes.

## Author

Konstantinos Zafeiris

---

For detailed implementation notes and improvement plan, see [PLAN.md](PLAN.md).
