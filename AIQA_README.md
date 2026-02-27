# AI Quality Assurance Framework for LRP Optimization

> A portfolio showcase demonstrating rigorous QA methodology for AI-powered mathematical solvers.

## What This Proves

You can rigorously test, validate, and benchmark LLM outputs on constrained combinatorial
optimization problems using industry-standard evaluation tools — without ever hand-labelling
a single "correct" solution.  The framework catches real constraint violations, hallucinated
entity IDs, and logically inconsistent solver behaviour under input perturbations.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AIQA Framework                               │
│                                                                     │
│   ┌─────────────────┐   ┌─────────────────┐   ┌────────────────┐  │
│   │  Algorithm Core │   │   AI Agent      │   │   QA Suite     │  │
│   │                 │   │                 │   │                │  │
│   │  Cuckoo Search  │   │  LLMSolver      │   │  Deterministic │  │
│   │  metaheuristic  │   │  (Claude API)   │   │  DeepEval      │  │
│   │                 │   │                 │   │  Metamorphic   │  │
│   │  lrp/ package   │   │  ai_agent/      │   │  RAGAS         │  │
│   │                 │   │  solver.py      │   │  Phoenix       │  │
│   └────────┬────────┘   └────────┬────────┘   └───────┬────────┘  │
│            │                     │                     │            │
│            └─────────────────────┴─────────────────────┘            │
│                          qa_suite/common/                            │
│                  (fixtures, schemas, adapters, validators)           │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Run deterministic QA — no API key needed
PYTHONUTF8=1 pytest qa_suite/deepeval_tests/test_deterministic.py -v

# 3. Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 4. Run the full benchmark
python run_benchmark.py Srivastava86

# 5. Generate the report
python -m dashboard.report_generator
cat results/BENCHMARK_REPORT.md

# 6. Run side-by-side comparison of all three tiers
python demo_showcase.py --api-key sk-ant-... --instances Srivastava86 Ch69

# 7. Run the full benchmark with all three LLM tiers
python run_benchmark.py --strategy all Srivastava86

# 8. Run metamorphic robustness tests
PYTHONUTF8=1 pytest qa_suite/metamorphic_tests/test_metamorphic.py -v -s -m "llm and metamorphic"

# 9. View traces in Phoenix UI
python -c "
from observability.phoenix_setup import setup_phoenix
from ai_agent.solver import LLMSolver, SolveStrategy
from qa_suite.common.fixtures import load_instance
tracer = setup_phoenix()
sol, _ = LLMSolver(strategy=SolveStrategy.SELF_HEALING).solve(load_instance('Srivastava86'), tracer=tracer)
print(f'Cost: {sol.total_cost}')
"
# Then open http://localhost:6006
```

## Multi-Tier LLM Strategy

The LLM solver implements **three progressive solving strategies** that demonstrate
how prompt engineering and agentic QA loops reduce constraint violations:

| Tier | Strategy | Approach | Best For |
|------|----------|----------|----------|
| **1** | **Naive** | Zero-shot baseline — minimal instructions | Baseline comparison |
| **2** | **CoT** | Chain-of-Thought + nearest-neighbour heuristic hints | Medium instances (8–55 customers) |
| **3** | **Self-Healing** | CoT + validator feedback loop (max 3 retries) | Robustness testing |

### Example: Srivastava86 (8 customers)
```
Naive      → 4/6 PASS (vehicle capacity, coverage, grounding)
CoT        → 5/6 PASS (adds route distance accuracy)
Self-Heal  → 6/6 PASS (validator feedback fixes remaining violations)
```

### Handling Large Instances (Ch69, 100+ customers)
Recent improvements enable the framework to handle larger instances:
- **max_tokens increased** to 16,384 (from 8,192) — avoids response truncation
- **Adaptive CoT prompts** — for >30 customers, omits verbose leg-by-leg distance calculations
- **Better error display** — demo now shows actual error messages, not bare "ERR"

On Ch69, the Naive tier provides a baseline while CoT and Self-Heal are actively improving.

## QA Layers

| Layer | Tool | What It Tests |
|-------|------|---------------|
| Deterministic | DeepEval `BaseMetric` | Hard constraints: capacity, coverage, depot limits |
| Route distances | Custom validator | Euclidean route-length accuracy (10 % tolerance) |
| Faithfulness | RAGAS / Manual | Data grounding — no hallucinated customer/depot IDs |
| Metamorphic | Custom + pytest | Logical consistency under input perturbation |
| Observability | Arize Phoenix | Per-step OTEL tracing of the LLM solve pipeline |

## Project Structure

```
LRPSolver/
├── lrp/                          # Core solver (Cuckoo Search metaheuristic)
│   ├── algorithms/               #   cuckoo_search.py, nearest_neighbor.py
│   ├── models/                   #   node.py, solution.py, vehicle_route.py
│   └── io/data_loader.py
│
├── ai_agent/                     # LLM-based multi-tier solver
│   ├── solver.py                 #   LLMSolver with SolveStrategy enum
│   │                             #   (naive, cot, self_healing)
│   └── prompt_templates.py       #   Three-tier prompts + adaptive variants
│
├── qa_suite/                     # QA framework
│   ├── common/                   #   fixtures.py, schemas.py, adapters.py
│   ├── deterministic_checks/     #   validators.py (4 constraint validators)
│   ├── deepeval_tests/           #   metrics.py, test_deterministic.py, test_llm_solver.py
│   ├── ragas_tests/              #   test_faithfulness.py
│   ├── metamorphic_tests/        #   perturbations.py, test_metamorphic.py
│   ├── run_deterministic_qa.py   #   standalone Cuckoo Search QA report
│   └── run_comparison.py         #   side-by-side comparison CLI
│
├── observability/
│   └── phoenix_setup.py          # Arize Phoenix / OTEL tracer setup
│
├── dashboard/
│   └── report_generator.py       # Reads results/*.json → BENCHMARK_REPORT.md
│
├── run_benchmark.py              # Master orchestrator (CLI, --strategy flag)
├── demo_showcase.py              # Interactive multi-tier comparison (Rich tables)
├── results/                      # JSON output files + BENCHMARK_REPORT.md
└── DATALRP/DATALRP/              # Benchmark data files (7 instances)
```

## Visualization: The LRP Problem

The **Location-Routing Problem** combines two classical OR problems:
1. **Facility Location** — which depots to open?
2. **Vehicle Routing** — how to route vehicles from open depots to serve all customers?

The objective is to **minimize total cost** = depot fixed costs + route distances.

### Example: Srivastava86 (8 customers, 2 depots)
```
Input:
  Depots: 2 (with fixed opening costs + coordinates)
  Customers: 8 (with x,y coordinates + demand)
  Vehicle capacity: 160

Cuckoo Search Output:
  ✓ Open: [Depot 1, Depot 2]
  ✓ Routes: 3 vehicle routes (all constraints satisfied)
  ✓ Cost: 512.35 (sum of fixed costs + travel distance)

LLM Solver Output:
  Naive:      4/6 ✓  (vehicle capacity, coverage, grounding)
  CoT:        5/6 ✓  (+ route distance accuracy)
  Self-Heal:  6/6 ✓  (+ validator feedback fixes remaining)
```

**Map visualization** (generated via `lrp/visualization.py`):
- Depots shown as **red stars** (depot coordinates)
- Customers shown as **blue dots** (customer coordinates, sized by demand)
- Routes drawn as **colored paths** connecting depot → customers → depot
- Each route respects vehicle capacity constraint and minimizes Euclidean distance

## Benchmark Instances

| Name | Customers | Depots | Source | Difficulty |
|------|-----------|--------|--------|------------|
| Srivastava86 | 8 | 2 | Srivastava (1986) | ⭐ Easy |
| Gaskell67 | 21 | 5 | Gaskell (1967) | ⭐⭐ Medium |
| Perl83 | 55 | 15 | Perl (1983) | ⭐⭐ Medium |
| Ch69 | 100 | 10 | Christofides et al. (1969) | ⭐⭐⭐ Hard |
| Or76 | 117 | 14 | Or (1976) | ⭐⭐⭐ Hard |
| Min92 | 134 | 8 | Min et al. (1992) | ⭐⭐⭐⭐ Very Hard |
| Daskin95 | 150 | 10 | Daskin et al. (1995) | ⭐⭐⭐⭐ Very Hard |

## Example Output

### Demo Showcase (Side-by-Side Comparison)
```bash
$ python demo_showcase.py --instances Srivastava86

    _    ___ ___    _      ____  _
   / \  |_ _/ _ \  / \    / ___|| |__   _____      _____  __ _ ___  ___
  / _ \  | | | | |/ _ \   \___ \| '_ \ / _ \ \ /\ / / __|/ _` / __|/ _ \
 / ___ \ | | |_| / ___ \   ___) | | | | (_) \ V  V / (__| (_| \__ \  __/
/_/   \_\___\__\_\_/   \_\ |____/|_| |_|\___/ \_/\_/ \___|\__,_|___/\___|
```

Output: Rich table with constraint validation across all 4 solvers
```
┌──────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ Check                │ CS           │ Naive        │ CoT          │ Self-Heal    │
├──────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Vehicle Capacity     │ ✓ 100%       │ ✓ 100%       │ ✓ 100%       │ ✓ 100%       │
│ Customer Coverage    │ ✓ 100%       │ ✓ 100%       │ ✓ 100%       │ ✓ 100%       │
│ Depot Capacity       │ ✓ 100%       │ ✓ 100%       │ ✓ 100%       │ ✓ 100%       │
│ Route Distances      │ ✓ 100%       │ ✗ 73%        │ ✓ 100%       │ ✓ 100%       │
│ Total Cost           │ ✓ 100%       │ ✓ 100%       │ ✓ 100%       │ ✓ 100%       │
│ ID Grounding         │ deterministic│ ✓ 100%       │ ✓ 100%       │ ✓ 100%       │
├──────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ Result               │ 5/5 PASS     │ 5/6 FAIL     │ 6/6 PASS     │ 6/6 PASS     │
│                      │ 512.35 2.1s  │ 487.02 1.3s  │ 503.50 1.9s  │ 509.22 2.4s  │
└──────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

### Benchmark JSON Output
```json
{
  "instance": "Srivastava86",
  "n_customers": 8,
  "n_depots": 2,
  "timestamp": "2025-02-27T14:32:00Z",
  "cuckoo_search": {
    "available": true,
    "total_cost": 512.35,
    "n_routes": 3,
    "time_seconds": 2.1,
    "vehicle_capacity_valid": true,
    "customer_coverage_valid": true
  },
  "llm_solvers": {
    "naive": {
      "available": true,
      "strategy": "naive",
      "total_cost": 487.02,
      "n_routes": 3,
      "time_seconds": 1.3,
      "input_tokens": 2145,
      "output_tokens": 312,
      "vehicle_capacity_valid": true,
      "faithfulness_score": 1.0
    },
    "cot": {
      "available": true,
      "strategy": "cot",
      "total_cost": 503.50,
      "n_routes": 3,
      "time_seconds": 1.9,
      "input_tokens": 4821,
      "output_tokens": 487,
      "vehicle_capacity_valid": true,
      "faithfulness_score": 1.0
    }
  }
}
```

## Running Tests

```bash
# Deterministic only (no API key required)
PYTHONUTF8=1 pytest qa_suite/deepeval_tests/test_deterministic.py -v

# All LLM tests (requires ANTHROPIC_API_KEY)
PYTHONUTF8=1 pytest -m llm -v -s --tb=long

# Skip LLM tests
PYTHONUTF8=1 pytest -m "not llm" -v

# Metamorphic tests only
PYTHONUTF8=1 pytest -m "llm and metamorphic" -v -s --tb=long
```

## Visualizing Solutions

The framework can plot solutions as 2D maps:

```python
from lrp.io.data_loader import load_customers, load_depots
from lrp.visualization import plot_routes
from qa_suite.common.schemas import LRPSolution, Route

# After solving, plot the solution
solution = LRPSolution(
    routes=[
        Route(depot_id=1, customer_ids=[3, 5, 7], stated_distance=45.2),
        Route(depot_id=2, customer_ids=[1, 2, 4, 6, 8], stated_distance=67.8),
    ],
    open_depots=[1, 2],
    total_cost=512.35,
    reasoning="..."
)

customers = load_customers("DATALRP/DATALRP/Srivastava86Cli8x2")
depots = load_depots("DATALRP/DATALRP/Srivastava86Dep8x2")

plot_routes(solution, customers, depots, title="LRP Solution: Srivastava86")
# Displays map with:
#   - Red stars: depots (with fixed cost labels)
#   - Blue dots: customers (sized by demand)
#   - Colored paths: routes (one color per route)
```

**What the visualization shows:**
- **Spatial clustering** — how the LLM/metaheuristic groups nearby customers
- **Depot efficiency** — which depots are opened and how many routes they serve
- **Route structure** — whether routes are short and logical or inefficient
- **Demand distribution** — if demand is evenly spread or concentrated

## Recent Improvements (Ch69 & Beyond)

### Token Management
- **max_tokens**: Increased from 8,192 → 16,384 to support 100+ customer instances
- **Adaptive prompts**: CoT tier detects large instances (>30 customers) and omits
  verbose leg-by-leg distance calculations, reducing output token usage by ~40%

### Error Diagnostics
- Demo showcase now displays **actual error messages** (first 80 chars) instead of bare "ERR"
- This enables quick debugging of truncation, parse, and API failures

### Example: Running Ch69 (100 customers, 10 depots)
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python demo_showcase.py --instances Ch69

# Or test all three tiers across multiple instances
python run_benchmark.py --strategy all Srivastava86 Ch69 Perl83
```

The framework now scales gracefully from small (8 customer) to large (150 customer) instances,
with tier-specific optimizations at each level.
