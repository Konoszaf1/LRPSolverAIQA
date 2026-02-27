<div align="center">

# AIQA: Multi-Solver Benchmarking for LLM Mathematical Optimization

### Does a large language model actually solve a vehicle routing problem, or does it just look like it does?

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Anthropic Claude](https://img.shields.io/badge/Anthropic-Claude-D97757?style=for-the-badge&logo=anthropic&logoColor=white)](https://www.anthropic.com/)
[![Pytest](https://img.shields.io/badge/tested_with-pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)](https://pytest.org/)
[![DeepEval](https://img.shields.io/badge/metrics-DeepEval-blue?style=for-the-badge)](https://docs.confident-ai.com/)
[![RAGAS](https://img.shields.io/badge/faithfulness-RAGAS-orange?style=for-the-badge)](https://docs.ragas.io/)
[![uv](https://img.shields.io/badge/package-uv-DE5FE9?style=for-the-badge)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## The Problem

The **Location-Routing Problem (LRP)** is a classical NP-hard optimization problem. Given a set of customers with demands and a set of candidate depots with fixed opening costs, the goal is to decide which depots to open and how to route vehicles from those depots to serve every customer, minimizing total cost subject to hard capacity constraints.

Ask an LLM to solve a 100-customer instance and it returns well-formatted JSON with step-by-step reasoning. On the surface it looks correct. Run automated validation and a different picture emerges: customers are silently dropped, vehicle capacity limits are exceeded by 40%, and stated route distances are numerically fabricated.

<div align="center">

| Check | Cuckoo Search | Naive LLM |
|---|:---:|:---:|
| Valid JSON | Yes | Yes |
| Coherent reasoning | N/A | Yes |
| All customers served | Yes | No |
| Vehicle capacity respected | Yes | No |
| Distances correct | Yes | No |
| Depot capacity respected | Yes | No |

</div>

None of these failures are visible without automated validation. A human reviewer looking at well-structured JSON and plausible cost numbers would likely sign off on it.

![Four-panel solver comparison on Gaskell67: Cuckoo Search routes are clean and non-overlapping; Naive LLM routes cross the map and drop customers; CoT LLM improves structure but overloads one vehicle; Self-Healing LLM matches Cuckoo Search after two repair cycles](docs/images/solver_comparison.png)

*All four panels use the same 21-customer, 5-depot instance (Gaskell67). Red x markers are dropped customers (coverage violation). The thick red halo marks a single route overloaded beyond vehicle capacity. In the Naive LLM panel, Depot 5 at the bottom of the map serves northern customers while Depot 2 at the top serves southern customers, producing visible crossing routes and inflated total cost.*

---

## What This Project Does

This project runs a classical Cuckoo Search metaheuristic and three tiers of LLM-based solver (Anthropic Claude) against the same standard OR benchmark instances, ranging from 8 to 150 customers. Every solution goes through the same validation pipeline regardless of source.

The three LLM tiers are designed to isolate where and why LLMs fail at constrained optimization, and whether structured prompting and validator feedback loops can recover:

| Tier | Strategy | Typical outcome on large instances |
|------|----------|------------------------------------|
| 1 | Naive zero-shot | Customers dropped, capacity breached, distances invented |
| 2 | Chain-of-Thought with nearest-neighbour guidance | Full coverage, but capacity and distance errors persist |
| 3 | Self-healing: CoT + validator feedback loop (max 3 retries) | Near-passing on most checks for instances up to ~100 customers |

**Benchmark results (Ch69, 100 customers):**

| Solver | Validators passed |
|--------|:-----------------:|
| Cuckoo Search | 5 / 5 |
| Naive LLM | 2 / 6 |
| CoT LLM | 3 / 6 |
| Self-Healing LLM | 5 / 6 |

---

## What Happens on a Harder Instance

The Gaskell67 figure above (21 customers) shows self-healing recovering to a passing solution. Scale up to Ch69 (100 customers, 10 depots) and self-healing runs out of rope: after three repair cycles the model still cannot compute correct Euclidean distances, so both the route-distance and total-cost validators remain red.

![Four-panel comparison on Ch69: Cuckoo Search routes are clean; Naive LLM routes shoot across the map with 18 dropped customers; CoT LLM covers all customers but states wrong distances; Self-Healing LLM uses correct routes but the stated distances are still 25% off](docs/images/hard_instance_comparison.png)

*Same layout as above, 5x the customers. The CS panel (top-left) has 14 compact routes across all 10 depots. The Naive LLM panel (top-right) shows 18 red x markers for dropped customers and routes that cross the entire map diagonally. The Self-Healing panel (bottom-right) has correct geometry -- the routes look identical to CS -- but the model's stated distance totals are wrong, which the validators catch.*

---

## Validation Pass Rates by Tier

![Per-validator pass-rate bar chart across Cuckoo Search, Naive LLM, CoT LLM, and Self-Healing LLM on the Ch69 100-customer benchmark instance](docs/images/tier_progression.png)

Each bar shows the pass rate for one validator on one solver tier. The Naive LLM fails route distance and total cost checks entirely on 100-customer instances. The Self-Healing tier closes most of the gap by feeding the exact violation messages back to the model as a repair prompt, but arithmetic errors survive the loop.

**Data flow:** A benchmark instance feeds into all four solvers in parallel. Every resulting solution then passes through the same AIQA pipeline: 5 deterministic validators, a faithfulness and ID-grounding check, metamorphic perturbation tests, DeepEval metric wrappers, and a dashboard report. The Self-Healing solver is the only one that reads validator output before it is finalized.

---

## The Multi-Tier LLM Approach

### Tier 1: Naive (Zero-Shot)

Minimal prompt: provide the instance data and ask for a JSON solution. No routing hints, no capacity reminders. This tier shows what an LLM does by default. On small instances (8 customers) it often passes. On larger ones it consistently drops customers and fabricates distances.

### Tier 2: Chain-of-Thought with Heuristic Guidance

The prompt walks the LLM through a nearest-neighbour assignment strategy and asks it to track remaining capacity at each step. Route structure improves, coverage errors mostly disappear, but distance calculations remain wrong. The model narrates correct-sounding arithmetic that does not match Euclidean geometry.

### Tier 3: Self-Healing Agent

After the initial CoT attempt, the solution is validated. If any of the 5 checks fail, a repair prompt is built that includes the exact violations as text, and the model is asked to fix them. This loop runs up to 3 times. The approach treats the validator as a runtime feedback signal rather than a post-hoc test, which is how LLMs can be used safely in optimization pipelines.

---

## The AIQA Validation Suite

Every solution, from any solver, is evaluated by the same five deterministic checks plus two additional layers:

| Layer | What it checks | What it catches |
|-------|---------------|-----------------|
| Vehicle capacity | Sum of demand on each route vs. vehicle limit | Routes that exceed per-vehicle load |
| Customer coverage | Every customer ID in the instance appears in exactly one route | Dropped or duplicated customers |
| Depot capacity | Aggregate demand routed through each depot vs. depot limit | Depot overloads |
| Route distances | Recomputed Euclidean distance vs. stated distance (5% tolerance) | Fabricated distance values |
| Total cost | Recomputed depot fixed costs + route distances vs. stated total | Wrong objective value |
| ID grounding / faithfulness | Every ID in the solution exists in the input; RAGAS scoring | Phantom customers and depots |
| Metamorphic tests | Perturb the instance (scale demands, remove customers, jitter coordinates) and check that solution quality changes in the expected direction | Logical inconsistencies that pass individual checks |

The DeepEval layer wraps these checks as `BaseMetric` objects so the full suite runs in pytest with CI-compatible pass/fail output.

---

## Quickstart

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Anthropic API key (for LLM tiers only)

### Install

```bash
git clone https://github.com/yourusername/LRPSolver.git && cd LRPSolver
uv sync
```

### Interactive demo

Runs all four solvers on two random instances and prints a side-by-side validation table:

```bash
# PowerShell
$env:ANTHROPIC_API_KEY="sk-ant-..."
uv run python demo_showcase.py

# or pass the key directly
uv run python demo_showcase.py --api-key sk-ant-...

# specific instances
uv run python demo_showcase.py --instances Gaskell67 Ch69
```

### Full benchmark

```bash
# single instance (default: Srivastava86)
uv run python run_benchmark.py

# all instances, all three LLM tiers
uv run python run_benchmark.py --all --strategy all

# self-healing tier only
uv run python run_benchmark.py --strategy self_healing
```

### Tests

```bash
# deterministic checks, no API key required
uv run pytest qa_suite/deepeval_tests/test_deterministic.py -v

# LLM tests, all three tiers (requires ANTHROPIC_API_KEY)
uv run pytest -m llm -v -s

# regenerate README images
uv run python generate_readme_images.py

# dashboard report
uv run python -m dashboard.report_generator
```

---

## Benchmark Datasets

Classical OR instances in `DATALRP/DATALRP/`:

| Instance | Customers | Depots | Source |
|----------|:---------:|:------:|--------|
| Srivastava86 | 8 | 2 | Srivastava (1986) |
| Gaskell67 | 21 | 5 | Gaskell (1967) |
| Perl83 | 55 | 15 | Perl (1983) |
| Ch69 | 100 | 10 | Christofides (1969) |
| Or76 | 117 | 14 | Or (1976) |
| Min92 | 134 | 8 | Min (1992) |
| Daskin95 | 150 | 10 | Daskin (1995) |

---

## Project Structure

```
lrp/                              # Core LRP solver package
  config.py                       #   Vehicle capacity, CuckooConfig dataclass
  models/                         #   Node, Distance, Solution, VehicleRoute
  io/data_loader.py               #   Benchmark file parsers
  algorithms/                     #   Cuckoo Search, nearest neighbor, 2-opt
  visualization.py                #   Route plotting (matplotlib)

ai_agent/                         # Multi-tier LLM solver
  solver.py                       #   LLMSolver + SolveStrategy enum + self-healing loop
  prompt_templates.py             #   3 prompt tiers: naive, CoT, repair

qa_suite/                         # AIQA validation framework
  common/                         #   Shared fixtures, schemas, adapters, faithfulness
  deterministic_checks/           #   5 validators (capacity, coverage, distance, cost, depot)
  deepeval_tests/                 #   DeepEval BaseMetric wrappers + pytest integration
  metamorphic_tests/              #   Perturbation functions + metamorphic test suite
  ragas_tests/                    #   RAGAS faithfulness evaluation

observability/                    # Arize Phoenix OTEL tracing setup
dashboard/                        # Benchmark report generator
generate_readme_images.py         # Regenerates docs/images/ from real solver output
demo_showcase.py                  # 4-solver Rich terminal UI demo
run_benchmark.py                  # Master benchmark CLI
```

---

## Tech Stack

| | Technology |
|---|---|
| Runtime | Python 3.11+ with uv |
| LLM | Anthropic Claude (Sonnet / Haiku) |
| Solution schemas | Pydantic v2 with cross-field validation |
| QA metrics | DeepEval BaseMetric wrappers |
| Faithfulness | RAGAS + manual ID-grounding checks |
| Observability | Arize Phoenix (OTEL traces) |
| Terminal UI | Rich |
| Retry | Tenacity with exponential backoff |
| Plotting | Matplotlib |
| Code quality | Ruff + mypy |

---

## The Cuckoo Search Solver

Cuckoo Search is a nature-inspired metaheuristic. The implementation here builds an initial population of solutions via a nearest-neighbour heuristic, then applies Levy flights for adaptive step-size control during optimization. Global moves transfer customers between depots; local moves reorder routes with 2-opt. Solutions are probabilistically abandoned to escape local optima. The algorithm always produces feasible solutions and serves as the ground truth for comparison.

---

## Author

**Konstantinos Zafeiris**
