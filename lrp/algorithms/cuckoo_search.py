"""Cuckoo Search metaheuristic for the Location-Routing Problem.

Implements the nature-inspired optimisation algorithm using Lévy flights,
global and local search operators, and probabilistic solution abandonment.

Reference:
    Yang, X.-S., & Deb, S. (2009). Cuckoo search via Lévy flights.
    World Congress on Nature & Biologically Inspired Computing, 210–214.
"""

from __future__ import annotations

import copy
import math
import random

from lrp.algorithms.nearest_neighbor import build_vehicle_routes
from lrp.config import CuckooConfig
from lrp.models.solution import Solution


class CuckooSearch:
    """Cuckoo Search optimiser for the Location-Routing Problem.

    Maintains a solution population and iteratively improves it using
    Lévy-flight-guided search. Global search moves customers between
    vehicles and depots; local search reorders customers within a route.
    Solutions are occasionally abandoned and replaced with random
    alternatives to escape local optima.

    Attributes:
        config: Hyperparameter configuration.
        threshold: Lévy step value separating global from local search.
    """

    def __init__(self, config: CuckooConfig | None = None) -> None:
        """Initialise the optimiser.

        Args:
            config: Algorithm hyperparameters. Defaults to ``CuckooConfig()``
                when omitted.
        """
        self.config: CuckooConfig = config or CuckooConfig()
        self.threshold: float = self._compute_threshold()

    # ------------------------------------------------------------------
    # Lévy flight helpers
    # ------------------------------------------------------------------

    def _levy_flight_step(self) -> float:
        """Draw a random step size from a Lévy distribution.

        Uses the Mantegna algorithm to approximate Lévy-stable random
        variables via two normally distributed samples.

        Returns:
            A scalar Lévy-distributed step value.
        """
        beta = self.config.levy_beta
        sigma_u = (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        normal_u = random.gauss(0, sigma_u)
        normal_v = random.gauss(0, 1)
        return normal_u / abs(normal_v) ** (1 / beta)

    def _compute_threshold(self) -> float:
        """Estimate the Lévy step threshold by averaging random samples.

        Returns:
            Mean absolute Lévy step over ``levy_averaging_steps`` draws.
        """
        n = self.config.levy_averaging_steps
        return (
            sum(abs(self._levy_flight_step()) for _ in range(n)) / n
        )

    # ------------------------------------------------------------------
    # Search operators
    # ------------------------------------------------------------------

    def _global_search(self, solution: Solution, scaled_step: int) -> None:
        """Transfer customers between vehicles, potentially across depots.

        Implements global exploration by randomly moving customers from one
        vehicle to another. Respects vehicle and depot capacity constraints.

        Args:
            solution: The solution to modify in place.
            scaled_step: Number of transfer attempts to make.
        """
        if len(solution.depots) < 2:
            return
        depot_1 = random.choice(solution.depots)
        depot_2 = random.choice(solution.depots)
        if not depot_1.vehicles or not depot_2.vehicles:
            return
        vehicle_1 = random.choice(depot_1.vehicles)
        vehicle_2 = random.choice(depot_2.vehicles)

        for _ in range(scaled_step):
            if not vehicle_2.customers or not vehicle_1.customers:
                break
            customer = random.choice(vehicle_2.customers)
            fits_vehicle = vehicle_1.capacity >= customer.demand
            fits_depot = depot_1.capacity >= customer.demand
            if fits_vehicle and fits_depot:
                insert_at = random.randrange(len(vehicle_1.customers))
                vehicle_1.customers.insert(insert_at, customer)
                vehicle_1.capacity -= customer.demand
                depot_1.capacity -= customer.demand
                vehicle_2.customers.remove(customer)
                vehicle_2.capacity += customer.demand
                depot_2.capacity += customer.demand

    def _local_search(self, solution: Solution, scaled_step: int) -> None:
        """Swap two customers within a single vehicle route.

        Implements local refinement by reordering visit sequences inside
        randomly chosen routes.

        Args:
            solution: The solution to modify in place.
            scaled_step: Number of swap attempts to make.
        """
        for _ in range(scaled_step):
            depot = random.choice(solution.depots)
            if not depot.vehicles:
                continue
            vehicle = random.choice(depot.vehicles)
            if len(vehicle.customers) < 2:
                continue
            customer_1, customer_2 = random.sample(vehicle.customers, 2)
            idx_1 = vehicle.customers.index(customer_1)
            idx_2 = vehicle.customers.index(customer_2)
            vehicle.customers[idx_1] = customer_2
            vehicle.customers[idx_2] = customer_1

    # ------------------------------------------------------------------
    # Random solution generation
    # ------------------------------------------------------------------

    def _create_random_solution(self, solution: Solution) -> Solution:
        """Randomly reassign all customers to depots, then rebuild routes.

        Guarantees every depot receives at least one customer before the
        remainder are distributed at random to any depot with sufficient
        capacity.

        Args:
            solution: Template solution (not modified; a deep copy is made).

        Returns:
            A new Solution with randomised assignments and rebuilt vehicle
            routes, ready for evaluation.
        """
        candidate = copy.deepcopy(solution)
        customer_pool = list(candidate.customers)

        for depot in candidate.depots:
            depot.assigned_customers = []
            depot.vehicles = []
            depot.capacity = depot.original_capacity
            depot.opened = False

        # Guarantee at least one customer per depot
        for depot in candidate.depots:
            if not customer_pool:
                break
            chosen = random.choice(customer_pool)
            depot.assigned_customers.append(chosen)
            depot.capacity -= chosen.demand
            if not depot.opened:
                depot.opened = True
            customer_pool.remove(chosen)

        # Distribute remaining customers to depots that can accommodate them
        for customer in customer_pool:
            eligible = [
                d for d in candidate.depots if d.capacity >= customer.demand
            ]
            if eligible:
                chosen_depot = random.choice(eligible)
                chosen_depot.assigned_customers.append(customer)
                chosen_depot.capacity -= customer.demand
                if not chosen_depot.opened:
                    chosen_depot.opened = True

        from lrp.config import VEHICLE_CAPACITY
        vc = getattr(candidate, "vehicle_capacity", VEHICLE_CAPACITY)
        for depot in candidate.depots:
            build_vehicle_routes(depot, vc)

        candidate.calculate_total_distance()
        return candidate

    # ------------------------------------------------------------------
    # Core optimisation helpers
    # ------------------------------------------------------------------

    def _recalculate(self, solution: Solution) -> None:
        """Recalculate all vehicle and solution distances from scratch.

        Args:
            solution: The solution to update in place.
        """
        for depot in solution.depots:
            for vehicle in depot.vehicles:
                vehicle.calculate_vehicle_distance()
        solution.calculate_total_distance()

    def _apply_two_opt(self, solution: Solution) -> None:
        """Apply 2-opt improvement to every vehicle route in the solution.

        Args:
            solution: The solution to improve in place.
        """
        from lrp.algorithms.two_opt import two_opt_route

        for depot in solution.depots:
            for vehicle in depot.vehicles:
                two_opt_route(vehicle)
        solution.calculate_total_distance()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def optimize(self, solutions: list[Solution]) -> Solution:
        """Run population-based Cuckoo Search.

        Each iteration:

        1. Pick a random nest *i*, generate a cuckoo via Lévy flight, compare
           against a different random nest *j*; replace *j* if the cuckoo is
           better.
        2. Sort the population by fitness and replace the worst
           ``abandonment_prob`` fraction with randomly generated solutions.
        3. Track the global best across all iterations.

        Args:
            solutions: Initial solution population to optimise.

        Returns:
            The best Solution found across the entire run.

        Raises:
            ValueError: If ``solutions`` is empty.
        """
        if not solutions:
            raise ValueError("At least one solution is required.")

        best = copy.deepcopy(min(solutions, key=lambda s: s.total_distance))

        for _ in range(self.config.num_iterations):
            # --- Phase 1: Lévy flight cuckoo generation ---
            i = random.randrange(len(solutions))
            levy_step = abs(self._levy_flight_step())
            scaled_step = max(1, math.ceil(self.config.step_scale * levy_step))

            cuckoo = copy.deepcopy(solutions[i])
            if levy_step >= self.threshold:
                self._global_search(cuckoo, scaled_step)
            else:
                self._local_search(cuckoo, scaled_step)
            self._recalculate(cuckoo)
            self._apply_two_opt(cuckoo)

            # Compare against a different random nest
            j = random.randrange(len(solutions))
            while j == i and len(solutions) > 1:
                j = random.randrange(len(solutions))
            if cuckoo.total_distance < solutions[j].total_distance:
                solutions[j] = cuckoo

            # --- Phase 2: Abandon worst Pa fraction ---
            num_abandon = max(1, int(self.config.abandonment_prob * len(solutions)))
            solutions.sort(key=lambda s: s.total_distance)
            for k in range(len(solutions) - num_abandon, len(solutions)):
                solutions[k] = self._create_random_solution(solutions[k])

            # --- Phase 3: Update global best ---
            current_best = min(solutions, key=lambda s: s.total_distance)
            if current_best.total_distance < best.total_distance:
                best = copy.deepcopy(current_best)

        print(f"  Cuckoo Search complete. Best cost: {best.total_distance:.4f}")
        return best
