"""Pydantic v2 schemas for LRP solution data exchanged between the solver and QA tests.

These models provide a language-model-friendly, serialisable representation of
an LRP solution, decoupled from the internal ``lrp`` domain classes.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator, model_validator


class Route(BaseModel):
    """A single vehicle route from one depot to a sequence of customers.

    Attributes:
        depot_id: Identifier of the originating depot.
        customer_ids: Ordered list of customer identifiers visited on this route.
            Must contain at least one customer.
        stated_distance: Total route distance as reported by the solver, or
            ``None`` when not available.
    """

    depot_id: int
    customer_ids: list[int]
    stated_distance: float | None = None

    @field_validator("customer_ids")
    @classmethod
    def at_least_one_customer(cls, v: list[int]) -> list[int]:
        """Ensure every route visits at least one customer."""
        if len(v) < 1:
            raise ValueError("customer_ids must contain at least one customer.")
        return v


class LRPSolution(BaseModel):
    """A complete LRP solution as returned by a solver or AI agent.

    Attributes:
        routes: All vehicle routes in this solution. Must be non-empty.
        open_depots: Identifiers of depots that are open (have at least one
            vehicle route).
        total_cost: Objective value — sum of depot fixed costs plus total
            route distances.
        reasoning: Optional free-text explanation (useful for AI agent outputs).
    """

    routes: list[Route]
    open_depots: list[int]
    total_cost: float
    reasoning: str = ""

    @field_validator("routes")
    @classmethod
    def at_least_one_route(cls, v: list[Route]) -> list[Route]:
        """Ensure the solution contains at least one vehicle route."""
        if len(v) < 1:
            raise ValueError("routes must contain at least one Route.")
        return v

    @model_validator(mode="after")
    def check_depot_consistency(self) -> "LRPSolution":
        """Ensure open_depots and route depot_ids are consistent."""
        route_depots = {r.depot_id for r in self.routes}
        open_set = set(self.open_depots)

        routes_not_open = route_depots - open_set
        if routes_not_open:
            raise ValueError(
                f"Routes reference depot(s) {sorted(routes_not_open)} "
                f"not listed in open_depots {self.open_depots}."
            )

        open_no_routes = open_set - route_depots
        if open_no_routes:
            raise ValueError(
                f"open_depots lists depot(s) {sorted(open_no_routes)} "
                f"that have no routes assigned."
            )

        return self
