"""Global constants and algorithm configuration for the LRP solver."""

from dataclasses import dataclass

#: Maximum load a single vehicle can carry (units).
VEHICLE_CAPACITY: int = 160


@dataclass(frozen=True)
class CuckooConfig:
    """Hyperparameters for the Cuckoo Search algorithm.

    Attributes:
        num_solutions: Size of the solution population.
        num_iterations: Number of optimisation iterations per solution.
        abandonment_prob: Probability of abandoning a solution each iteration.
        step_scale: Scaling factor applied to Lévy flight step sizes.
        levy_beta: Shape parameter for the Lévy distribution (0 < beta <= 2).
        levy_averaging_steps: Sample size used to estimate the Lévy threshold.
    """

    num_solutions: int = 10
    num_iterations: int = 100
    abandonment_prob: float = 0.25
    step_scale: float = 0.01
    levy_beta: float = 1.5
    levy_averaging_steps: int = 50
