"""Visualisation utilities for LRP solutions."""

from pathlib import Path

import matplotlib.pyplot as plt

from lrp.models.node import CustomerNode, DepotNode


def plot_routes(
    depots: list[DepotNode],
    customers: list[CustomerNode],
    output_path: str | Path = "plot.png",
    show: bool = True,
) -> None:
    """Render depot and customer locations with vehicle routes.

    Creates a two-panel figure: a map of the network with routes drawn on
    the left, and a legend panel on the right. Saves the figure to disk and
    optionally displays it.

    Args:
        depots: Depot nodes to plot (opened and closed).
        customers: Customer nodes to plot.
        output_path: Filesystem path for the saved PNG image.
        show: Whether to call ``plt.show()`` after saving.
    """
    fig = plt.figure(figsize=(12, 7))
    grid = fig.add_gridspec(1, 2, width_ratios=[4, 1])

    ax = fig.add_subplot(grid[0])

    depot_x = [d.x_cord for d in depots]
    depot_y = [d.y_cord for d in depots]
    ax.scatter(depot_x, depot_y, c="red", marker="s", label="Depots", zorder=3)

    customer_x = [c.x_cord for c in customers]
    customer_y = [c.y_cord for c in customers]
    ax.scatter(
        customer_x, customer_y, c="blue", marker="o", label="Customers", zorder=3
    )

    cmap = plt.get_cmap("tab20", max(len(depots), 1))
    for i, depot in enumerate(depots):
        for vehicle in depot.vehicles:
            xs = (
                [depot.x_cord]
                + [c.x_cord for c in vehicle.customers]
                + [depot.x_cord]
            )
            ys = (
                [depot.y_cord]
                + [c.y_cord for c in vehicle.customers]
                + [depot.y_cord]
            )
            ax.plot(
                xs,
                ys,
                color=cmap(i),
                label=f"Route #{vehicle.vehicle_number + 1} — Depot {depot.depot_number}",
            )

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    legend_ax = fig.add_subplot(grid[1])
    legend_ax.axis("off")
    handles, labels = ax.get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center")

    fig.suptitle("Depot and Customer Locations with Vehicle Routes", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()
