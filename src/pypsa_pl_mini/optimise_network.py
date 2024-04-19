import logging

from pypsa_pl_mini.helper_functions import ignore_future_warnings
from pypsa_pl_mini.custom_constraints import (
    define_operational_limit_link,
    define_battery_large_charger_capacity_constraint,
)


solver_options = {
    "highs": {
        "threads": 4,
        "solver": "ipm",
        "run_crossover": "off",
        "small_matrix_value": 1e-7,
        "large_matrix_value": 1e12,
        "primal_feasibility_tolerance": 1e-6,
        "dual_feasibility_tolerance": 1e-6,
        "ipm_optimality_tolerance": 1e-7,
        "parallel": "on",
        "random_seed": 0,
    }
}


def add_epsilon_to_optimal_capacities(network, eps=1e-5):
    # A negligible but non-zero capacity increase to avoid infeasibility in dispatch-only optimisation
    for component in ["Generator", "Link"]:
        is_extendable = network.df(component)["p_nom_extendable"]
        network.df(component).loc[is_extendable, "p_nom_opt"] *= 1 + eps


def reset_capacities(network, params):
    for component, nom_attr in [
        ("Generator", "p_nom"),
        ("Link", "p_nom"),
        ("Store", "e_nom"),
    ]:
        is_to_invest = network.df(component)["technology"].isin(
            params["investment_technologies"]
        )
        is_to_retire = network.df(component)["technology"].isin(
            params["retirement_technologies"]
        )
        network.df(component)[f"{nom_attr}_extendable"] = is_to_invest | is_to_retire

        is_cumulative = network.df(component)["lifetime"] == 1
        is_active = network.df(component)["build_year"] == params["year"]

        is_to_invest &= ~is_cumulative & is_active
        network.df(component).loc[is_to_invest, nom_attr] = network.df(component).loc[
            is_to_invest, f"{nom_attr}_min"
        ]
        is_to_retire &= is_cumulative & is_active
        network.df(component).loc[is_to_retire, nom_attr] = network.df(component).loc[
            is_to_retire, f"{nom_attr}_max"
        ]


@ignore_future_warnings
def create_and_solve_model(network, params):
    network.optimize.create_model()

    define_operational_limit_link(network, network.snapshots)
    define_battery_large_charger_capacity_constraint(network, network.snapshots)

    network.optimize.solve_model(
        solver_name=params["solver"],
        solver_options=solver_options.get(params["solver"], {}),
    )


def optimise_network(network, params):
    create_and_solve_model(network, params)
    if params["reoptimise_with_fixed_capacities"]:
        logging.info("Repeating optimization with optimal capacities fixed...")
        add_epsilon_to_optimal_capacities(network)
        network.optimize.fix_optimal_capacities()
        create_and_solve_model(network, params)
        reset_capacities(network, params)
