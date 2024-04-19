import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from pypsa_pl_mini.plots import plot_bar, plot_area, plot_line
from pypsa_pl_mini.plots import mpl_style, get_order_and_colors
from pypsa_pl_mini.process_output_network import (
    calculate_statistics,
    calculate_capex,
    calculate_opex,
    calculate_output_capacities,
    calculate_input_capacities,
    calculate_storage_capacities,
    calculate_output_capacity_additions,
    calculate_input_capacity_additions,
    calculate_storage_capacity_additions,
    calculate_flows,
    calculate_curtailed_vres_energy,
    calculate_marginal_prices,
)


def get_label_threshold(ylim, figsize, default=0):
    if ylim is not None:
        return (ylim[1] - ylim[0]) / figsize[1] * 0.27
    else:
        return default


# Annual quantities


def plot_installed_capacities(
    network,
    bus_carrier="electricity",
    capacity_type="generation",
    ylim=None,
    figsize=(5, 8),
):

    if capacity_type == "generation":
        # Output capacity, e.g. electrical capacity of a power plant
        df = calculate_output_capacities(network, bus_carrier=bus_carrier)
    elif capacity_type == "consumption":
        # Input capacity, e.g. electrical capacity of an electrolyser
        df = calculate_input_capacities(network, bus_carrier=bus_carrier)

    df = df.groupby(["year", "carrier"]).agg({"value": "sum"}).reset_index()
    df["value"] = (df["value"] / 1e3).round(2)

    carrier_order, carrier_colors = get_order_and_colors(network)
    fig = plot_bar(
        df,
        title=f"Installed {bus_carrier} {capacity_type} capacity [GW]",
        cat_order=carrier_order,
        cat_colors=carrier_colors,
        label_threshold=get_label_threshold(ylim, figsize, 3.5),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)

    return fig, df


def plot_capacity_additions(
    network,
    bus_carrier="electricity",
    capacity_type="generation",
    ylim=None,
    figsize=(5, 6),
):

    if capacity_type == "generation":
        # Output capacity, e.g. electrical capacity of a power plant
        df = calculate_output_capacity_additions(network, bus_carrier=bus_carrier)
    elif capacity_type == "consumption":
        # Input capacity, e.g. electrical capacity of an electrolyser
        df = calculate_input_capacity_additions(network, bus_carrier=bus_carrier)
    df = df.groupby(["year", "carrier"]).agg({"value": "sum"}).reset_index()
    df["value"] = (df["value"] / 1e3).round(2)

    carrier_order, carrier_colors = get_order_and_colors(network)
    fig = plot_bar(
        df,
        title=f"{bus_carrier.capitalize()} {capacity_type} capacity additions [GW]",
        cat_order=carrier_order,
        cat_colors=carrier_colors,
        label_threshold=get_label_threshold(ylim, figsize, 1),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(
        y=0,
        linestyle="--",
        color=mpl_style["axes.edgecolor"],
        linewidth=mpl_style["axes.linewidth"],
    )
    return fig, df


def plot_storage_capacities(
    network,
    title_carrier="electricity",
    bus_carriers=["battery large electricity", "hydro PSH electricity"],
    ylim=None,
    figsize=(5, 6),
):

    df = calculate_storage_capacities(network, bus_carriers=bus_carriers)

    df = df.groupby(["year", "carrier"]).agg({"value": "sum"}).reset_index()
    df["value"] = (df["value"] / 1e3).round(2)

    carrier_order, carrier_colors = get_order_and_colors(network)
    fig = plot_bar(
        df,
        title=f"Installed {title_carrier} storage capacity [GWh]",
        cat_order=carrier_order,
        cat_colors=carrier_colors,
        label_threshold=get_label_threshold(ylim, figsize, 0.5),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)

    return fig, df


def plot_storage_capacity_additions(
    network,
    title_carrier="electricity",
    bus_carriers=["battery large electricity", "hydro PSH electricity"],
    ylim=None,
    figsize=(5, 4),
):

    df = calculate_storage_capacity_additions(network, bus_carriers=bus_carriers)

    df = df.groupby(["year", "carrier"]).agg({"value": "sum"}).reset_index()
    df["value"] = (df["value"] / 1e3).round(2)

    carrier_order, carrier_colors = get_order_and_colors(network)
    fig = plot_bar(
        df,
        title=f"{title_carrier.capitalize()} storage capacity additions [GWh]",
        cat_order=carrier_order,
        cat_colors=carrier_colors,
        label_threshold=get_label_threshold(ylim, figsize, 1.5),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(
        y=0,
        linestyle="--",
        color=mpl_style["axes.edgecolor"],
        linewidth=mpl_style["axes.linewidth"],
    )
    return fig, df


def plot_annual_generation(
    network, bus_carrier="electricity", ylim=None, figsize=(5, 8)
):
    df = calculate_flows(network, bus_carrier=bus_carrier)
    df = df.groupby(["year", "carrier"]).agg({"value": "sum"}).reset_index()
    df["value"] = df["value"].round(2)

    df = df[~df["carrier"].str.contains("final use")]

    carrier_order, carrier_colors = get_order_and_colors(network)
    fig = plot_bar(
        df,
        title=f"{bus_carrier.capitalize()} generation [TWh]",
        cat_order=carrier_order,
        cat_colors=carrier_colors,
        label_threshold=get_label_threshold(ylim, figsize, 5),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(
        y=0,
        linestyle="--",
        color=mpl_style["axes.edgecolor"],
        linewidth=mpl_style["axes.linewidth"],
    )

    return fig, df


def plot_curtailed_vres_energy(network, ylim=None, figsize=(5, 4)):

    df = calculate_curtailed_vres_energy(network)

    df = df.groupby(["year", "carrier"]).agg({"value": "sum"}).reset_index()
    df["value"] = (df["value"] / 1e6).round(2)

    if df.empty:
        return plt.figure(), df

    carrier_order, carrier_colors = get_order_and_colors(network)
    fig = plot_bar(
        df,
        title="Curtailed vRES energy [TWh]",
        cat_order=carrier_order,
        cat_colors=carrier_colors,
        label_threshold=get_label_threshold(ylim, figsize, 0.5),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(
        y=0,
        linestyle="--",
        color=mpl_style["axes.edgecolor"],
        linewidth=mpl_style["axes.linewidth"],
    )
    return fig, df


def get_fuel_consumption(network):
    fuels = network.carriers[network.carriers.index.str.endswith("supply")].index
    fuels = fuels.str[: -len(" supply")]
    df = pd.concat([calculate_flows(network, bus_carrier=fuel) for fuel in fuels])
    df = df.groupby(["year", "carrier"]).agg({"value": "sum"}).reset_index()
    df = df[df["carrier"].str.endswith("supply")]
    return df


def plot_fuel_consumption(network, ylim=None, figsize=(5, 6)):
    df = get_fuel_consumption(network)
    df["value"] = (df["value"] * 3.6).round(1)

    carrier_order, carrier_colors = get_order_and_colors(network)
    fig = plot_bar(
        df,
        title="Fuel consumption [PJ]",
        cat_order=carrier_order,
        cat_colors=carrier_colors,
        label_threshold=get_label_threshold(ylim, figsize, 30),
        label_digits=0,
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)

    return fig, df


def plot_co2_emissions(network, ylim=None, figsize=(5, 6)):
    df = get_fuel_consumption(network)
    df = df.merge(
        network.carriers["co2_emissions"].dropna(),
        how="inner",
        left_on="carrier",
        right_index=True,
    )
    df["value"] = (df["value"] * df["co2_emissions"]).round(2)
    df = df.drop(columns="co2_emissions")

    carrier_order, carrier_colors = get_order_and_colors(network)
    fig = plot_bar(
        df,
        title="CO₂ emissions [Mt]",
        cat_order=carrier_order,
        cat_colors=carrier_colors,
        label_threshold=get_label_threshold(ylim, figsize, 5),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)

    return fig, df


def plot_opex(
    network, cost_attr="marginal_cost", agg="aggregation", ylim=None, figsize=(5, 8)
):
    title = {
        "marginal_cost": "Total variable costs [bln PLN]",
        "variable_cost": "Variable O&M costs [bln PLN]",
        "co2_cost": "CO₂ costs [bln PLN]",
    }

    df = calculate_opex(network, cost_attr=cost_attr)

    df = df.groupby(["year", agg]).agg({"value": "sum"}).reset_index()

    # Convert to bln PLN
    df["value"] = (df["value"] / 1e9).round(2)

    df = df[df["value"].abs() > 0]

    order, colors = get_order_and_colors(network, agg=agg)
    fig = plot_bar(
        df,
        title=title[cost_attr],
        cat_var=agg,
        cat_order=order,
        cat_colors=colors,
        label_threshold=get_label_threshold(ylim, figsize, 1),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(
        y=0,
        linestyle="--",
        color=mpl_style["axes.edgecolor"],
        linewidth=mpl_style["axes.linewidth"],
    )
    return fig, df


def plot_capex(
    network, cost_attr="capital_cost", agg="aggregation", ylim=None, figsize=(5, 6)
):
    title = {
        "capital_cost": "Total capital costs [bln PLN]",
        "fixed_cost": "Fixed O&M costs [bln PLN]",
        "annual_investment_cost": "Annual investment costs [bln PLN]",
        "investment_cost": "Overnight investment costs [bln PLN]",
    }

    df = calculate_capex(network, cost_attr=cost_attr)
    df = df.groupby(["year", agg]).agg({"value": "sum"}).reset_index()

    # Convert to bln PLN
    df["value"] = (df["value"] / 1e9).round(2)

    df = df[df["value"].abs() > 0]
    if df.empty:
        return plt.figure(), df

    order, colors = get_order_and_colors(network, agg=agg)
    fig = plot_bar(
        df,
        title=title[cost_attr],
        cat_var=agg,
        cat_order=order,
        cat_colors=colors,
        label_threshold=get_label_threshold(ylim, figsize, 2),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(
        y=0,
        linestyle="--",
        color=mpl_style["axes.edgecolor"],
        linewidth=mpl_style["axes.linewidth"],
    )
    return fig, df


def plot_detailed_costs(network, agg="aggregation", ylim=None, figsize=(8, 6)):

    dfs = []

    costs = [
        ("variable_cost", "Var. O&M", calculate_opex),
        ("co2_cost", "CO₂", calculate_opex),
        ("fixed_cost", "Fix. O&M", calculate_capex),
        ("annual_investment_cost", "Ann. invest.", calculate_capex),
    ]

    for cost_attr, label, calculate_cost in costs:
        df = calculate_cost(network, cost_attr=cost_attr)
        df = df.groupby(["year", agg]).agg({"value": "sum"}).reset_index()
        df["cost component"] = label
        dfs.append(df)
    df = pd.concat(dfs)

    df["cost component"] = pd.Categorical(
        df["cost component"], categories=[label for _, label, _ in costs], ordered=True
    )

    df = df[df["year"] == network.meta["year"]].drop(columns="year")

    # Convert to bln PLN
    df["value"] = (df["value"] / 1e9).round(2)

    df = df[df["value"].abs() > 0]

    order, colors = get_order_and_colors(network, agg=agg)
    fig = plot_bar(
        df,
        x_var="cost component",
        title="Annual cost components [bln PLN]",
        cat_var=agg,
        cat_order=order,
        cat_colors=colors,
        label_threshold=get_label_threshold(ylim, figsize, 1),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel(network.meta["year"])
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(
        y=0,
        linestyle="--",
        color=mpl_style["axes.edgecolor"],
        linewidth=mpl_style["axes.linewidth"],
    )
    return fig, df


def plot_average_unit_cost_and_price(network, ylim=None, figsize=(5, 4)):

    # TODO: generalise for a case with separate final use carriers

    df = calculate_statistics(network)
    df = df[
        [
            "year",
            "carrier",
            "Withdrawal",
            "Revenue",
            "Operational Expenditure",
            "Capital Expenditure",
        ]
    ]
    df["Total Cost"] = df["Operational Expenditure"] + df["Capital Expenditure"]

    df_cost = (
        df.groupby(["year"])
        .agg(**{"Avg. unit cost": ("Total Cost", "sum")})
        .reset_index()
    )
    is_final_use = df["carrier"].str.contains("final use")
    df_price_use = (
        df[is_final_use]
        .groupby(["year"])
        .agg(
            **{
                "Avg. price": ("Revenue", lambda x: -np.sum(x)),
                "Final use": ("Withdrawal", "sum"),
            }
        )
        .reset_index()
    )

    df = pd.merge(df_cost, df_price_use, on="year")

    metrics = ["Avg. unit cost", "Avg. price"]

    for col in metrics:
        df[col] = (df[col] / df["Final use"]).round(1)

    df = df.melt(
        id_vars="year", value_vars=metrics, var_name="metric", value_name="value"
    )

    df["metric"] = pd.Categorical(df["metric"], categories=metrics, ordered=True)

    df = df[df["year"] == network.meta["year"]].drop(columns="year")

    df["metric2"] = df["metric"]

    fig = plot_bar(
        df,
        x_var="metric",
        cat_var="metric2",
        title="Average unit cost and price [PLN/MWh]",
        cat_colors={metric: "#535ce3" for metric in metrics},
        show_total_label=False,
        figsize=figsize,
    )

    df = df.drop(columns="metric2")

    ax = fig.axes[0]
    ax.set_xlabel(network.meta["year"])
    if ylim is not None:
        ax.set_ylim(*ylim)

    # Remove legend
    ax.get_legend().remove()

    return fig, df


def plot_total_costs(
    network, costs=["OPEX", "CAPEX"], agg="aggregation", ylim=None, figsize=(5, 8)
):
    df = calculate_statistics(network)
    df = df[["year", agg, "Operational Expenditure", "Capital Expenditure"]]
    df["Total costs"] = 0
    if "OPEX" in costs:
        df["Total costs"] += df["Operational Expenditure"]
    if "CAPEX" in costs:
        df["Total costs"] += df["Capital Expenditure"]

    df = df.groupby(["year", agg]).agg(value=("Total costs", "sum")).reset_index()

    # Convert to bln PLN
    df["value"] = (df["value"] / 1e9).round(2)

    df = df[df["value"].abs() > 0]
    # df = df[df["carrier"] != "electricity final use"]

    order, colors = get_order_and_colors(network, agg=agg)
    fig = plot_bar(
        df,
        title="Total annual costs [bln PLN]",
        cat_var=agg,
        cat_order=order,
        cat_colors=colors,
        label_threshold=get_label_threshold(ylim, figsize, 2),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(
        y=0,
        linestyle="--",
        color=mpl_style["axes.edgecolor"],
        linewidth=mpl_style["axes.linewidth"],
    )
    return fig, df


def plot_net_revenues(network, costs=["OPEX", "CAPEX"], agg="aggregation", ylim=None, figsize=(5, 8)):
    df = calculate_statistics(network)
    df = df[
        ["year", agg, "Revenue", "Operational Expenditure", "Capital Expenditure"]
    ]
    df["Net revenue"] = df["Revenue"]
    if "OPEX" in costs:
        df["Net revenue"] -= df["Operational Expenditure"]
    if "CAPEX" in costs:
        df["Net revenue"] -= df["Capital Expenditure"]

    df = df.groupby(["year", agg]).agg(value=("Net revenue", "sum")).reset_index()

    # Convert to bln PLN
    df["value"] = (df["value"] / 1e9).round(2)

    df = df[df["value"].abs() > 0]
    df = df[df[agg] != "electricity final use"]

    order, colors = get_order_and_colors(network, agg=agg)
    fig = plot_bar(
        df,
        title="Net revenue [bln PLN]",
        cat_var=agg,
        cat_order=order,
        cat_colors=colors,
        label_threshold=get_label_threshold(ylim, figsize, 1),
        figsize=figsize,
    )

    ax = fig.axes[0]
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(
        y=0,
        linestyle="--",
        color=mpl_style["axes.edgecolor"],
        linewidth=mpl_style["axes.linewidth"],
    )
    return fig, df


# Hourly quantities


def plot_hourly_generation(
    network,
    bus_carrier="electricity",
    subperiods=None,
    ylim=None,
    figsize=(8, 5),
):
    df = calculate_flows(network, bus_carrier, annual=False)
    df = df.groupby(level=["carrier"]).sum()
    df = (df / 1e3).round(3)
    df = df.transpose()

    df = df.drop(columns=[col for col in df.columns if "final use" in col])

    df.index = pd.to_datetime(df.index)

    if subperiods is None:
        subperiods = [("", (0, len(df)))]

    carrier_order, carrier_colors = get_order_and_colors(network)

    for subperiod, (i_start, i_stop) in subperiods:
        subdf = df.iloc[i_start:i_stop, :]

        fig = plot_area(
            subdf,
            is_wide=True,
            title=f"{bus_carrier.capitalize()} generation{' – ' if subperiod != '' else ''}{subperiod} [GW]",
            cat_order=carrier_order,
            cat_colors=carrier_colors,
            ylim=ylim,
            figsize=figsize,
        )

        ax = fig.axes[0]
        ax.set_xlabel("")
        # if ylim is not None:
        #     ax.set_ylim(*ylim)
        ax.axhline(
            y=0,
            linestyle="--",
            color=mpl_style["axes.edgecolor"],
            linewidth=mpl_style["axes.linewidth"],
        )

        ticks = subdf.index[subdf.index.hour == 0]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks.strftime("%d.%m\n%H:%M"))
        ax.set_xlabel(None)

    # Returns fig for the last subperiod
    return fig, df


def plot_prices(
    network,
    bus_carrier="electricity",
    subperiods=None,
    ylim=None,
    figsize=(8, 3),
):
    df = calculate_marginal_prices(network, bus_carriers=[bus_carrier]).transpose()
    df.index = pd.to_datetime(df.index)

    if subperiods is None:
        subperiods = [("", (0, len(df)))]

    for subperiod, (i_start, i_stop) in subperiods:
        subdf = df.iloc[i_start:i_stop, :]

        fig = plot_line(
            subdf,
            title=f"Marginal {bus_carrier} cost{' – ' if subperiod != '' else ''}{subperiod} [PLN/MWh]",
            figsize=figsize,
        )

        ax = fig.axes[0]
        ax.set_xlabel("")
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.legend().remove()
        ticks = subdf.index[subdf.index.hour == 0]
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks.strftime("%d.%m\n%H:%M"))
        ax.set_xlabel(None)

    # Returns fig for the last subperiod
    return fig, df
