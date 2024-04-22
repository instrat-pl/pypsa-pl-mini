import pandas as pd

from pypsa_pl_mini.custom_statistics import make_capex_calculator, make_opex_calculator


def get_attr(attr):
    def getter(n, c):
        df = n.df(c)
        if attr in df:
            values = df[attr].fillna("")
        else:
            values = pd.Series("", index=df.index)
        return values.rename(attr)

    return getter


def get_bus_attr(bus, attr):
    def getter(n, c):
        df = n.df(c)
        if bus in df:
            values = df[bus].map(n.buses[attr]).fillna("")
        else:
            values = pd.Series("", index=df.index)
        return values.rename(f"{bus}_{attr}")

    return getter


def make_custom_groupby(extra_attrs=[], buses=[]):
    attrs = ["area", "aggregation", "carrier", "technology", "qualifier"] + extra_attrs

    def custom_groupby(n, c, **kwargs):
        return [get_attr(attr)(n, c) for attr in attrs] + [
            get_bus_attr(bus, "carrier")(n, c) for bus in buses
        ]

    return custom_groupby


def calculate_statistics(network):
    year = network.meta.get("year", "")

    index = ["component", "area", "aggregation", "carrier", "technology", "qualifier"]
    df = network.statistics(groupby=make_custom_groupby()).reset_index(names=index)

    is_store = df["component"] == "Store"
    df.loc[
        is_store,
        ["Dispatch", "Withdrawal", "Supply", "Operational Expenditure", "Revenue"],
    ] *= network.snapshot_weightings["generators"].values[0]
    # df.loc[is_store, "Capacity Factor"] = np.nan

    df["year"] = year
    df = df.set_index(["year"] + index).reset_index()
    return df


def calculate_opex(network, cost_attr="marginal_cost"):
    year = network.meta.get("year", "")

    index = ["component", "area", "aggregation", "carrier", "technology", "qualifier"]
    df = (
        make_opex_calculator(attr=cost_attr)(network, groupby=make_custom_groupby())
        .rename("value")
        .reset_index()
    )

    is_store = df["component"] == "Store"
    df.loc[is_store, "value"] *= network.snapshot_weightings["generators"].values[0]

    df["year"] = year
    df = df.set_index(["year"] + index).reset_index()
    return df


def calculate_capex(network, cost_attr="capital_cost"):
    year = network.meta.get("year", "")

    index = ["component", "area", "carrier", "technology", "qualifier"]
    df = (
        make_capex_calculator(attr=cost_attr)(network, groupby=make_custom_groupby())
        .rename("value")
        .reset_index()
    )

    df["year"] = year
    df = df.set_index(["year"] + index).reset_index()
    return df


def calculate_output_capacities(network, bus_carrier="electricity", type="final"):
    reverse_links = network.meta.get("reverse_links", False)
    inf = network.meta.get("inf", None)
    year = network.meta.get("year", None)

    if type == "final":
        calculate_capacity = network.statistics.optimal_capacity
    elif type == "initial":
        calculate_capacity = network.statistics.installed_capacity

    df = (
        calculate_capacity(bus_carrier=bus_carrier, groupby=make_custom_groupby())
        .rename("value")
        .reset_index()
    )
    if reverse_links:
        df_link = (
            calculate_capacity(
                comps=["Link"], groupby=make_custom_groupby(buses=["bus0"])
            )
            .rename("value")
            .reset_index()
        )
        df_link = df_link[df_link["bus0_carrier"] == bus_carrier].drop(
            columns=["bus0_carrier"]
        )
        df = pd.concat([df[df["component"] != "Link"], df_link])
    if inf:
        df = df[df["value"] != inf]
    if year:
        df["year"] = year
    return df.reset_index(drop=True)


def calculate_output_capacity_additions(network, bus_carrier="electricity"):
    df_init = calculate_output_capacities(
        network, bus_carrier=bus_carrier, type="initial"
    )
    df_final = calculate_output_capacities(
        network, bus_carrier=bus_carrier, type="final"
    )
    index = [col for col in df_init.columns if col != "value"]
    df = pd.merge(
        df_init, df_final, on=index, suffixes=("_init", "_final"), how="outer"
    ).fillna(0)
    df["value"] = df["value_final"] - df["value_init"]
    return df.drop(columns=["value_init", "value_final"])


def calculate_input_capacities(network, bus_carrier="electricity", type="final"):
    reverse_links = network.meta.get("reverse_links", False)
    inf = network.meta.get("inf", None)
    year = network.meta.get("year", None)

    if type == "final":
        calculate_capacity = network.statistics.optimal_capacity
    elif type == "initial":
        calculate_capacity = network.statistics.installed_capacity

    # Generators
    df_gen = (
        calculate_capacity(
            comps=["Generator"],
            groupby=make_custom_groupby(extra_attrs=["sign"], buses=["bus"]),
        )
        .rename("p_nom")
        .reset_index()
    )
    df_gen = df_gen[(df_gen["sign"] < 0) & (df_gen["bus_carrier"] == bus_carrier)].drop(
        columns=["sign", "bus_carrier"]
    )
    # Links
    if reverse_links:
        df_link = (
            calculate_capacity(
                bus_carrier=bus_carrier, comps=["Link"], groupby=make_custom_groupby()
            )
            .rename("value")
            .reset_index()
        )
    else:
        df_link = (
            calculate_capacity(
                comps=["Link"], groupby=make_custom_groupby(buses=["bus0"])
            )
            .rename("value")
            .reset_index()
        )
        df_link = df_link[df_link["bus0_carrier"] == bus_carrier].drop(
            columns=["bus0_carrier"]
        )
    df = pd.concat([df_gen, df_link])
    if inf:
        df = df[df["value"] != inf]
    if year:
        df["year"] = year
    return df.reset_index(drop=True)


def calculate_input_capacity_additions(network, bus_carrier="electricity"):
    df_init = calculate_input_capacities(
        network, bus_carrier=bus_carrier, type="initial"
    )
    df_final = calculate_input_capacities(
        network, bus_carrier=bus_carrier, type="final"
    )
    index = [col for col in df_init.columns if col != "value"]
    df = pd.merge(df_init, df_final, on=index, suffixes=("_init", "_final"))
    df["value"] = df["value_final"] - df["value_init"]
    return df.drop(columns=["value_init", "value_final"])


def calculate_storage_capacities(network, bus_carriers=None, type="final"):
    year = network.meta.get("year", None)

    if type == "final":
        calculate_capacity = network.statistics.optimal_capacity
    elif type == "initial":
        calculate_capacity = network.statistics.installed_capacity

    df = (
        calculate_capacity(groupby=make_custom_groupby(buses=["bus"]), storage=True)
        .rename("value")
        .reset_index()
    )
    if bus_carriers is not None:
        df = df[df["bus_carrier"].isin(bus_carriers)]
    if year:
        df["year"] = year
    return df.reset_index(drop=True)


def calculate_storage_capacity_additions(network, bus_carriers=None):
    df_init = calculate_storage_capacities(
        network, bus_carriers=bus_carriers, type="initial"
    )
    df_final = calculate_storage_capacities(
        network, bus_carriers=bus_carriers, type="final"
    )
    index = [col for col in df_init.columns if col != "value"]
    df = pd.merge(df_init, df_final, on=index, suffixes=("_init", "_final"))
    df["value"] = df["value_final"] - df["value_init"]
    return df.drop(columns=["value_init", "value_final"])


def calculate_curtailed_vres_energy(network):
    year = network.meta.get("year", None)

    df = (
        network.statistics.curtailment(groupby=make_custom_groupby())
        .rename("value")
        .reset_index()
        .drop(columns="component")
    )
    df = df[df["value"] > 0]

    if year:
        df["year"] = year
        df = df.set_index("year").reset_index()
    return df


def calculate_flows(network, bus_carrier="electricity", annual=True):
    year = network.meta.get("year", "")

    df = network.statistics.energy_balance(
        aggregate_bus=False, aggregate_time=False
    ).reset_index()
    df = df[df["bus_carrier"] == bus_carrier].drop(columns="bus_carrier")
    df = df.merge(
        network.buses["area"], how="left", left_on="bus", right_index=True
    ).drop(columns="bus")
    df["year"] = year

    df = df.merge(
        network.carriers[["aggregation"]],
        left_on="carrier",
        right_index=True,
        how="left",
    )

    df = df.set_index(["year", "component", "area", "carrier", "aggregation"]).sort_index()

    # Provide annual value in TWh
    if annual:
        df *= network.snapshot_weightings["generators"]
        df = (df.sum(axis=1) / 1e6).rename("value").sort_index().reset_index()

    return df


def calculate_marginal_prices(network, bus_carriers=None):
    year = network.meta.get("year", "")

    df = pd.concat(
        [network.buses_t["marginal_price"].T, network.buses[["area", "carrier"]]],
        axis=1,
    )
    df = df.rename(columns={"carrier": "bus_carrier"})
    df = df[df["bus_carrier"].isin(bus_carriers)]
    df["year"] = year
    df = df.set_index(["year", "area", "bus_carrier"])
    return df
