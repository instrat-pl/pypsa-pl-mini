import pandas as pd
import numpy as np
import pypsa

from pypsa_pl_mini.config import data_dir
from pypsa_pl_mini.make_time_profiles import make_profile_funcs
from pypsa_pl_mini.mathematical_operations import calculate_annuity


def load_and_preprocess_inputs(params, custom_operation=None):

    inputs = {
        name: pd.read_csv(
            data_dir(
                "input",
                f"{name};variant={params[name]}.csv",
            )
        )
        for name in [
            "technology_carrier_definitions",
            "technology_cost_data",
            "installed_capacity",
            "annual_energy_flows",
            "capacity_utilisation",
        ]
    }

    if custom_operation is not None:
        inputs = custom_operation(inputs)

    # Pivot df_tech to wide format
    df_tech = (
        inputs["technology_cost_data"]
        .pivot(
            index=["technology", "technology_year"],
            columns="parameter",
            values="value",
        )
        .reset_index()
    )

    # Get co2_cost from df_tech
    df_co2_cost = df_tech[["technology_year", "co2_cost"]].dropna()
    # Drop co2 emissions technology from df_tech
    df_tech = df_tech[df_tech["technology"] != "co2 emissions"].drop(columns="co2_cost")

    inputs["technology_cost_data"] = df_tech
    inputs["co2_cost"] = df_co2_cost

    # In capacity data, replace the "inf" value with a large number, virtually infinite in the modelling context
    inputs["installed_capacity"]["nom"] = (
        inputs["installed_capacity"]["nom"].replace(np.inf, params["inf"]).astype(float)
    )

    # Split df_flow into df_final use and df_flow (all other flows)
    df_flow = inputs["annual_energy_flows"]
    is_final_use = df_flow["carrier"].str.contains("final use")
    df_final_use = df_flow[is_final_use].copy()
    assert (df_final_use["type"] == "inflow").all()
    assert (df_final_use["parameter"] == "flow").all()

    inputs["annual_energy_flows"] = df_flow[~is_final_use]

    # Pivot df_util and df_final_use to wide format

    df_util = (
        inputs["capacity_utilisation"]
        .pivot(
            index=["area", "technology", "qualifier", "year"],
            columns="parameter",
            values="value",
        )
        .reset_index()
    )
    df_final_use = df_final_use.pivot(
        index=["area", "carrier", "year"],
        columns="parameter",
        values="value",
    ).reset_index()

    # In df_final_use convert flow (in TWh) to p_set_annual (in MW)
    df_final_use["p_set_annual"] = df_final_use["flow"] * 1e6 / 8760
    df_final_use = df_final_use.drop(columns="flow")

    inputs["capacity_utilisation"] = df_util
    inputs["final_use"] = df_final_use

    return inputs


def create_custom_network(params):
    # Get default component attributes
    attrs = pypsa.descriptors.Dict(
        {k: v.copy() for k, v in pypsa.components.component_attrs.items()}
    )

    # Define custom atributes
    # columns: ["type", "unit", "default", "description", "status"]
    for component in ["Bus", "Generator", "Link", "Store"]:
        attrs[component].loc["area"] = [
            "string",
            np.nan,
            np.nan,
            "geographical location of an asset",
            "Input (required)",
        ]
    for component in ["Generator", "Link", "Store"]:
        attrs[component].loc["technology"] = [
            "string",
            np.nan,
            np.nan,
            "specific technology of an asset",
            "Input (required)",
        ]
        attrs[component].loc["qualifier"] = [
            "string",
            np.nan,
            np.nan,
            "extra details about an asset influencing its modelled behaviour",
            "Input (optional)",
        ]
        attrs[component].loc["aggregation"] = [
            "string",
            np.nan,
            np.nan,
            "aggregation category",
            "Input (optional)",
        ]
        attrs[component].loc["variable_cost"] = [
            "static or series",
            "currency/MWh",
            0,
            "variable cost of production, excluding CO2 cost",
            "Input (optional)",
        ]
        attrs[component].loc["co2_cost"] = [
            "static or series",
            "currency/MWh",
            0,
            "CO2 cost component of variable cost of production",
            "Input (optional)",
        ]
        attrs[component].loc["fixed_cost"] = [
            "float",
            "currency/MW",
            0,
            "fixed annual O&M cost of maintaining 1 MW of capacity",
            "Input (optional)",
        ]
        attrs[component].loc["investment_cost"] = [
            "float",
            "currency/MW",
            0,
            "total overnight investment cost of extending capacity by 1 MW",
            "Input (optional)",
        ]
        attrs[component].loc["annual_investment_cost"] = [
            "float",
            "currency/MW",
            0,
            "annualised investment cost of extending capacity by 1 MW",
            "Input (optional)",
        ]
    for component in ["Carrier"]:
        attrs[component].loc["order"] = [
            "int",
            np.nan,
            0,
            "rank of carrier used for ordering in plots",
            "Input (optional)",
        ]
        attrs[component].loc["aggregation"] = [
            "string",
            np.nan,
            np.nan,
            "aggregation category of carrier",
            "Input (optional)",
        ]

    # Create network with custom attributes
    network = pypsa.Network(override_component_attrs=attrs)
    network.name = params["run_name"]
    network.meta = params

    return network


def add_snapshots(network, params):
    # Read electricity demand profile to get snapshots
    df = pd.read_csv(
        data_dir(
            "input",
            "timeseries",
            f"electricity_demand_profile;year={params['weather_year']}.csv",
        ),
        usecols=["hour"],
    )

    network.set_snapshots(df["hour"])

    # We have to increase snapshot weightings to ensure normalization to a year
    network.snapshot_weightings.loc[:, "objective"] = 8760 / len(network.snapshots)
    network.snapshot_weightings.loc[:, "generators"] = 8760 / len(network.snapshots)
    # Weighting cannot be set for stores as it leads to wrong results
    network.snapshot_weightings.loc[:, "stores"] = 1


def add_carriers(network, inputs, params):

    # Merge df_carr with co2_emissions column of df_tech
    # IMPORTANT: co2_emissions within the carrier need to be independent of the specific technology
    df_carr = inputs["technology_carrier_definitions"]
    df_tech = inputs["technology_cost_data"]

    df = df_carr[["carrier", "color", "order", "aggregation", "technology"]].merge(
        df_tech.loc[
            df_tech["technology_year"] == params["year"],
            ["technology", "co2_emissions"],
        ],
        on="technology",
        how="left",
    )

    df = df.drop(columns="technology").groupby("carrier").first()

    network.mremove("Carrier", network.carriers.index)
    network.import_components_from_dataframe(df, "Carrier")


def add_buses(network, inputs, params):

    df_cap = inputs["installed_capacity"]
    df_carr = inputs["technology_carrier_definitions"]
    # List all technology and area combinations present in the capacities
    df = df_cap[["technology", "area"]].drop_duplicates()

    # Combine with bus carriers of techs input and output buses
    bus_carrier_columns = [
        "bus_carrier",
        "input_carrier",
        "output_carrier",
        "output2_carrier",
    ]
    df = df.merge(
        df_carr[["technology", *bus_carrier_columns]], on="technology", how="inner"
    )

    # Identify all unique area and bus carrier combinations
    df = pd.concat(
        [
            df[["area", col]].dropna().rename(columns={col: "carrier"})
            for col in bus_carrier_columns
        ]
    ).drop_duplicates()

    # Define bus names
    df["bus"] = df["area"] + " " + df["carrier"]
    df = df.sort_values("bus")

    network.mremove("Bus", network.buses.index)
    network.import_components_from_dataframe(df.set_index("bus"), "Bus")


def process_capacity_data(inputs, params):

    df_cap = inputs["installed_capacity"]
    df_carr = inputs["technology_carrier_definitions"]
    df_tech = inputs["technology_cost_data"]
    df_co2_cost = inputs["co2_cost"]
    df_util = inputs["capacity_utilisation"]
    df_final_use = inputs["final_use"]

    # (I) Determine carrier, buses, sign, and p_nom/e_nom

    df = df_cap.merge(
        df_carr[
            [
                "technology",
                "carrier",
                "aggregation",
                "bus_carrier",
                "input_carrier",
                "output_carrier",
                "component",
            ]
        ],
        on="technology",
        how="inner",
    )

    # (1) Generators
    is_gen = df["component"] == "Generator"
    is_positive_gen = is_gen & df["input_carrier"].isna()
    is_negative_gen = is_gen & df["output_carrier"].isna()
    df.loc[is_positive_gen, "bus"] = (
        df.loc[is_positive_gen, "area"]
        + " "
        + df.loc[is_positive_gen, "output_carrier"]
    )
    df.loc[is_negative_gen, "bus"] = (
        df.loc[is_negative_gen, "area"] + " " + df.loc[is_negative_gen, "input_carrier"]
    )
    df.loc[is_positive_gen, "sign"] = 1
    df.loc[is_negative_gen, "sign"] = -1
    df.loc[is_gen, "p_nom"] = df.loc[is_gen, "nom"]

    # (2) Links
    is_link = df["component"] == "Link"
    df.loc[is_link, "bus_input"] = (
        df.loc[is_link, "area"] + " " + df.loc[is_link, "input_carrier"]
    )
    df.loc[is_link, "bus_output"] = (
        df.loc[is_link, "area"] + " " + df.loc[is_link, "output_carrier"]
    )
    df.loc[is_link, "p_nom"] = df.loc[is_link, "nom"]

    # (3) Stores
    is_store = df["component"] == "Store"
    df.loc[is_store, "bus"] = (
        df.loc[is_store, "area"] + " " + df.loc[is_store, "bus_carrier"]
    )
    df.loc[is_store, "e_nom"] = df.loc[is_store, "nom"]

    df = df.drop(columns=["bus_carrier", "input_carrier", "output_carrier"])

    # (II) Determine technological and cost parameters

    # Determine technology year
    # (1) If capacity is cumulative and its (virtual) build year can be found in df_tech, use it as technology year
    df_years = df_tech[["technology", "technology_year"]].drop_duplicates()
    df_years["build_year"] = df_years["technology_year"]
    df = df.merge(df_years, on=["technology", "build_year"], how="left")
    df.loc[~df["cumulative"], "technology_year"] = np.nan
    # (2) If not, use the formula: technology_year = 5 * ceil(build_year / 5) - 5
    # e.g. build_year=2020 -> technology_year=2015, build_year=2021 -> technology_year=2020
    df["technology_year"] = (
        df["technology_year"]
        .fillna(5 * (np.ceil(df["build_year"] / 5) - 1))
        .astype(int)
    )

    df = df.merge(df_tech, on=["technology", "technology_year"], how="left")
    df = df.merge(df_co2_cost, on="technology_year", how="left")

    # Calculate marginal cost
    df["variable_cost"] = df["variable_cost"].fillna(0)
    df["co2_cost"] = df["co2_emissions"].fillna(0) * df["co2_cost"]
    df["marginal_cost"] = df["variable_cost"] + df["co2_cost"]

    # Set default efficiency = 1
    df["efficiency"] = (
        1.0 if "efficiency" not in df.columns else df["efficiency"].fillna(1.0)
    )

    # If not provided, determine retire year by technological lifetime
    df["retire_year"] = df["retire_year"].fillna(df["build_year"] + df["lifetime"] - 1)
    # Then calculate the actual lifetime
    df["lifetime"] = df["retire_year"] - df["build_year"] + 1
    # Select capacities based on build and retire year
    df = df[
        (df["build_year"] <= params["year"]) & (df["retire_year"] >= params["year"])
    ]

    # Calculate capital cost
    df["fixed_cost"] = (
        0 if "fixed_cost" not in df.columns else df["fixed_cost"].fillna(0)
    )
    df["investment_cost"] = (
        0 if "investment_cost" not in df.columns else df["investment_cost"].fillna(0)
    )
    # Consider investment costs only for capacities not specified as cumulative
    has_investment_cost = ~df["cumulative"]
    df.loc[~has_investment_cost, "investment_cost"] = 0
    df.loc[~has_investment_cost, "annual_investment_cost"] = 0
    df.loc[has_investment_cost, "annual_investment_cost"] = df.loc[
        has_investment_cost, "investment_cost"
    ] * calculate_annuity(
        lifetime=df.loc[has_investment_cost, "lifetime"],
        discount_rate=params["discount_rate"],
    )

    df["capital_cost"] = df["fixed_cost"] + df["annual_investment_cost"]

    # (III) Determine extendability
    is_to_invest = df["technology"].isin(params["investment_technologies"])
    is_to_retire = df["technology"].isin(params["retirement_technologies"])
    is_extendable = is_to_invest | is_to_retire

    is_gen_or_link = is_gen | is_link
    for is_component, nom in [(is_gen_or_link, "p_nom"), (is_store, "e_nom")]:
        df.loc[is_component, f"{nom}_extendable"] = False
        df.loc[is_component, f"{nom}_min"] = df.loc[is_component, nom]
        df.loc[is_component, f"{nom}_max"] = df.loc[is_component, nom]
        df.loc[is_component & is_extendable, f"{nom}_extendable"] = True
        # For investment-allowed capacities: nom < nom_opt < inf
        df.loc[
            is_component & is_to_invest
            # Capacities specified as cumulative may not be extended
            & ~df["cumulative"]
            # Only capacities with build year equal to simulation year can be extended
            & (df["build_year"] == params["year"]),
            f"{nom}_max",
        ] = np.inf
        # For retirement-allowed capacities: 0 < nom_opt < nom
        df.loc[
            is_component & is_to_retire
            # Only capacities specified as cumulative may be retired
            & df["cumulative"]
            # Only cumulative capacities with build year equal to simulation year can be retired
            & (df["build_year"] == params["year"]),
            f"{nom}_min",
        ] = 0

    # (IV) Incorporate final use assumptions
    # df_final_use might contain the following attributes: p_set_annual

    df_final_use = df_final_use.rename(columns={"year": "build_year"})
    df = df.merge(df_final_use, on=["area", "carrier", "build_year"], how="left")

    # (V) Incorporate capacity utilisation assumptions
    # df_util might contain the following attributes: p_min_pu, p_max_pu, p_set_pu, p_set_pu_annual

    df["qualifier"] = df["qualifier"].fillna("none")
    df_util["qualifier"] = df_util["qualifier"].fillna("none")
    df_util = df_util.rename(columns={"year": "build_year"})

    df = df.merge(
        df_util, on=["area", "technology", "qualifier", "build_year"], how="left"
    )
    df["qualifier"] = df["qualifier"].replace("none", np.nan)

    df["p_min_pu"] = 0.0 if "p_min_pu" not in df.columns else df["p_min_pu"].fillna(0.0)
    df["p_max_pu"] = 1.0 if "p_max_pu" not in df.columns else df["p_max_pu"].fillna(1.0)

    if "p_set_pu" in df.columns:
        p_set = df["p_set_pu"] * df["p_nom"]
        df["p_set"] = df["p_set"].fillna(p_set) if "p_set" in df.columns else p_set
        df = df.drop(columns="p_set_pu")

    if "p_set_pu_annual" in df.columns:
        p_set_annual = df["p_set_pu_annual"] * df["p_nom"]
        df["p_set_annual"] = (
            df["p_set_annual"].fillna(p_set_annual)
            if "p_set_annual" in df.columns
            else p_set_annual
        )
        df = df.drop(columns="p_set_pu_annual")

    df = df.dropna(axis=1, how="all")
    df = df.sort_values("name")

    return df


def add_capacities(network, df_cap, df_attr_t, params):

    for component, df in df_cap.groupby("component"):

        network.mremove(component, network.df(component).index.intersection(df["name"]))

        df_t = df.merge(df_attr_t, on=["carrier", "qualifier"], how="inner")

        df_attrs_t = (
            df_t.groupby(["carrier", "qualifier"])
            .agg({"attribute": lambda x: ",".join(sorted(x.unique()))})
            .rename(columns={"attribute": "attrs_t"})
            .reset_index()
        )

        df = df.merge(df_attrs_t, on=["carrier", "qualifier"], how="left")
        df["attrs_t"] = df["attrs_t"].fillna("")

        for attrs_t, df in df.groupby("attrs_t"):
            attrs_t = attrs_t.split(",")

            df_t = df.merge(df_attr_t, on=["carrier", "qualifier"], how="inner")
            dfs_t = {
                attr: pd.concat(
                    [
                        make_profile_funcs[profile_type](
                            df_t, network.snapshots, params
                        )
                        for profile_type, df_t in df_t.groupby("profile_type")
                    ],
                    axis=1,
                ).sort_index(axis=1)
                for attr, df_t in df_t.groupby("attribute")
            }

            df.loc[df["qualifier"] == "none", "qualifier"] = np.nan
            df = df.set_index("name")

            if component == "Generator":

                network.madd(
                    component,
                    df.index,
                    bus=df["bus"],
                    area=df["area"],
                    carrier=df["carrier"],
                    technology=df["technology"],
                    qualifier=df["qualifier"],
                    aggregation=df["aggregation"],
                    p_nom=df["p_nom"],
                    p_nom_extendable=df["p_nom_extendable"],
                    p_nom_min=df["p_nom_min"],
                    p_nom_max=df["p_nom_max"],
                    sign=df["sign"],
                    efficiency=df["efficiency"],
                    variable_cost=df["variable_cost"],
                    co2_cost=df["co2_cost"],
                    marginal_cost=df["marginal_cost"],
                    fixed_cost=df["fixed_cost"],
                    investment_cost=df["investment_cost"],
                    annual_investment_cost=df["annual_investment_cost"],
                    capital_cost=df["capital_cost"],
                    p_min_pu=(dfs_t if "p_min_pu" in attrs_t else df)["p_min_pu"],
                    p_max_pu=(dfs_t if "p_max_pu" in attrs_t else df)["p_max_pu"],
                    p_set=dfs_t["p_set"] if "p_set" in attrs_t else np.nan,
                    build_year=df["build_year"],
                    lifetime=df["lifetime"],
                )
                network.generators = network.df(component).sort_index()

            elif component == "Link":

                if not params["reverse_links"]:
                    network.madd(
                        component,
                        df.index,
                        bus0=df["bus_input"],
                        bus1=df["bus_output"],
                        area=df["area"],
                        carrier=df["carrier"],
                        technology=df["technology"],
                        qualifier=df["qualifier"],
                        aggregation=df["aggregation"],
                        p_nom=df["p_nom"] / df["efficiency"],
                        p_nom_extendable=df["p_nom_extendable"],
                        p_nom_min=df["p_nom_min"] / df["efficiency"],
                        p_nom_max=df["p_nom_max"] / df["efficiency"],
                        efficiency=df["efficiency"],
                        variable_cost=df["variable_cost"] * df["efficiency"],
                        co2_cost=df["co2_cost"],
                        marginal_cost=df["marginal_cost"] * df["efficiency"],
                        fixed_cost=df["fixed_cost"] * df["efficiency"],
                        investment_cost=df["investment_cost"] * df["efficiency"],
                        annual_investment_cost=df["annual_investment_cost"]
                        * df["efficiency"],
                        capital_cost=df["capital_cost"] * df["efficiency"],
                        p_min_pu=(dfs_t if "p_min_pu" in attrs_t else df)["p_min_pu"],
                        p_max_pu=(dfs_t if "p_max_pu" in attrs_t else df)["p_max_pu"],
                        p_set=dfs_t["p_set"] if "p_set" in attrs_t else np.nan,
                        build_year=df["build_year"],
                        lifetime=df["lifetime"],
                    )
                else:
                    network.madd(
                        component,
                        df.index,
                        bus0=df["bus_output"],
                        bus1=df["bus_input"],
                        area=df["area"],
                        carrier=df["carrier"],
                        technology=df["technology"],
                        qualifier=df["qualifier"],
                        aggregation=df["aggregation"],
                        p_nom=df["p_nom"],
                        p_nom_extendable=df["p_nom_extendable"],
                        p_nom_min=df["p_nom_min"],
                        p_nom_max=df["p_nom_max"],
                        efficiency=1 / df["efficiency"],
                        variable_cost=-df["variable_cost"],
                        co2_cost=-df["co2_cost"],
                        marginal_cost=-df["marginal_cost"],
                        fixed_cost=df["fixed_cost"],
                        investment_cost=df["investment_cost"],
                        annual_investment_cost=df["annual_investment_cost"],
                        capital_cost=df["capital_cost"],
                        p_min_pu=-(dfs_t if "p_max_pu" in attrs_t else df)["p_max_pu"],
                        p_max_pu=-(dfs_t if "p_min_pu" in attrs_t else df)["p_min_pu"],
                        p_set=-dfs_t["p_set"] if "p_set" in attrs_t else np.nan,
                        build_year=df["build_year"],
                        lifetime=df["lifetime"],
                    )
                network.links = network.df(component).sort_index()

            elif component == "Store":

                network.madd(
                    component,
                    df.index,
                    bus=df["bus"],
                    area=df["area"],
                    carrier=df["carrier"],
                    technology=df["technology"],
                    qualifier=df["qualifier"],
                    aggregation=df["aggregation"],
                    e_nom=df["e_nom"],
                    e_nom_extendable=df["e_nom_extendable"],
                    e_nom_min=df["e_nom_min"],
                    e_nom_max=df["e_nom_max"],
                    variable_cost=df["variable_cost"],
                    co2_cost=df["co2_cost"],
                    marginal_cost=df["marginal_cost"],
                    fixed_cost=df["fixed_cost"],
                    investment_cost=df["investment_cost"],
                    annual_investment_cost=df["annual_investment_cost"],
                    capital_cost=df["capital_cost"],
                    build_year=df["build_year"],
                    lifetime=df["lifetime"],
                    e_cyclic=True,
                )
                network.stores = network.df(component).sort_index()


def add_energy_flow_constraints(network, inputs, params):

    df_flow = inputs["annual_energy_flows"]
    df_carr = inputs["technology_carrier_definitions"]

    if params["constrained_energy_flows"] == "none":
        return
    elif params["constrained_energy_flows"] == "all":
        df = df_flow
    else:
        df = df_flow[df_flow["carrier"].isin(params["constrained_energy_flows"])]

    df_carr_component = df_carr[["carrier", "component"]].drop_duplicates()
    # Important: we assume that all technologies within a carrier are represented by a single component type
    assert df_carr_component["carrier"].value_counts().max() == 1
    df = df.merge(df_carr_component, on="carrier", how="left")

    # If links are reverted, only outflow constraints are allowed
    if params["reverse_links"]:
        assert df.loc[df["component"] == "Link", "type"].eq("outflow").all()
    else:
        assert df.loc[df["component"] == "Link", "type"].eq("inflow").all()

    # Convert TWh to MWh
    df["value"] = df["value"] * 1e6

    df["sense"] = df["parameter"].map(
        {
            "max flow": "<=",
            "min flow": ">=",
            "flow": "==",
        }
    )

    df = df[df["year"] == params["year"]].drop(columns="year")

    df = df.set_index("name")

    network.mremove("GlobalConstraint", network.global_constraints.index)
    network.madd(
        "GlobalConstraint",
        df.index,
        type="operational_limit",
        investment_period=(
            np.nan if isinstance(params["year"], int) else params["year"][0]
        ),
        carrier_attribute=df["carrier"],
        sense=df["sense"],
        constant=df["value"],
    )
