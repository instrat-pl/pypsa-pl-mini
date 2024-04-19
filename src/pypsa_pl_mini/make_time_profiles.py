import pandas as pd
import numpy as np

from pypsa_pl_mini.config import data_dir
from pypsa_pl_mini.mathematical_operations import modify_vres_availability_profile


def make_electricity_final_use_load_profile(df, snapshots, params):
    df_t = pd.read_csv(
        data_dir(
            "input",
            "timeseries",
            f"electricity_demand_profile;year={params['weather_year']}.csv",
        )
    )
    # Create dataframe with profiles for each final use component based on area match
    df_t = df_t.set_index("hour").loc[snapshots]
    df_t = df_t.transpose().reset_index(names="area")
    df_t = (
        df[["name", "area", "p_set_annual"]]
        .merge(df_t, on="area", how="inner")
        .drop(columns="area")
    )
    # Multiply the load profile by p_set_annual
    df_t[snapshots] *= df_t["p_set_annual"].values[:, np.newaxis]
    df_t = df_t.drop(columns="p_set_annual")
    df_t = df_t.set_index("name").transpose()
    return df_t


def make_heat_final_use_load_profile(df, snapshots, params):
    df_t = pd.read_csv(
        data_dir(
            "input",
            "timeseries",
            f"space_heating_demand_profile;year={params['weather_year']}.csv",
        )
    )
    df_t = df_t.set_index("hour").loc[snapshots]
    df_t = params["share_space_heating"] * df_t + (1 - params["share_space_heating"])
    df_t = df_t.transpose().reset_index(names="area")
    df_t = (
        df[["name", "area", "p_set_annual", "p_nom"]]
        .merge(df_t, on="area", how="inner")
        .drop(columns="area")
    )
    # Multiply the load profile by p_set_annual
    df_t[snapshots] *= df_t["p_set_annual"].values[:, np.newaxis]
    # TODO: find ways of dealing with p_nom > p_set_annual preserving utilisation factor
    df_t[snapshots] = np.minimum(df_t[snapshots], df_t["p_nom"].values[:, np.newaxis])
    df_t = df_t.drop(columns=["p_set_annual", "p_nom"])
    df_t = df_t.set_index("name").transpose()
    return df_t


def make_vres_availability_profile(df, snapshots, params):
    dfs_t = {
        carrier: pd.read_csv(
            data_dir(
                "input",
                "timeseries",
                f"availability_profile;carrier={carrier};year={params['weather_year']}.csv",
            )
        )
        for carrier in df["carrier"].unique()
    }
    df_t = pd.concat(
        [
            df_t.set_index("hour")
            .loc[snapshots]
            .transpose()
            .reset_index(names="area")
            .assign(**{"carrier": carrier})
            for carrier, df_t in dfs_t.items()
        ]
    )
    df_t = (
        df[["name", "area", "carrier", "qualifier", "p_max_pu_annual"]]
        .merge(df_t, on=["area", "carrier"], how="inner")
        .drop(columns=["area", "carrier"])
    )

    # Modify availability profiles s.t. they match the assumed annual availability factors
    df_t[snapshots] = modify_vres_availability_profile(
        df_t[snapshots].values.transpose(),
        annual_availability_factor=df_t["p_max_pu_annual"].values,
    ).transpose()

    # For prosumer vRES capacities, reduce the availability by self consumption rate
    is_prosumer = df_t["qualifier"] == "prosumer"
    df_t.loc[is_prosumer, snapshots] *= 1 - params["prosumer_self_consumption"]

    df_t = df_t.drop(columns=["qualifier", "p_max_pu_annual"])
    df_t = df_t.set_index("name").transpose()
    return df_t


def make_constant_load_profile(df, snapshots, params):
    df = df.set_index("name")
    df_t = pd.DataFrame(1, index=snapshots, columns=df.index)
    df_t *= df["p_set"]
    return df_t


make_profile_funcs = {
    "electricity final use load profile": make_electricity_final_use_load_profile,
    "heat final use load profile": make_heat_final_use_load_profile,
    "vres availability profile": make_vres_availability_profile,
    "constant load profile": make_constant_load_profile,
}
