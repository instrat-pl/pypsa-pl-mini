import pandas as pd


def define_time_dependent_attributes(df_cap, params):

    index = ["carrier", "qualifier", "attribute"]
    df = pd.DataFrame(columns=index + ["profile_type"]).set_index(index)

    df_cap["qualifier"] = df_cap["qualifier"].fillna("none")

    # (1) Electricity final use profiles - p_set
    electricity_final_use_carrier_qualifier = df_cap.loc[
        df_cap["carrier"] == "electricity final use", ["carrier", "qualifier"]
    ].drop_duplicates()
    for carrier, qualifier in electricity_final_use_carrier_qualifier.itertuples(
        index=False
    ):
        df.loc[(carrier, qualifier, "p_set"), :] = [
            "electricity final use load profile",
        ]

    # (2) CHP generation following heat final use load profile - p_set
    if "p_set_annual" in df_cap.columns:
        chp_carrier_qualifier = df_cap.loc[
            df_cap["p_set_annual"].notna() & df_cap["carrier"].str.contains("CHP"),
            ["carrier", "qualifier"],
        ].drop_duplicates()

        for carrier, qualifier in chp_carrier_qualifier.itertuples(index=False):
            df.loc[(carrier, qualifier, "p_set"), :] = [
                "heat final use load profile",
            ]

    # (3) vRES availability profiles - p_max_pu
    vres_carrier_qualifier = df_cap.loc[
        df_cap["carrier"].str.contains(("wind|solar")), ["carrier", "qualifier"]
    ].drop_duplicates()

    for carrier, qualifier in vres_carrier_qualifier.itertuples(index=False):
        df.loc[(carrier, qualifier, "p_max_pu"), :] = [
            "vres availability profile",
        ]

    # (4) Fixed generation or load profiles - p_set
    if "p_set" in df_cap.columns:
        const_generation_carrier_qualifier = df_cap.loc[
            df_cap["p_set"].notna(),
            ["carrier", "qualifier"],
        ].drop_duplicates()
        for carrier, qualifier in const_generation_carrier_qualifier.itertuples(
            index=False
        ):
            df.loc[(carrier, qualifier, "p_set"), :] = [
                "constant load profile",
            ]

    df = df.reset_index()
    return df
