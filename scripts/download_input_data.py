import logging
import gspread
import pandas as pd

from pypsa_pl_mini.config import data_dir


def gsheet_to_df(url, sheet_name):
    gc = gspread.oauth()
    gs = gc.open_by_url(url)
    logging.info(f'Downloading sheet {sheet_name} from "{gs.title}"')
    ws = gs.worksheet(sheet_name)
    df = pd.DataFrame(ws.get_all_records())
    df = df.replace("", pd.NA)
    return df


def ignore_commented_rows_columns(df):
    df = df.loc[
        ~df.iloc[:, 0].fillna(" ").str.startswith("#"), ~df.columns.str.startswith("#")
    ]
    return df


def download_input_data():
    for name, variant, url in [
        (
            "technology_carrier_definitions",
            "mini",
            "https://docs.google.com/spreadsheets/d/1oM4T3LirR-XGO1fQ_KhiuQXW8t3I4AKj8q0n8P0s-aE",
        ),
        (
            "technology_cost_data",
            "instrat_2024",
            "https://docs.google.com/spreadsheets/d/1P-CGOaUUJt3J-6DfelAx5ilRSy0r2gCyJp_ZeHu1wbI",
        ),
        (
            "installed_capacity",
            "historical_totals",
            "https://docs.google.com/spreadsheets/d/1fwosQK76x_FoXRSI6tphexjMchXSIX0NqAfHNCDI_BA",
        ),
        (
            "annual_energy_flows",
            "historical",
            "https://docs.google.com/spreadsheets/d/1OWm53wIPTVJf0PGUrUxhjpzfVJgyMhwdBLg5cuRzvZY",
        ),
        (
            "capacity_utilisation",
            "historical",
            "https://docs.google.com/spreadsheets/d/1OTZmzscUlB6uxuaWvN5Et1qpixFMubnh2m4-qbZD7rk",
        ),
        # (
        #     "capacity_addition_potentials",
        #     "instrat_2024",
        #     "https://docs.google.com/spreadsheets/d/13mkEhrIJyCPVeOW_toOVa1p8E8wNg6rgj5hBjQQBflw",
        # ),
    ]:
        df = gsheet_to_df(url, sheet_name=variant)
        df = ignore_commented_rows_columns(df)
        df.to_csv(data_dir("input", f"{name};variant={variant}.csv"), index=False)


if __name__ == "__main__":

    download_input_data()
