from functools import reduce

import numpy as np

from pypsa.descriptors import nominal_attrs
from pypsa.statistics import (
    pass_empty_series_if_keyerror,
    port_mask,
    get_ports,
    aggregate_components,
    get_weightings,
    aggregate_timeseries,
)


def make_capex_calculator(attr="capital_cost", name="Capital Expenditure"):
    # Based on https://github.com/PyPSA/PyPSA/blob/v0.27.0/pypsa/statistics.py#L377
    def capex(
        n,
        comps=None,
        aggregate_groups="sum",
        groupby=None,
        bus_carrier=None,
        nice_names=True,
    ):
        """
        Calculate the capital expenditure of the network in given currency.

        If `bus_carrier` is given, only components which are connected to buses
        with carrier `bus_carrier` are considered.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.
        """

        @pass_empty_series_if_keyerror
        def func(n, c):
            col = n.df(c).eval(f"{nominal_attrs[c]}_opt * {attr}")
            if bus_carrier is not None:
                masks = [
                    port_mask(n, c, port=port, bus_carrier=bus_carrier)
                    for port in get_ports(n, c)
                ]
                mask = reduce(np.logical_or, masks)
                col = col[mask.astype(bool)]
            return col

        df = aggregate_components(
            n,
            func,
            comps=comps,
            agg=aggregate_groups,
            groupby=groupby,
            nice_names=nice_names,
        )
        df.attrs["name"] = name
        df.attrs["unit"] = "currency"
        return df

    return capex


def make_opex_calculator(attr="marginal_cost", name="Operational Expenditure"):
    # Based on https://github.com/PyPSA/PyPSA/blob/v0.27.0/pypsa/statistics.py#L613
    def opex(
        n,
        comps=None,
        aggregate_time="sum",
        aggregate_groups="sum",
        groupby=None,
        bus_carrier=None,
        nice_names=True,
    ):
        """
        Calculate the operational expenditure in the network in given currency.

        If `bus_carrier` is given, only components which are connected to buses
        with carrier `bus_carrier` are considered.

        For information on the list of arguments, see the docs in
        `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.

        Parameters
        ----------
        aggregate_time : str, bool, optional
            Type of aggregation when aggregating time series.
            Note that for {'mean', 'sum'} the time series are aggregated
            using snapshot weightings. With False the time series is given. Defaults to 'sum'.
        """

        @pass_empty_series_if_keyerror
        def func(n, c):
            if c in n.branch_components:
                p = n.pnl(c).p0
            elif c == "StorageUnit":
                p = n.pnl(c).p_dispatch
            else:
                p = n.pnl(c).p
            if bus_carrier is not None:
                masks = [
                    port_mask(n, c, port=port, bus_carrier=bus_carrier)
                    for port in get_ports(n, c)
                ]
                mask = reduce(np.logical_or, masks)
                p = p * mask

            opex = p * n.get_switchable_as_dense(c, attr)
            opex = opex.loc[:, (opex != 0).any()]
            weights = get_weightings(n, c)
            return aggregate_timeseries(opex, weights, agg=aggregate_time)

        df = aggregate_components(
            n,
            func,
            comps=comps,
            agg=aggregate_groups,
            groupby=groupby,
            nice_names=nice_names,
        )
        df.attrs["name"] = name
        df.attrs["unit"] = "currency"
        return df

    return opex
