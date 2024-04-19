from xarray import DataArray
from numpy import isnan
from linopy.expressions import merge


def define_operational_limit_link(n, sns):
    """
    Based on https://github.com/PyPSA/PyPSA/blob/v0.27.0/pypsa/optimization/global_constraints.py#L312.

    Defines operational limit constraints. It limits the net production at bus0 of a link.

    Parameters
    ----------
    n : pypsa.Network
    sns : list-like
        Set of snapshots to which the constraint should be applied.

    Returns
    -------
    None.
    """
    m = n.model
    weightings = n.snapshot_weightings.loc[sns]
    glcs = n.global_constraints.query('type == "operational_limit"')

    if n._multi_invest:
        period_weighting = n.investment_period_weightings.years[sns.unique("period")]
        weightings = weightings.mul(period_weighting, level=0, axis=0)

    for name, glc in glcs.iterrows():
        snapshots = (
            sns
            if isnan(glc.investment_period)
            else sns[sns.get_loc(glc.investment_period)]
        )
        lhs = []
        rhs = glc.constant

        # links
        links = n.links.query("carrier == @glc.carrier_attribute")
        if not links.empty:
            # Negative p0 is positve production at bus0
            p = -m["Link-p"].loc[snapshots, links.index]
            w = DataArray(weightings.generators[snapshots])
            if "dim_0" in w.dims:
                w = w.rename({"dim_0": "snapshot"})
            expr = (p * w).sum()
            lhs.append(expr)

        if not lhs:
            continue

        lhs = merge(lhs)
        sign = "=" if glc.sense == "==" else glc.sense
        m.add_constraints(lhs, sign, rhs, f"GlobalConstraint-{name}")


def define_battery_large_charger_capacity_constraint(n, sns):
    # TODO: generalise for other technologies with fixed ratios
    reverse_links = n.meta["reverse_links"]

    parent_tech = "battery large power"
    child_tech = "battery large charger"

    parents = n.links.query(
        f"technology == '{parent_tech}' and p_nom_max > p_nom_min and p_nom_extendable"
    ).index

    if parents.empty:
        return

    def rename_parents_to_children(parents):
        return parents.str.replace(parent_tech, child_tech)

    children = rename_parents_to_children(parents)

    m = n.model

    p_nom = m.variables["Link-p_nom"]
    p_nom_children = p_nom.loc[children]
    p_nom_parents = p_nom.loc[parents]

    if reverse_links:
        ratio = 1 / n.links.loc[children, "efficiency"]
        ratio = ratio.rename_axis("Link-ext").to_xarray()
    else:
        ratio = 1

    p_nom_parents = p_nom_parents.assign_coords(
        {"Link-ext": rename_parents_to_children(p_nom_parents.coords["Link-ext"])}
    )

    lhs = p_nom_children - ratio * p_nom_parents
    m.add_constraints(lhs, "==", 0, name="Link-ext-battery_large_charger_capacity")
