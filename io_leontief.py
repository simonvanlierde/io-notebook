import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import math

    import numpy as np

    REGIONS = ["Europe", "Asia", "Americas"]
    PRODUCTS = ["cement", "electricity", "services"]
    SECTORS = [f"{r} {p}" for r in REGIONS for p in PRODUCTS]  # region-major, like RaMa-Scene's rows
    SEED = 20260926
    N_SECTORS, N_REGIONS = len(SECTORS), len(REGIONS)


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Input-output analysis in the browser: the Leontief model

    This notebook runs entirely in your browser (Python compiled to WebAssembly).
    It works through the calculation behind environmentally extended
    input-output (EEIO) tools such as RaMa-Scene, on an economy small enough
    to read: 3 regions × 3 products = 9 sectors.

    ### Learning goals

    - Read the technical coefficient matrix $\mathbf{A}$ and the final demand vector $\mathbf{y}$ of a small multi-regional economy.
    - Calculate total product output with the Leontief quantity model, $\mathbf{x} = (\mathbf{I} - \mathbf{A})^{-1}\mathbf{y}$.
    - Calculate the carbon footprint of final demand from the emission intensity vector $\mathbf{e}$.
    - Build a scenario that changes one technical coefficient and one final demand, and compare it with the baseline.
    - Check the result with the self-checks a real engine should pass on every run.

    The notation follows Miller and Blair (2009), *Input-Output Analysis:
    Foundations and Extensions*, 2nd edition. Bold capitals are matrices, bold
    lower case letters are column vectors, a prime ($\mathbf{e}'$) is a transpose,
    and a hat ($\hat{\mathbf{e}}$) puts a vector on the diagonal of a matrix.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. A small synthetic economy

    The economy has three inputs:

    - $\mathbf{A}$, the technical coefficient matrix. $A_{ij}$ is the input from
      sector $i$ that sector $j$ needs to make one euro of output.
    - $\mathbf{y}$, the final demand vector. $y_i$ is what households,
      governments and investment buy from sector $i$, in M EUR.
    - $\mathbf{e}$, the emission intensity vector. $e_i$ is the emissions of
      sector $i$ per M EUR of output, in kt CO2-eq.

    Sectors are ordered region by region: the three European products first,
    then Asia, then the Americas.

    The numbers are random, not real data. The generator borrows its recipe
    from a regression harness for RaMa-Scene that is still in progress and not
    in the public repository: a seeded `numpy.random.default_rng` and a dense,
    non-symmetric matrix. That harness generates $\mathbf{L}$ directly; here we
    generate $\mathbf{A}$ and solve. A dense, non-symmetric $\mathbf{A}$
    matters: a mix-up of rows and columns changes the answer, so the checks
    below can catch it.
    """)
    return


@app.function
def make_economy(seed=SEED):
    """Return (A, y, e): technical coefficients, final demand (M EUR), emission intensities (kt CO2-eq per M EUR)."""
    n = N_SECTORS
    rng = np.random.default_rng(seed)
    A = rng.random((n, n)) * 0.12  # column sums well below 1 (checked in tests), so the economy is productive
    y = rng.random(n) * 100
    e = rng.random(n) * 0.5
    e[1 :: len(PRODUCTS)] *= 4  # electricity is the most emission-intensive product
    e[0 :: len(PRODUCTS)] *= 2  # then cement
    return A, y, e


@app.cell
def _():
    A, y, e = make_economy()
    return A, e, y


@app.cell(hide_code=True)
def _(A, e, mo, y):
    mo.vstack(
        [
            mo.md("**Technical coefficient matrix A** (rows: supplier, columns: buyer)"),
            mo.ui.table(
                [{"sector": s, **{c: round(float(v), 3) for c, v in zip(SECTORS, row)}} for s, row in zip(SECTORS, A)],
                selection=None,
                pagination=False,
            ),
            mo.md("**Final demand y** and **emission intensity e**"),
            mo.ui.table(
                [
                    {"sector": s, "y (M EUR)": round(float(yi), 1), "e (kt/M EUR)": round(float(ei), 3)}
                    for s, yi, ei in zip(SECTORS, y, e)
                ],
                selection=None,
                pagination=False,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The Leontief model

    To deliver final demand $\mathbf{y}$, every sector also has to supply the
    others. Total product output $\mathbf{x}$ therefore satisfies
    $\mathbf{x} = \mathbf{A}\mathbf{x} + \mathbf{y}$. Rearranged:

    $$\mathbf{x} = (\mathbf{I} - \mathbf{A})^{-1}\mathbf{y} = \mathbf{L}\mathbf{y}$$

    $\mathbf{L}$ is the *Leontief inverse* (Miller and Blair, 2009, chapter 2).

    We solve $(\mathbf{I} - \mathbf{A})\mathbf{x} = \mathbf{y}$ with
    `numpy.linalg.solve` instead of calculating $\mathbf{L}$ first. Solving is
    faster and more accurate. The same switch is proposed for RaMa-Scene's
    engine, on a local branch that is not merged.
    """)
    return


@app.function
def leontief_output(A, y):
    """Total output x from (I - A) x = y, solved without forming the inverse."""
    return np.linalg.solve(np.eye(len(A)) - A, y)


@app.cell
def _(A, y):
    x = leontief_output(A, y)
    return (x,)


@app.cell(hide_code=True)
def _(mo, x, y):
    mo.md(
        f"Final demand adds up to **{y.sum():.1f} M EUR**; the economy has to "
        f"produce **{x.sum():.1f} M EUR** in total to deliver it. The difference, "
        f"{x.sum() - y.sum():.1f} M EUR, is intermediate production: sectors supplying each other."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Emissions and the footprint

    The emissions of each sector are its intensity times its output,
    $\hat{\mathbf{e}}\mathbf{x}$. Their sum is the carbon footprint of final
    demand:

    $$q = \mathbf{e}'\mathbf{L}\mathbf{y} = \mathbf{e}'\mathbf{x}$$

    Grouped by region, the sector emissions show *where* the emissions caused
    by this demand happen.
    """)
    return


@app.function
def footprint(e, A, y):
    """Total emissions caused by final demand y: e' L y."""
    return float(e @ leontief_output(A, y))


@app.function
def emissions_by_region(e, x, n_regions=N_REGIONS):
    """Emissions per producing region; rows are region-major, like RaMa-Scene's."""
    return (e * x).reshape(n_regions, -1).sum(axis=1)


@app.cell(hide_code=True)
def _(A, e, mo, x, y):
    _by_region = emissions_by_region(e, x)
    mo.vstack(
        [
            mo.md(f"Baseline footprint: **{footprint(e, A, y):.2f} kt CO2-eq**"),
            mo.ui.table(
                [
                    {"region": r, "emissions (kt CO2-eq)": round(float(v), 2), "share": f"{v / _by_region.sum():.0%}"}
                    for r, v in zip(REGIONS, _by_region)
                ],
                selection=None,
                pagination=False,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. A scenario

    A scenario is a counterfactual version of the economy. This one changes
    two numbers, each by −50 % to +50 %:

    - one technical coefficient $A_{ij}$, a technology change (for example,
      Asian cement needs less electricity per euro);
    - one sector's final demand $y_i$, a demand change.

    The scenario works on copies of $\mathbf{A}$ and $\mathbf{y}$, so the
    baseline stays unchanged. RaMa-Scene expresses its scenarios the same way:
    a percentage change to selected cells of $\mathbf{A}$ or of final demand.
    Here, one cell of each.
    """)
    return


@app.function
def apply_scenario(A, y, coefficient=(0, 0), coefficient_pct=0.0, demand_sector=0, demand_pct=0.0):
    """Copies of (A, y) with A[coefficient] and y[demand_sector] scaled by (1 + pct/100)."""
    A_s, y_s = A.copy(), y.copy()
    A_s[coefficient] *= 1 + coefficient_pct / 100
    y_s[demand_sector] *= 1 + demand_pct / 100
    return A_s, y_s


@app.cell(hide_code=True)
def _(mo):
    supplier = mo.ui.dropdown(SECTORS, value="Asia electricity", label="Coefficient: supplier")
    buyer = mo.ui.dropdown(SECTORS, value="Asia cement", label="buyer")
    coefficient_pct = mo.ui.slider(-50, 50, step=5, value=0, label="Change in A[supplier, buyer] (%)", show_value=True)
    demand_sector = mo.ui.dropdown(SECTORS, value="Europe cement", label="Final demand for")
    demand_pct = mo.ui.slider(-50, 50, step=5, value=0, label="Change in final demand (%)", show_value=True)
    mo.vstack(
        [
            mo.hstack([supplier, buyer], justify="start"),
            coefficient_pct,
            mo.hstack([demand_sector], justify="start"),
            demand_pct,
        ]
    )
    return buyer, coefficient_pct, demand_pct, demand_sector, supplier


@app.cell
def _(A, buyer, coefficient_pct, demand_pct, demand_sector, e, supplier, y):
    A_s, y_s = apply_scenario(
        A,
        y,
        coefficient=(SECTORS.index(supplier.value), SECTORS.index(buyer.value)),
        coefficient_pct=coefficient_pct.value,
        demand_sector=SECTORS.index(demand_sector.value),
        demand_pct=demand_pct.value,
    )
    x_s = leontief_output(A_s, y_s)
    baseline, scenario = footprint(e, A, y), footprint(e, A_s, y_s)
    return A_s, baseline, scenario, x_s, y_s


@app.cell(hide_code=True)
def _(baseline, e, mo, scenario, x, x_s):
    _before, _after = emissions_by_region(e, x), emissions_by_region(e, x_s)
    _change = scenario - baseline
    mo.vstack(
        [
            mo.md(
                f"Scenario footprint: **{scenario:.2f} kt CO2-eq** "
                f"(baseline {baseline:.2f}; change {_change:+.3f} kt, {_change / baseline:+.2%})"
            ),
            mo.ui.table(
                [
                    {
                        "region": r,
                        "baseline": round(float(p), 3),
                        "scenario": round(float(q), 3),
                        "change": round(float(q - p), 3),
                    }
                    for r, p, q in zip(REGIONS, _before, _after)
                ],
                selection=None,
                pagination=False,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Updating the baseline instead of re-solving: Woodbury

    On EXIOBASE, $\mathbf{I} - \mathbf{A}$ has 9 800 rows. A serverless
    explorer cannot solve that system again for every slider move, but it can
    keep the baseline $\mathbf{L}$ (or, in practice, the few rows of
    $\mathbf{e}'\mathbf{L}$ it needs) and *update* it. A scenario that changes
    the coefficients in $k$ columns of $\mathbf{A}$ is a rank-$k$ change:

    $$\mathbf{A}_s = \mathbf{A} + \mathbf{U}\mathbf{V}'$$

    where $\mathbf{U}$ ($n \times k$) holds the changes to those columns and
    $\mathbf{V}$ ($n \times k$) picks them out of the identity matrix. The
    Woodbury matrix identity (the Sherman–Morrison formula when $k = 1$) gives the new inverse from the old one:

    $$(\mathbf{I} - \mathbf{A} - \mathbf{U}\mathbf{V}')^{-1}
      = \mathbf{L} + \mathbf{L}\mathbf{U}\,(\mathbf{I}_k - \mathbf{V}'\mathbf{L}\mathbf{U})^{-1}\,\mathbf{V}'\mathbf{L}$$

    Only a $k \times k$ system is solved, and $k$ is the number of changed
    columns: here 1, whatever the size of the economy. Applied to the
    scenario's final demand, $\mathbf{x}_s = \mathbf{L}\mathbf{y}_s +
    \mathbf{L}\mathbf{U}\,(\mathbf{I}_k - \mathbf{V}'\mathbf{L}\mathbf{U})^{-1}\,
    \mathbf{V}'\mathbf{L}\mathbf{y}_s$.

    The update is exact algebra, but it takes a different numerical route
    from the full solve, so the harness below checks that both agree.
    """)
    return


@app.function
def low_rank_factors(A, A_s):
    """(U, V) with A_s = A + U V': U holds the changed columns of A, V the matching columns of I."""
    cols = np.flatnonzero((A_s != A).any(axis=0))
    return (A_s - A)[:, cols], np.eye(len(A))[:, cols]


@app.function
def woodbury_output(L, U, V, y):
    """Total output for A + U V' from the baseline Leontief inverse L, via the Woodbury identity."""
    x = L @ y
    if U.shape[1] == 0:
        return x
    LU = L @ U
    return x + LU @ np.linalg.solve(np.eye(U.shape[1]) - V.T @ LU, V.T @ x)


@app.cell
def _(A, A_s, x_s, y_s):
    U, V = low_rank_factors(A, A_s)
    x_w = woodbury_output(np.linalg.inv(np.eye(len(A)) - A), U, V, y_s)
    woodbury_gap = np.abs(x_w - x_s).max() / np.abs(x_s).max()
    return U, woodbury_gap


@app.cell(hide_code=True)
def _(U, mo, woodbury_gap):
    mo.md(
        f"This scenario changes **{U.shape[1]}** column(s) of A, so the update solves a "
        f"{U.shape[1]} × {U.shape[1]} system. Largest relative difference from the full "
        f"re-solve: **{woodbury_gap:.1e}**."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. The harness check

    A model can return a plausible-looking number and still be wrong. The
    first, second and fourth checks follow those that a scenario explorer
    built on the RaMa-Scene engine (in progress, not public) runs next to
    every scenario. Its sign check is narrower: every scenario there is a
    cut, and no total may rise. The Leontief identity is a direct check of
    the solve, and the Woodbury check tests the update from section 5. Here
    all five rerun each time you move a slider.

    * **0 % control**: a scenario that changes nothing must give exactly the baseline.
    * **Per-region recomputation**: emissions by region, recomputed by a
      different route (explicit inverse, one region at a time, compensated
      summation), agree with the main calculation to 1e-9.
    * **Leontief identity**: the solved output satisfies
      $(\mathbf{I} - \mathbf{A})\mathbf{x} = \mathbf{y}$, up to rounding (around 1e-16 to 1e-15 relative).
    * **Effect sign**: with only non-negative numbers, raising a coefficient
      or a demand can only raise the footprint, and lowering can only lower it.
    * **Woodbury update equals re-solve**: the output updated from the
      baseline $\mathbf{L}$ agrees with the full solve to 1e-12 relative.
    """)
    return


@app.function
def harness_checks(A, y, e, A_s, y_s, coefficient_pct=0.0, demand_pct=0.0):
    """Self-checks on a baseline (A, y) and a scenario (A_s, y_s): {name: (passed, detail)}."""
    checks = {}

    A_0, y_0 = apply_scenario(A, y)  # every change 0 %
    control, baseline = footprint(e, A_0, y_0), footprint(e, A, y)
    checks["0 % control equals baseline"] = (control == baseline, f"{control!r} vs {baseline!r}")

    x_s = leontief_output(A_s, y_s)
    engine = emissions_by_region(e, x_s)
    L = np.linalg.inv(np.eye(len(A_s)) - A_s)
    per_sector = e * (L @ y_s)
    size = len(A_s) // len(engine)
    direct = [math.fsum(per_sector[r * size : (r + 1) * size]) for r in range(len(engine))]
    worst = max(abs(g - d) / abs(d) for g, d in zip(engine, direct))
    checks["per-region recomputation agrees"] = (worst < 1e-9, f"largest relative difference {worst:.1e}")

    residual = np.abs((np.eye(len(A_s)) - A_s) @ x_s - y_s).max() / np.abs(y_s).max()
    checks["Leontief identity (I - A) x = y"] = (residual < 1e-12, f"relative residual {residual:.1e}")

    change = footprint(e, A_s, y_s) - baseline
    signs = {np.sign(coefficient_pct), np.sign(demand_pct)} - {0}
    if len(signs) == 1:
        expected = signs.pop()
        checks["effect sign as expected"] = (np.sign(change) == expected, f"change {change:+.3g}")
    else:
        checks["effect sign as expected"] = (True, "not applicable (no change, or changes pull both ways)")

    U, V = low_rank_factors(A, A_s)
    x_w = woodbury_output(np.linalg.inv(np.eye(len(A)) - A), U, V, y_s)
    gap = np.abs(x_w - x_s).max() / np.abs(x_s).max()
    checks["Woodbury update equals re-solve"] = (gap < 1e-12, f"rank {U.shape[1]}, relative difference {gap:.1e}")
    return {name: (bool(ok), detail) for name, (ok, detail) in checks.items()}


@app.cell(hide_code=True)
def _(A, A_s, coefficient_pct, demand_pct, e, mo, y, y_s):
    _checks = harness_checks(A, y, e, A_s, y_s, coefficient_pct.value, demand_pct.value)
    _all = all(ok for ok, _ in _checks.values())
    mo.vstack(
        [
            mo.md(f"**Harness check: {'all pass' if _all else 'FAIL'}**"),
            mo.ui.table(
                [
                    {"check": name, "result": "pass" if ok else "FAIL", "detail": detail}
                    for name, (ok, detail) in _checks.items()
                ],
                selection=None,
                pagination=False,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What the real engine adds

    [RaMa-Scene](https://github.com/CMLPlatform/ramascene) runs the same
    algebra on EXIOBASE: 49 regions (44 countries and 5 rest-of-world regions)
    × 200 products = 9 800 sectors, so $\mathbf{A}$ has about 96 million
    cells. Its test suite also compares results with reference values
    computed independently in Octave.

    ### Reflect
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(
                "- Set a coefficient change and a demand change with opposite signs. Why "
                'does the sign check then say "not applicable"?'
            ),
            mo.accordion(
                {
                    "Show answer": mo.md(
                        r"""
    The check relies on all numbers being non-negative. Then $\mathbf{L}$ is
    non-negative too, so raising a coefficient or a demand can only raise the
    footprint, and lowering one can only lower it. With opposite signs, one
    change pushes the footprint up and the other pushes it down. The net
    direction depends on how large each effect is, so no sign is guaranteed
    and the check does not judge it.
    """
                    )
                }
            ),
            mo.md(
                "- Make Asian cement use 50 % less Asian electricity. Why do the emissions "
                "of Europe and the Americas change too, although their coefficients did "
                "not?"
            ),
            mo.accordion(
                {
                    "Show answer": mo.md(
                        r"""
    $\mathbf{A}$ is dense: every sector buys from sectors in every region. Asian
    cement now needs less Asian electricity, so Asia makes less electricity.
    That in turn needs fewer inputs, from every region, and so on through all
    rounds of $\mathbf{L} = \mathbf{I} + \mathbf{A} + \mathbf{A}^2 + \dots$.
    Total output $\mathbf{x}$ falls in sectors far from the change, and
    emissions are $\hat{\mathbf{e}}\mathbf{x}$, so they fall in every region,
    Europe and the Americas included.
    """
                    )
                }
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
