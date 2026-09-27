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
    It walks through the calculation behind environmentally extended
    input-output (EEIO) tools such as RaMa-Scene, on an economy small enough
    to read: 3 regions × 3 products = 9 sectors.

    The steps:

    1. a small synthetic economy (the technical coefficients **A** and final demand **y**);
    2. the Leontief model, x = (I − A)⁻¹ y;
    3. an emissions extension **b** and the carbon footprint b · x;
    4. a scenario you control with sliders;
    5. the self-checks a real engine should pass on every run.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. A small synthetic economy

    `A[i, j]` is how much of sector *i*'s output sector *j* buys to make one
    euro of its own output. `y[i]` is what households, governments and
    investment buy from sector *i* (final demand). `b[i]` is the emissions per
    euro of output of sector *i*.

    The numbers are random, not real data. The generator follows the
    synthetic matrices of the RaMa-Scene regression harness
    (`ramascene/tests/test_engine_regression.py`: seeded
    `numpy.random.default_rng`, small random off-diagonal entries). A dense,
    non-symmetric **A** matters: a mix-up of rows and columns changes the
    answer, so the checks below can catch it.
    """)
    return


@app.function
def make_economy(seed=SEED, n=N_SECTORS):
    """Return (A, y, b): coefficients, final demand (M EUR), emission intensities (kt CO2-eq per M EUR)."""
    rng = np.random.default_rng(seed)
    A = rng.random((n, n)) * 0.12  # column sums stay below 1, so the economy is productive
    y = rng.random(n) * 100
    b = rng.random(n) * 0.5
    b[1 :: len(PRODUCTS)] *= 4  # electricity is the most emission-intensive product
    b[0 :: len(PRODUCTS)] *= 2  # then cement
    return A, y, b


@app.cell
def _():
    A, y, b = make_economy()
    return A, b, y


@app.cell(hide_code=True)
def _(A, b, mo, y):
    mo.vstack(
        [
            mo.md("**Technical coefficients A** (rows: supplier, columns: buyer)"),
            mo.ui.table(
                [{"sector": s, **{c: round(float(v), 3) for c, v in zip(SECTORS, row)}} for s, row in zip(SECTORS, A)],
                selection=None,
                pagination=False,
            ),
            mo.md("**Final demand y** and **emission intensity b**"),
            mo.ui.table(
                [
                    {"sector": s, "y (M EUR)": round(float(yi), 1), "b (kt/M EUR)": round(float(bi), 3)}
                    for s, yi, bi in zip(SECTORS, y, b)
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

    To deliver final demand **y**, every sector also has to supply the others.
    Total output **x** satisfies x = A x + y, so (I − A) x = y and
    x = (I − A)⁻¹ y. The matrix L = (I − A)⁻¹ is the *Leontief inverse*.

    We solve (I − A) x = y with `numpy.linalg.solve` instead of computing the
    inverse: solving is faster and more accurate. The same switch is proposed
    for RaMa-Scene's engine (a local branch, not merged).
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

    Emissions of sector *i* are b[i] · x[i]. Summed, that is the carbon
    footprint of final demand: **b · L · y** = b · x. Grouped by region, it
    shows *where* the emissions of this demand happen.
    """)
    return


@app.function
def footprint(b, A, y):
    """Total emissions caused by final demand y: b . L . y."""
    return float(b @ leontief_output(A, y))


@app.function
def emissions_by_region(b, x, n_regions=N_REGIONS):
    """Emissions per producing region; rows are region-major, like RaMa-Scene's."""
    return (b * x).reshape(n_regions, -1).sum(axis=1)


@app.cell(hide_code=True)
def _(A, b, mo, x, y):
    _by_region = emissions_by_region(b, x)
    mo.vstack(
        [
            mo.md(f"Baseline footprint: **{footprint(b, A, y):.2f} kt CO2-eq**"),
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

    Change one technical coefficient (for example: Asian cement needs less
    electricity per euro) and/or one sector's final demand, by −50 % to +50 %.
    RaMa-Scene expresses its scenarios the same way: a percentage change to
    selected cells of A or of final demand (here, one cell of each).
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
def _(A, b, buyer, coefficient_pct, demand_pct, demand_sector, supplier, y):
    A_s, y_s = apply_scenario(
        A,
        y,
        coefficient=(SECTORS.index(supplier.value), SECTORS.index(buyer.value)),
        coefficient_pct=coefficient_pct.value,
        demand_sector=SECTORS.index(demand_sector.value),
        demand_pct=demand_pct.value,
    )
    x_s = leontief_output(A_s, y_s)
    baseline, scenario = footprint(b, A, y), footprint(b, A_s, y_s)
    return A_s, baseline, scenario, x_s, y_s


@app.cell(hide_code=True)
def _(b, baseline, mo, scenario, x, x_s):
    _before, _after = emissions_by_region(b, x), emissions_by_region(b, x_s)
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
    ## 5. The harness check

    A model can return a plausible-looking number and still be wrong. The
    first, second and fourth checks are the ones a test harness runs next to
    every scenario on the real RaMa-Scene engine; the Leontief identity is a
    direct check of the solve. Here they rerun each time you move a slider.

    * **0 % control**: a scenario that changes nothing must give exactly the baseline.
    * **Per-region recomputation**: emissions by region, recomputed by a
      different route (explicit inverse, one region at a time, compensated
      summation), agree with the main calculation to 1e-9.
    * **Leontief identity**: the solved output satisfies (I − A) x = y, up to
      rounding (around 1e-16 to 1e-15 relative).
    * **Effect sign**: with only non-negative numbers, raising a coefficient
      or a demand can only raise the footprint, and lowering can only lower it.
    """)
    return


@app.function
def harness_checks(A, y, b, A_s, y_s, coefficient_pct=0.0, demand_pct=0.0):
    """Self-checks on a baseline (A, y) and a scenario (A_s, y_s): {name: (passed, detail)}."""
    checks = {}

    A_0, y_0 = apply_scenario(A, y)  # every change 0 %
    control, baseline = footprint(b, A_0, y_0), footprint(b, A, y)
    checks["0 % control equals baseline"] = (control == baseline, f"{control!r} vs {baseline!r}")

    x_s = leontief_output(A_s, y_s)
    engine = emissions_by_region(b, x_s)
    L = np.linalg.inv(np.eye(len(A_s)) - A_s)
    per_sector = b * (L @ y_s)
    size = len(A_s) // len(engine)
    direct = [math.fsum(per_sector[r * size : (r + 1) * size]) for r in range(len(engine))]
    worst = max(abs(e - d) / abs(d) for e, d in zip(engine, direct))
    checks["per-region recomputation agrees"] = (worst < 1e-9, f"largest relative difference {worst:.1e}")

    residual = np.abs((np.eye(len(A_s)) - A_s) @ x_s - y_s).max() / np.abs(y_s).max()
    checks["Leontief identity (I - A) x = y"] = (residual < 1e-12, f"relative residual {residual:.1e}")

    change = footprint(b, A_s, y_s) - baseline
    signs = {np.sign(coefficient_pct), np.sign(demand_pct)} - {0}
    if len(signs) == 1:
        expected = signs.pop()
        checks["effect sign as expected"] = (np.sign(change) == expected, f"change {change:+.3g}")
    else:
        checks["effect sign as expected"] = (True, "not applicable (no change, or changes pull both ways)")
    return {name: (bool(ok), detail) for name, (ok, detail) in checks.items()}


@app.cell(hide_code=True)
def _(A, A_s, b, coefficient_pct, demand_pct, mo, y, y_s):
    _checks = harness_checks(A, y, b, A_s, y_s, coefficient_pct.value, demand_pct.value)
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

    RaMa-Scene runs the same algebra on EXIOBASE: 49 regions (44 countries and 5
    rest-of-world regions) × 200 products = 9 800 sectors, so **A** has about 96 million cells. Its harness also
    reproduces reference values computed independently in Octave. Try: set a
    coefficient change and a demand change with opposite signs, and see why
    the sign check then says "not applicable".
    """)
    return


if __name__ == "__main__":
    app.run()
