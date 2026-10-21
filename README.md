# io-notebook

[![checks](https://github.com/simonvanlierde/io-notebook/actions/workflows/checks.yml/badge.svg)](https://github.com/simonvanlierde/io-notebook/actions/workflows/checks.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A [marimo](https://marimo.io) notebook that teaches environmentally extended
input-output analysis on a synthetic economy of 9 sectors: the Leontief quantity
model, the carbon footprint of final demand and a scenario you change with a
slider. It ends with self-checks modelled on the checks a scenario explorer runs
against [RaMa-Scene](https://github.com/CMLPlatform/ramascene). It needs only
numpy, so it also runs in the browser through Pyodide.

The notation follows Miller and Blair (2009), *Input-Output Analysis:
Foundations and Extensions*, second edition: $\mathbf{A}$ for the technical
coefficients, $\mathbf{L} = (\mathbf{I} - \mathbf{A})^{-1}$ for the Leontief
inverse, and $q = \mathbf{e}'\mathbf{L}\mathbf{y}$ for the footprint.

## Run

```sh
uv run marimo edit io_leontief.py   # edit and run
uv run marimo run io_leontief.py    # app view
```

Export to a static page (`--mode edit` lets readers change the code). The page
loads Pyodide and packages from cdn.jsdelivr.net, so readers need internet
access. `dist/` is not committed.

```sh
uv run marimo export html-wasm io_leontief.py -o dist --mode run
python3 -m http.server --bind 127.0.0.1 8000 --directory dist
```

## Checks

```sh
uv run pytest                     # pure functions, harness checks, and a headless run of every cell
uv run ruff check . && uv run ruff format --check .
uv run marimo check io_leontief.py
# browser smoke test against a served dist/ (install Chromium once first):
uv run --with playwright playwright install chromium
uv run --with playwright python scripts/check_wasm.py http://127.0.0.1:8000/ dist/shot.png
```

## Deploy

Off unless the repository variable `DEPLOY_PAGES` is `true`. Then every push to
`main` that passes the checks exports the page, smoke-tests it in Chromium and
deploys it to the Cloudflare Pages project `io-notebook`, which the first run
creates. It needs the repository secrets `CLOUDFLARE_API_TOKEN` (with the
*Cloudflare Pages: Edit* permission) and `CLOUDFLARE_ACCOUNT_ID`.

## Data and licence

The notebook uses synthetic data only. I took its recipe, a seeded
`numpy.random.default_rng` and a dense, non-symmetric matrix, from a RaMa-Scene
regression harness, which is public on [my fork of RaMa-Scene](https://github.com/simonvanlierde/ramascene/blob/main/ramascene/tests/test_engine_regression.py).
The same harness is an open pull request against the upstream repository
([#72](https://github.com/CMLPlatform/ramascene/pull/72)).

The code is released under the [MIT licence](LICENSE).
