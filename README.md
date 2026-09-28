# io-notebook

A [marimo](https://marimo.io) notebook that teaches environmentally extended
input-output analysis on a 9-sector synthetic economy: the Leontief quantity
model, the carbon footprint of final demand, a slider-driven scenario, and
self-checks modelled on those a scenario explorer runs against a real engine,
[RaMa-Scene](https://github.com/CMLPlatform/ramascene). It needs only numpy,
so it also runs in the browser through Pyodide.

The notation follows Miller and Blair (2009), *Input-Output Analysis:
Foundations and Extensions*, second edition: $\mathbf{A}$ for the technical
coefficients, $\mathbf{L} = (\mathbf{I} - \mathbf{A})^{-1}$ for the Leontief
inverse, and $q = \mathbf{e}'\mathbf{L}\mathbf{y}$ for the footprint.

## Run locally

```sh
uv run marimo edit io_leontief.py   # edit and run
uv run marimo run io_leontief.py    # app view
```

## Export to a static web page

```sh
uv run marimo export html-wasm io_leontief.py -o dist --mode run
python3 -m http.server --bind 127.0.0.1 8000 --directory dist
```

Use `--mode edit` to let readers change the code in the page. The page loads
Pyodide and packages from cdn.jsdelivr.net, so readers need internet access.
`dist/` (about 27 MB) is not committed.

## Deploy

Every push to `main` that passes the checks exports the page, smoke-tests it
in Chromium, and deploys it to the Cloudflare Pages project `io-notebook`.
The first run creates the project. The workflow needs two repository secrets:
`CLOUDFLARE_API_TOKEN` (a token with the *Cloudflare Pages: Edit*
permission) and `CLOUDFLARE_ACCOUNT_ID`.

## Checks

```sh
uv run pytest                     # pure functions, harness checks, and a headless run of every cell
uv run ruff check . && uv run ruff format --check .
uv run marimo check io_leontief.py
# browser smoke test against a served dist/ (install Chromium once first):
uv run --with playwright playwright install chromium
uv run --with playwright python scripts/check_wasm.py http://127.0.0.1:8000/ dist/shot.png
```

The synthetic economy borrows its recipe from a RaMa-Scene regression harness
that is still in progress and not in the public repository. The recipe is a
seeded `numpy.random.default_rng` and a dense, non-symmetric matrix. The scenario
explorer whose checks the notebook mirrors is not public either.

## Licence

The code is released under the [MIT licence](LICENSE). The notebook uses
synthetic data only.
