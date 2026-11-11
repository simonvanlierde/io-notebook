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

## Social card

[`public/og.png`](public/og.png) is the 1200x630 card shown when the link is shared. It is rendered from
[`docs/og-card.html`](docs/og-card.html), which sets the equation in KaTeX and takes its fonts from the
notebook page's own theme, so rendering needs network access. The figure is the baseline footprint
the notebook prints for `make_economy()`. marimo copies `public/` into the export, and `head.html` holds the tags.
With Playwright and Chromium installed:

```sh
playwright screenshot --browser=chromium --viewport-size=1200,630 --wait-for-timeout=2500 \
  file://$PWD/docs/og-card.html public/og.png
sed -i.bak "s|og.png?v=[0-9a-f]*|og.png?v=$(shasum -a 256 public/og.png | cut -c1-8)|" head.html
rm head.html.bak
```

The `sed` line sets the `?v=` on the `og:image` URL to the PNG's hash, because link previews are cached by
image URL. `tests/test_io_leontief.py` fails when the two disagree.

## Deploy

The page is live at <https://simonvanlierde.com/io-notebook/>. Every push
to `main` that passes the checks exports the page, smoke-tests it in Chromium
and deploys it to GitHub Pages.

Each pull request gets a preview at a `pr-<number>` address on Cloudflare
Workers, linked from the PR as its `preview` deployment (`wrangler.jsonc`). The
job needs `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` in the `preview`
environment's secrets, and skips without them. Drafts get no preview. When a PR
closes or turns draft, `preview-retire.yml` points its address at a short
"preview closed" page, since wrangler cannot delete an alias.

## Data and licence

The notebook uses synthetic data only. I took its recipe, a seeded
`numpy.random.default_rng` and a dense, non-symmetric matrix, from a RaMa-Scene
regression harness, which is public on [my fork of RaMa-Scene](https://github.com/simonvanlierde/ramascene/blob/main/ramascene/tests/test_engine_regression.py).
The same harness is an open pull request against the upstream repository
([#72](https://github.com/CMLPlatform/ramascene/pull/72)).

The code is released under the [MIT licence](LICENSE).
