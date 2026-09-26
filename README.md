# io-notebook

A [marimo](https://marimo.io) notebook that teaches environmentally extended
input-output analysis on a 9-sector synthetic economy: the Leontief model,
an emissions footprint, a slider-driven scenario, and the self-checks a real
engine (RaMa-Scene) runs on every scenario. It needs only numpy, so it also
runs in the browser through Pyodide.

## Run locally

```sh
uvx marimo edit io_leontief.py   # edit and run
uvx marimo run io_leontief.py    # app view
```

## Export to a static web page

```sh
uv run marimo export html-wasm io_leontief.py -o dist --mode run
python3 -m http.server --bind 127.0.0.1 8000 --directory dist
```

Use `--mode edit` to let readers change the code in the page. The page loads
Pyodide and packages from cdn.jsdelivr.net, so readers need internet access.
`dist/` (about 27 MB) is not committed.

## Checks

```sh
uv run pytest                     # pure functions: Leontief identity, 0 % control, a hand-computed 2x2
uv run ruff check . && uv run ruff format --check .
uv run marimo check io_leontief.py
# browser smoke test against a served dist/:
uv run --with playwright python scripts/check_wasm.py http://127.0.0.1:8000/ shot.png
```

The synthetic economy follows the synthetic matrices in RaMa-Scene's
regression harness (`ramascene/tests/test_engine_regression.py`): a seeded
`numpy.random.default_rng` and a dense, non-symmetric matrix.

## Licence

[author: decide]
