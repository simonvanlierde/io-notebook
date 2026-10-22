"""Open the exported WASM notebook in headless Chromium and check it runs and has no WCAG 2.2 AA violations.

Usage: uv run --with playwright python scripts/check_wasm.py http://127.0.0.1:PORT/ SCREENSHOT.png
Needs the dist/ export served over HTTP and internet access (Pyodide loads from cdn.jsdelivr.net).
"""

import json
import re
import sys
import time

from playwright.sync_api import expect, sync_playwright

URL, SHOT = sys.argv[1], sys.argv[2]
AXE_URL = "https://cdn.jsdelivr.net/npm/axe-core@4.13.0/axe.min.js"  # same CDN the page already needs
AXE_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]
# Elements of marimo's own widgets that fail axe and that the notebook cannot change (axe-core 4.13.0, marimo 0.25.1).
# All rules stay on for the rest of the page. If marimo renames an element, the new name fails the scan loudly.
AXE_EXCLUDE = [  # light-DOM elements, left out of the scan
    '[data-testid="notebook-actions-dropdown"]',  # button-name: marimo's icon-only menu button
    '[data-testid="watermark"]',  # color-contrast: marimo's "made with marimo" badge
]
# Shadow-DOM elements: axe's exclude cannot reach inside a shadow root, so drop these violations after the scan.
# (rule, marimo widget host, regex on the node target). Cell content in marimo-table still gets checked.
AXE_MARIMO_WIDGETS = [
    ("aria-input-field-name", "marimo-slider", ""),  # marimo's slider thumb has no accessible name
    ("color-contrast", "marimo-table", r'"button\[|opacity-80|italic'),  # muted table toolbar buttons, "No selection"
]
FOOTPRINT = re.compile(r"Scenario footprint: ([\d.]+) kt CO2-eq")


def is_marimo_widget(rule, target):
    text = json.dumps(target)
    return any(r == rule and host in text and re.search(inner, text) for r, host, inner in AXE_MARIMO_WIDGETS)


def scenario_text(page):
    return page.get_by_text(FOOTPRINT).first.inner_text()


def assert_all_pass(page, label):
    cells = page.locator("td").filter(has_text=re.compile(r"^(pass|FAIL)$"))
    # The table is its own cell and can render after the summary line the caller waited for.
    expect(cells).to_have_count(5, timeout=60_000)
    results = cells.all_inner_texts()
    print(f"check results {label}:", results)
    # one row per check in harness_checks (io_leontief.py)
    assert results.count("pass") == 5 and "FAIL" not in results, results


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1500})
    page.on("console", lambda m: m.type == "error" and print("console error:", m.text[:200]))
    start = time.monotonic()
    page.goto(URL)
    expect(page.get_by_text("Harness check: all pass")).to_be_visible(timeout=300_000)
    print(f"notebook ran in {time.monotonic() - start:.1f} s")

    assert_all_pass(page, "before")
    before = scenario_text(page)
    print("before:", before)

    slider = page.get_by_role("slider").first
    slider.focus()
    slider.press("Home")  # coefficient change -> -50 %
    expect(page.get_by_text(FOOTPRINT).first).not_to_have_text(before, timeout=60_000)
    # the harness table is its own cell: wait for its effect-sign detail to show the cut, not stale results
    expect(page.get_by_text(re.compile(r"^change -")).first).to_be_visible(timeout=60_000)
    expect(page.get_by_text("Harness check: all pass")).to_be_visible(timeout=60_000)
    after = scenario_text(page)
    print("after slider to -50 %:", after)
    assert_all_pass(page, "after")
    assert float(FOOTPRINT.search(after)[1]) < float(FOOTPRINT.search(before)[1])

    page.add_script_tag(url=AXE_URL)
    violations = page.evaluate(
        "([tags, exclude]) => axe.run({exclude}, {runOnly: tags}).then(r => r.violations)", [AXE_TAGS, AXE_EXCLUDE]
    )
    for v in violations:
        v["nodes"] = [n for n in v["nodes"] if not is_marimo_widget(v["id"], n["target"])]
    violations = [v for v in violations if v["nodes"]]
    for v in violations:
        print(f"axe {v['impact']} {v['id']}: {v['help']} ({len(v['nodes'])} nodes, first: {v['nodes'][0]['target']})")
    assert not violations, f"{len(violations)} WCAG 2.2 AA violations"
    print("axe: no WCAG 2.2 AA violations")

    page.screenshot(path=SHOT)
    print("screenshot:", SHOT)
    browser.close()
