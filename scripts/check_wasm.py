"""Open the exported WASM notebook in headless Chromium and check it runs.

Usage: uv run --with playwright python scripts/check_wasm.py http://127.0.0.1:PORT/ SCREENSHOT.png
Needs the dist/ export served over HTTP and internet access (Pyodide loads from cdn.jsdelivr.net).
"""

import re
import sys
import time

from playwright.sync_api import expect, sync_playwright

URL, SHOT = sys.argv[1], sys.argv[2]
FOOTPRINT = re.compile(r"Scenario footprint: ([\d.]+) kt CO2-eq")


def scenario_text(page):
    return page.get_by_text(FOOTPRINT).first.inner_text()


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1500})
    page.on("console", lambda m: m.type == "error" and print("console error:", m.text[:200]))
    start = time.monotonic()
    page.goto(URL)
    expect(page.get_by_text("Harness check: all pass")).to_be_visible(timeout=300_000)
    print(f"notebook ran in {time.monotonic() - start:.1f} s")

    results = page.locator("td").filter(has_text=re.compile(r"^(pass|FAIL)$")).all_inner_texts()
    print("check results:", results)
    assert results.count("pass") == 4 and "FAIL" not in results, results
    before = scenario_text(page)
    print("before:", before)

    slider = page.get_by_role("slider").first
    slider.focus()
    slider.press("Home")  # coefficient change -> -50 %
    expect(page.get_by_text(FOOTPRINT).first).not_to_have_text(before, timeout=60_000)
    expect(page.get_by_text("Harness check: all pass")).to_be_visible(timeout=60_000)
    after = scenario_text(page)
    print("after slider to -50 %:", after)
    results = page.locator("td").filter(has_text=re.compile(r"^(pass|FAIL)$")).all_inner_texts()
    print("check results after:", results)
    assert results.count("pass") == 4 and "FAIL" not in results, results
    assert float(FOOTPRINT.search(after)[1]) < float(FOOTPRINT.search(before)[1])

    slider.evaluate("el => el.blur()")
    page.mouse.move(0, 0)
    page.get_by_role("heading", name="4. A scenario").evaluate("el => el.scrollIntoView({block: 'start'})")
    page.wait_for_timeout(500)
    page.screenshot(path=SHOT)
    print("screenshot:", SHOT)
    browser.close()
