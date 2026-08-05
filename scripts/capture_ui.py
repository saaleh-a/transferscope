"""Capture TransferScope UI screenshots for design review."""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(r"C:\Users\t-sabanur\transferscope\uishots")
OUT.mkdir(exist_ok=True)

PORT = sys.argv[1] if len(sys.argv) > 1 else "8513"

NAV = [
    ("transfer_impact", "Transfer Impact"),
    ("shortlist", "Shortlist Generator"),
    ("hot_or_not", "Hot or Not"),
    ("backtest", "Backtest Validator"),
    ("diagnostics", "Diagnostics"),
    ("about", "About & Methodology"),
]


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.goto(f"http://localhost:{PORT}", wait_until="networkidle", timeout=120_000)
        page.wait_for_timeout(8000)

        errors = page.locator('[data-testid="stAlert"]:has-text("Failed to render")')
        if errors.count():
            print("  !! render error on load:", errors.first.inner_text()[:400])

        for name, label in NAV:
            try:
                page.get_by_role("radio", name=label).click(timeout=15_000)
            except Exception:
                try:
                    page.locator(f'label:has-text("{label}")').first.click(timeout=10_000)
                except Exception as exc:  # noqa: BLE001
                    print(f"  !! nav to {label} failed: {type(exc).__name__}")
                    continue
            # Wait for Streamlit to finish the rerun rather than a fixed sleep.
            page.wait_for_timeout(1500)
            try:
                page.wait_for_selector('[data-testid="stStatusWidget"]', state="detached", timeout=45_000)
            except Exception:
                pass
            page.wait_for_timeout(2500)

            err = page.locator('[data-testid="stAlert"]:has-text("Failed to render")')
            if err.count():
                print(f"  !! {label} render error:", err.first.inner_text()[:300])

            path = OUT / f"{name}.png"
            page.screenshot(path=str(path), full_page=True)
            print(f"  saved {path.name} ({path.stat().st_size // 1024} KB)")

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
