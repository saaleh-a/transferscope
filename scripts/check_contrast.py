"""Check WCAG contrast ratios for the Tactical Noir palette."""
from __future__ import annotations


def _lin(c: float) -> float:
    c = c / 255.0
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def ratio(fg: str, bg: str) -> float:
    a, b = luminance(fg), luminance(bg)
    lo, hi = min(a, b), max(a, b)
    return (hi + 0.05) / (lo + 0.05)


BACKGROUNDS = {"bg_primary": "#0E1117", "bg_secondary": "#161B22", "bg_card": "#1C2128"}

FOREGROUNDS = {
    "text_primary": "#C9D1D9",
    "text_secondary": "#8B949E",
    "text_muted": "#484F58",
    "accent_gold": "#D4A843",
    "accent_amber": "#E3A507",
    "accent_crimson": "#F45B69",
    "accent_green": "#2DD4A8",
    "accent_blue": "#58A6FF",
    "accent_teal": "#39D2C0",
}

print(f"{'foreground':<18} " + " ".join(f"{b:>14}" for b in BACKGROUNDS))
print("-" * 68)
for fname, fhex in FOREGROUNDS.items():
    cells = []
    for bhex in BACKGROUNDS.values():
        r = ratio(fhex, bhex)
        flag = "OK " if r >= 4.5 else ("lg " if r >= 3.0 else "FAIL")
        cells.append(f"{r:>9.2f} {flag:<4}")
    print(f"{fname:<18} " + " ".join(cells))

print()
print("AA normal text needs 4.5:1;  AA large text (>=18.66px bold / 24px) needs 3.0:1")
