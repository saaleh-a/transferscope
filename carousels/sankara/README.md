# Sankara Carousel

10-slide Instagram carousel (1080 × 1350 px) on Thomas Sankara.

**Account:** @s7.exe  
**Constraint:** No mention of Traoré. Copy is deliberately spare and specific.

---

## Files

| File | Purpose |
|---|---|
| `index.html` | All 10 slides in one page |
| `styles.css` | Design system — import via `<link>` in the HTML |

---

## Exporting slides

### Option A — Chrome DevTools screenshot (recommended)

1. Open `index.html` in **Chrome** (drag-and-drop into the address bar).
2. Open DevTools → go to the **Elements** panel.
3. Right-click any `.slide` div in the Elements tree → **Capture node screenshot**
   - This exports exactly 1080 × 1350 px as a PNG.
   - Repeat for all 10 slides.

### Option B — Print to PDF (per-slide)

1. Temporarily comment out all slides except one in `index.html`.
2. File → Print → **Save as PDF** → set paper size to **1080 × 1350** (custom) → no margins.
3. Repeat for each slide.

### Option C — Puppeteer (batch export)

```bash
npm install puppeteer
node export.js   # see snippet below
```

```js
// export.js — requires Node + Puppeteer
const puppeteer = require('puppeteer');
const path = require('path');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  await page.setViewport({ width: 1080, height: 1350 });
  await page.goto('file://' + path.resolve('index.html'));

  const slides = await page.$$('.slide');
  for (let i = 0; i < slides.length; i++) {
    await slides[i].screenshot({ path: `slide-${String(i + 1).padStart(2, '0')}.png` });
  }

  await browser.close();
  console.log('Done — 10 PNGs exported.');
})();
```

---

## Swapping in photos

Each slide that supports a photo has a `<div class="photo-slot">` inside it.
To activate photos:

1. Add `with-photo` to the slide's class list:

```html
<!-- Before -->
<div class="slide slide-1">

<!-- After -->
<div class="slide slide-1 with-photo">
```

2. Place your image inside the `.photo-slot` div:

```html
<div class="photo-slot" data-label="…">
  <img src="photos/sankara-portrait.jpg" alt="Thomas Sankara" />
</div>
```

3. The `.photo-overlay` gradient is applied automatically — it keeps text readable over any photo.

### Sourcing public-domain / CC images

The following archives carry usable images. Always verify the licence before publishing.

| Source | Search term | Notes |
|---|---|---|
| [Wikimedia Commons](https://commons.wikimedia.org) | `Thomas Sankara` | Several PD/CC-BY images, mainly from official state photography |
| [Internet Archive](https://archive.org) | `Sankara 1983 Burkina` | Digitised press photos, check each item's rights statement |
| [Getty Images — editorial](https://www.gettyimages.com) | `Thomas Sankara` | Editorial use only; not for commercial carousel posts |
| [Magnum Photos](https://www.magnumphotos.com) | `Sankara` | Licenced; contact for social use |

If no verified public-domain photo is available, the **no-photo variant** (default) works entirely through typography and graphic system — no visual quality is lost.

---

## Design system

### Color tokens

| Token | Hex | Usage |
|---|---|---|
| `--ink` | `#0A0A0A` | Primary dark background / dark text |
| `--red` | `#C8102E` | Accent marks, diagonal cuts, bullet markers |
| `--gold` | `#F5C518` | Gold rule (one per slide), kickers, @s7.exe attribution |
| `--cream` | `#F0EDE4` | Light-slide backgrounds, primary light text |
| `--grey` | `#808080` | Labels, captions, subdued body copy |

### Typefaces

| Role | Font | Weight |
|---|---|---|
| Display / headline | Barlow Condensed | 900 (black), 700 (bold) |
| Body / captions | DM Sans | 300 (light), 400 (regular), 500 (medium) |

Both served from Google Fonts CDN; swap to local `.woff2` files for offline use.

### The gold rule

Every slide carries **exactly one** horizontal gold line (`<div class="gold-rule">`).
It anchors the eye and reinforces the visual rhythm across all 10 slides.
Do not add a second gold rule to a slide — that breaks the system.

### Dark / light alternation

- Dark slides (`background: #0A0A0A`): 1, 3, 5, 7, 9  
- Light slides (`.light` class, `background: #F0EDE4`): 2, 4, 6, 8, 10  

The alternation gives visual breathing room when swiping in the Instagram feed.

### Diagonal cuts

Red or gold triangles are applied with `clip-path: polygon()` on `.diag-block` elements.
They add tension without text distortion and stay readable at all sizes.

### Halftone texture

The `.slide::before` pseudo-element overlays a `radial-gradient` dot grid at 16% opacity.
This breaks the flat look without adding noise or reducing contrast.

### Safe margins

Instagram crops ~5% on all edges when displayed in feed.
Content is kept inside `--mg-x: 56px` (horizontal) and `--mg-y: 72px` (vertical) gutters.
No critical text or graphics sit outside these bounds.

---

## Copy notes

- No mention of Traoré anywhere.
- All statistics sourced from: UNICEF Burkina Faso historical reports, OAU 1987 summit records, FAO West Africa food security data, and academic biographies (Murrey 2018; Harsch 2014).
- Dates and figures should be verified against primary sources before publishing.
