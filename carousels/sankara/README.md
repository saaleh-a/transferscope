# Sankara Carousel

**10-slide Instagram carousel · 1080 × 1350 px · @s7.exe**

A constructivist-typographic carousel about Thomas Sankara and the four years
(1983–87) he led Burkina Faso. No mention of Traoré. Copy is intentionally
tight, concrete, and varied in rhythm.

---

## Files

```
carousels/sankara/
├── index.html      Carousel viewer (open in any browser)
├── styles.css      All visual styles
├── photo/          Drop your images here (see below)
└── README.md       This file
```

---

## Viewing

Open `index.html` in Chrome or Firefox. The slides are shown at 45 % scale
inside the browser for comfortable viewing. Use:

- **← / →** arrow keys (or on-screen buttons) to navigate
- **P** key (or "Toggle Photos" button) to switch between the no-photo and
  photo variants

---

## Exporting slides

### Option A — Screenshot per slide (recommended)

1. Open `index.html` in Chrome.
2. Open DevTools → toggle device toolbar (`Ctrl+Shift+M` / `Cmd+Shift+M`).
3. Set a **custom size of 1080 × 1350**.
4. Set zoom to **100%** in the browser address bar.
5. In DevTools Console, paste:

   ```js
   document.querySelector('.slide-window').style.width  = '1080px';
   document.querySelector('.slide-window').style.height = '1350px';
   document.querySelector('.slide-stage').style.transform = 'none';
   ```

6. Navigate to each slide and use **Capture screenshot** in DevTools
   (⋮ menu → Capture screenshot, or `Ctrl+Shift+P` → "Capture full size").
7. Each screenshot is ready for Instagram at exactly 1080 × 1350.

### Option B — Print to PDF

Print the page (`Ctrl+P` / `Cmd+P`) with:

- Paper size: **Custom** → 1080 px × 1350 px (or 28.575 cm × 35.56 cm)
- Margins: **None**
- Background graphics: **ON**

Each slide prints on its own page. You can then extract pages as PNG with
any PDF viewer or tool (e.g. Adobe Acrobat, `pdftoppm`, ImageMagick).

---

## Photo vs. no-photo variant

By default the carousel uses **bold typography + geometric shapes** as the
primary visual (no photos required). To activate the photo overlay:

- Click **"Toggle Photos"** in the viewer, or press **P**.
- Place your images in the `photo/` subdirectory, named as below.

| Slide | Filename            | Notes |
|-------|---------------------|-------|
| 01    | `photo/s1-cover.jpg`| Portrait of Sankara (face, 2:3 ratio). Should be high-contrast. |
| 06    | `photo/s6-trees.jpg`| Sahel landscape or reforestation. Desaturated works best. |

Only slides 1 and 6 have photo slots in this build. Images render at 35–55 %
opacity so typography remains legible. You can add more photo slots by
following the pattern in the HTML:

```html
<div class="photo-slot"
     data-caption="Your caption here"
     style="top:0;left:0;width:540px;height:600px;">
  <img src="photo/your-image.jpg" alt="Description" />
  <span class="photo-caption">Your caption here</span>
</div>
```

The `.photo-slot` is hidden by default and shown only when `<body>` has the
class `photo-mode`.

### Where to find public-domain / CC images

- **Wikimedia Commons** — search "Thomas Sankara" — several press photos from
  the 1980s are in the public domain.
- **Internet Archive** — documentary footage stills.
- **ECOWAS / AU photo archives** — sometimes open-licensed.

Always verify the license before use. Attribution to the original author
should appear in the `alt` text or caption.

---

## Design tokens

| Token          | Value     | Usage |
|----------------|-----------|-------|
| `--black`      | `#0A0A0A` | Slide backgrounds |
| `--red`        | `#C8102E` | Accent blocks, stamps, list marks |
| `--gold`       | `#F5C518` | The one gold line per slide; highlighted words |
| `--offwhite`   | `#F0EDE4` | Primary text; display type |
| `--grey`       | `#808080` | Secondary text; eyebrows; footer |
| `--font-display` | Barlow Condensed | All display / headline type |
| `--font-body`  | DM Sans         | Body copy, captions |

---

## The gold-line rule

**Every slide has exactly one gold (`#F5C518`) element.** This is a strict
design constraint: one gold line, or one gold highlighted word — not both.
It creates visual rhythm across the set. Do not add extra gold elements when
customising.

---

## Typography notes

- **Barlow Condensed 900** — display headlines (slide titles, large stats)
- **Barlow Condensed 700** — labels, eyebrows, sub-headings
- **DM Sans 300** — body copy (light weight keeps density readable)
- **DM Sans 500** — inline emphasis within body copy

Line lengths are kept short by design. Instagram reads on mobile; walls of
text kill engagement.

---

## Customising copy

All copy is in `index.html`, inline within each `.slide` block. Each slide
is clearly commented (`SLIDE 01`, `SLIDE 02`, etc.). Edit the HTML directly.

Keep the copy:
- **Concrete** — specific numbers, dates, names
- **Short** — 3–5 words per line in display text; 2–3 sentences in body
- **Varied** — mix long and very short sentences; avoid triads
- **Active** — prefer verbs over nouns

---

## Adding or removing slides

1. Copy a `.slide` block from `index.html`.
2. Change the class to the next number (e.g. `slide-11`).
3. Add corresponding CSS in `styles.css` under a new `SLIDE 11` section.
4. Update the `TOTAL` constant in the JavaScript (`var TOTAL = 10;`).
5. Update all `footer-count` spans.

---

*Created for @s7.exe. No mention of Traoré.*
