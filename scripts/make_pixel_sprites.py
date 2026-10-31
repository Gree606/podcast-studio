"""Turn the vector characters, ships and fleet into pixel-art sprites (frontend/sprites/*.png).

Each drawing is rendered large in a headless browser, shrunk to a small pixel grid, reduced
to a limited palette with hard (binary) transparency, and given a one-pixel dark outline,
which is what makes them read as 8/16-bit game sprites. Parts that animate (eyes, mouth,
rocket flame, saucer lights, tractor beam) are rendered as separate overlay frames that
the CSS switches in steps.

Dev tool, not used at runtime: needs Playwright + Chromium and Pillow.
    python scripts/make_pixel_sprites.py
"""
from __future__ import annotations

import asyncio
import io
import re
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "frontend" / "sprites"
SRC = (ROOT / "scripts" / "sprite_sources.svg").read_text(encoding="utf-8")
OUTLINE = (21, 18, 46, 255)
SCALE = 8  # render this many times larger than the sprite, then shrink


def symbol(sid: str) -> tuple[str, str]:
    m = re.search(rf'<symbol id="{sid}" viewBox="([^"]+)">(.*?)</symbol>', SRC, re.DOTALL)
    return m.group(1), re.sub(r"<title>.*?</title>", "", m.group(2))


def fleet(cls: str) -> tuple[str, str]:
    m = re.search(rf'<svg class="fl {re.escape(cls)}" viewBox="([^"]+)">(.*?)</svg>', SRC, re.DOTALL)
    return m.group(1), re.sub(r"<title>.*?</title>", "", m.group(2))


# name -> (viewBox, svg body, sprite width in pixels, extra CSS applied while rendering, outline?)
def jobs() -> dict[str, tuple[str, str, int, str, bool]]:
    host_vb, host = symbol("av-host")
    exp_vb, exp = symbol("av-expert")
    # sprite-only touches for Gree: long hair framing the face and a little blush (reads as "lady astronaut" at 34 px)
    exp = exp.replace('<g class="eyes">', '<path d="M39 58c-3 10-2 20 2 27l6-2c-3-7-4-15-3-24zM81 58c3 10 2 20-2 27l-6-2c3-7 4-15 3-24z" fill="#1d1d2b"/>'
                      '<circle cx="49" cy="73" r="2.6" fill="#e07a7a"/><circle cx="71" cy="73" r="2.6" fill="#e07a7a"/><g class="eyes">', 1)
    ufo_vb, ufo = symbol("ship-ufo")
    rkt_vb, rkt = symbol("ship-rocket")
    still = ".m-open{display:none}"
    # at 30 px Gree's glasses turn into noise: the sprite gets simple dot eyes instead
    no_specs = ("circle[stroke='#1d1d2b'],path[d='M58 66h4']{display:none}.eyes circle{r:4px}"
                "circle[fill='#c98e63']{fill:#b9774a}.m-closed{stroke-width:2.6px;stroke:#6b2e20}")
    j = {
        # characters: base, eyes shut, mouth open (blink/talk overlays are diffs of these)
        "alien": (host_vb, host, 40, still + "circle[fill^='var(--host-bg']{display:none}", True),
        "alien_blink": (host_vb, host + '<path d="M36 66q10 5 20 0M64 66q10 5 20 0" stroke="#15122e" stroke-width="5" fill="none"/>', 40,
                        still + "circle[fill^='var(--host-bg']{display:none}.eyes{display:none}", True),
        "alien_talk": (host_vb, host, 40, ".m-closed{display:none}circle[fill^='var(--host-bg']{display:none}", True),
        "astro": (exp_vb, exp, 44, no_specs + still + "circle[fill^='var(--expert-bg']{display:none}", True),
        "astro_blink": (exp_vb, exp + '<path d="M47 67q5 3 10 0M63 67q5 3 10 0" stroke="#1d1d2b" stroke-width="3.6" fill="none"/>', 44,
                        no_specs + still + "circle[fill^='var(--expert-bg']{display:none}.eyes{display:none}", True),
        "astro_talk": (exp_vb, exp, 44, no_specs + ".m-closed{display:none}circle[fill^='var(--expert-bg']{display:none}", True),
    }
    # a smaller Gree for the rocket's porthole (scaling the 34 px sprite to 22 px would give uneven pixels)
    for name in ("astro", "astro_blink", "astro_talk"):
        vb, body, _, css, ol = j[name]
        j[name.replace("astro", "astros")] = (vb, body, 30, css, ol)
    j.update({
        # ships: body without its animated parts, then each part on its own
        "ufo": (ufo_vb, ufo, 80, ".beam,.blink{display:none}", True),
        "ufo_lights_a": (ufo_vb, ufo, 80, "#s *:not(.blink){display:none}.b2{display:none}", False),
        "ufo_lights_b": (ufo_vb, ufo, 80, "#s *:not(.b2){display:none}", False),
        "ufo_beam": (ufo_vb, ufo, 80, "#s *:not(.beam){display:none}.beam{opacity:.5!important}", False),
        "ufo_dome": ("0 0 200 150", '<ellipse cx="100" cy="66" rx="48" ry="46" fill="none" stroke="#bfe9ff" stroke-width="5"/>'
                     '<path d="M70 40a40 40 0 0 1 22-18" stroke="#fff" stroke-width="6" fill="none" stroke-linecap="square"/>', 80, "", False),
        "rocket": (rkt_vb, rkt, 80, ".flame{display:none}", True),
        "rocket_flame": (rkt_vb, rkt, 80, "#s *:not(.flame){display:none}", False),
        "rocket_rim": ("0 0 200 150", '<circle cx="90" cy="75" r="40" fill="none" stroke="#9aa7c7" stroke-width="8"/>'
                       '<path d="M62 56a34 34 0 0 1 16-14" stroke="#fff" stroke-width="5" fill="none"/>', 80, "", False),
    })
    logo_vb, logo = symbol("logo")
    j["logo"] = (logo_vb, logo, 26, "", True)  # header icon (shown at 2x) and favicon
    # background bodies (smooth gradients in the old theme)
    j["planet"] = ("0 0 200 120", '<defs><radialGradient id="pg" cx=".35" cy=".3" r=".75"><stop offset="0" stop-color="#c4b5ff"/>'
                   '<stop offset=".5" stop-color="#6c4cf5"/><stop offset="1" stop-color="#241a6b"/></radialGradient></defs>'
                   '<ellipse cx="100" cy="62" rx="96" ry="20" fill="none" stroke="#ffd166" stroke-width="7" transform="rotate(-14 100 62)"/>'
                   '<circle cx="100" cy="60" r="50" fill="url(#pg)"/>', 72, "", True)
    j["moon"] = ("0 0 100 100", '<defs><radialGradient id="mg" cx=".3" cy=".3" r=".8"><stop offset="0" stop-color="#ffffff"/>'
                 '<stop offset=".6" stop-color="#9aa7c7"/><stop offset="1" stop-color="#4b5578"/></radialGradient></defs>'
                 '<circle cx="50" cy="50" r="46" fill="url(#mg)"/><circle cx="36" cy="40" r="9" fill="#7d88aa"/><circle cx="62" cy="62" r="6" fill="#7d88aa"/>', 26, "", True)
    for cls, w in (("sat", 40), ("ufo-s", 32), ("rocket-s", 46), ("comet", 58), ("station", 38), ("astro", 24),
                   ("rock r1", 16), ("rock r2", 14), ("rock r3", 18)):
        vb, body = fleet(cls)
        j["fl_" + cls.split()[-1].replace("-", "_")] = (vb, body, w, ".lamp{opacity:1!important}", True)
    return j


def pixelate(png: bytes, w: int, outline: bool, palette: Image.Image | None = None) -> tuple[Image.Image, Image.Image]:
    """Shrink to ``w`` pixels wide where every pixel takes the most common colour of its block
    (not the average, which smears features into in-between greys), with hard transparency."""
    big = Image.open(io.BytesIO(png)).convert("RGBA")
    h = max(1, round(big.height * w / big.width))
    big = big.resize((w * SCALE, h * SCALE), Image.LANCZOS)
    # frames of one character share the base frame's palette, or the skin tone shifts between frames
    pal = (big.convert("RGB").quantize(palette=palette, dither=Image.Dither.NONE) if palette is not None
           else big.convert("RGB").quantize(colors=16, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    colors = pal.getpalette()
    idx, alpha = pal.load(), big.getchannel("A").load()
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    px = out.load()
    for by in range(h):
        for bx in range(w):
            counts: dict[int, int] = {}
            opaque = 0
            for y in range(by * SCALE, (by + 1) * SCALE):
                for x in range(bx * SCALE, (bx + 1) * SCALE):
                    if alpha[x, y] >= 128:
                        opaque += 1
                        c = idx[x, y]
                        counts[c] = counts.get(c, 0) + 1
            if opaque * 2 >= SCALE * SCALE:  # at least half the block is drawn
                c = max(counts, key=counts.get)
                px[bx, by] = tuple(colors[c * 3:c * 3 + 3]) + (255,)
    if outline:  # a 1-pixel dark outline around the opaque shape: the classic sprite look
        px, src = out.load(), out.copy().load()
        for y in range(h):
            for x in range(w):
                if src[x, y][3]:
                    continue
                if any(0 <= x + dx < w and 0 <= y + dy < h and src[x + dx, y + dy][3]
                       for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    px[x, y] = OUTLINE
    return out, pal


def overlay(base: Image.Image, frame: Image.Image) -> Image.Image:
    """Only the pixels where ``frame`` differs from ``base`` (so it can sit on top of the base)."""
    out = Image.new("RGBA", base.size, (0, 0, 0, 0))
    b, f, o = base.load(), frame.load(), out.load()
    for y in range(base.height):
        for x in range(base.width):
            if b[x, y] != f[x, y]:
                o[x, y] = f[x, y] if f[x, y][3] else b[x, y][:3] + (0,)
    return out


async def main() -> None:
    from playwright.async_api import async_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    done: dict[str, Image.Image] = {}
    palettes: dict[str, Image.Image] = {}
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox", "--disable-gpu"])
        page = await browser.new_page()
        for name, (vb, body, w, css, outline) in jobs().items():
            _, _, vw, vh = (float(x) for x in vb.split())
            W, H = w * SCALE, round(w * SCALE * vh / vw)
            await page.set_content(f"<html><body style='margin:0;background:transparent'><style>*{{animation:none!important}}{css}</style>"
                                   f"<svg id='s' xmlns='http://www.w3.org/2000/svg' viewBox='{vb}' width='{W}' height='{H}'>{body}</svg></body></html>")
            png = await (await page.query_selector("#s")).screenshot(omit_background=True)
            base = name.split("_")[0] if name.endswith(("_blink", "_talk")) else None
            done[name], pal = pixelate(png, w, outline, palettes.get(base))
            palettes.setdefault(name, pal)
        await browser.close()
    for who in ("alien", "astro", "astros"):
        done[f"{who}_blink"] = overlay(done[who], done[f"{who}_blink"])
        done[f"{who}_talk"] = overlay(done[who], done[f"{who}_talk"])
    for name, img in done.items():
        img.save(OUT / f"{name}.png", optimize=True)
        print(f"{name:16s} {img.width}x{img.height}")


def star_tiles() -> None:
    """Three tiles of single-pixel stars (a few plus-shaped twinklers) for the parallax sky."""
    import random

    rnd = random.Random(606)
    cols = [(255, 255, 255), (191, 233, 255), (255, 209, 227), (255, 233, 176)]
    for name, size, n in (("stars_a", 96, 14), ("stars_b", 140, 9), ("stars_c", 200, 6)):
        im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        px = im.load()
        for _ in range(n):
            x, y = rnd.randrange(1, size - 1), rnd.randrange(1, size - 1)
            c = rnd.choice(cols)
            px[x, y] = c + (255,)
            if rnd.random() < .3:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    px[x + dx, y + dy] = c + (150,)
        im.save(OUT / f"{name}.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    star_tiles()
    asyncio.run(main())
