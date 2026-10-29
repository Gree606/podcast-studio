"""Render the app's hover/click sound variants to WAV files for a listening test.

Runs the real frontend code (SFX_VOICES in frontend/app.js) in a headless browser with an
OfflineAudioContext, so the files sound exactly like the buttons. Each file plays the sound
three times, as you would hear it moving across a row or clicking a few buttons.

Dev tool (needs Playwright + Chromium and the app running):
    python scripts/render_sfx.py http://127.0.0.1:8000 data/listen/sfx
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys
from pathlib import Path

RENDER = """async ([kind, name]) => {
  const sr = 44100, ctx = new OfflineAudioContext(1, Math.ceil(sr * 1.6), sr);
  for (const t of [0.15, 0.6, 1.05]) SFX_VOICES[kind][name](ctx, t, ctx.destination, 1);
  const d = (await ctx.startRendering()).getChannelData(0);
  const buf = new ArrayBuffer(44 + d.length * 2), v = new DataView(buf);
  const w = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF"); v.setUint32(4, 36 + d.length * 2, true); w(8, "WAVEfmt "); v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, "data"); v.setUint32(40, d.length * 2, true);
  for (let i = 0; i < d.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, d[i])) * 32767, true);
  let bin = ""; const bytes = new Uint8Array(buf);
  for (let i = 0; i < bytes.length; i += 8192) bin += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return btoa(bin);
}"""


async def main(url: str, out: Path) -> None:
    from playwright.async_api import async_playwright

    out.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox", "--disable-gpu"])
        page = await browser.new_page()
        await page.goto(url)
        await page.wait_for_function("typeof SFX_VOICES === 'object'")
        voices = await page.evaluate("Object.fromEntries(Object.entries(SFX_VOICES).map(([k, v]) => [k, Object.keys(v)]))")
        for kind, names in voices.items():
            for name in names:
                (out / f"{kind}_{name}.wav").write_bytes(base64.b64decode(await page.evaluate(RENDER, [kind, name])))
                print(f"{kind}_{name}.wav")
        await browser.close()
    print(json.dumps(voices))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000", Path(sys.argv[2] if len(sys.argv) > 2 else "data/listen/sfx")))
