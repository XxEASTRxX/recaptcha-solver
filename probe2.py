"""Probe 2: capture the reCAPTCHA challenge itself.

Probe 1 proved a headless Chromium reaches the challenge (reload returns a real rresp,
bframe opens, /payload is fetched). This saves the payload bytes and dumps the bframe
DOM so we know exactly what a solver must answer: how many tiles, their image URLs, and
how the answer is submitted.
"""
from __future__ import annotations

import base64
import json
import os
import sys

from playwright.sync_api import sync_playwright

DEMO = "https://www.google.com/recaptcha/api2/demo"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "capture")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    payloads: list[bytes] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True,
                                     args=["--disable-blink-features=AutomationControlled"])
        ctx = browser.new_context(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36"),
            locale="en-US", viewport={"width": 1280, "height": 900})
        page = ctx.new_page()

        def on_response(resp):
            if "/recaptcha/api2/payload" in resp.url:
                try:
                    payloads.append(resp.body())
                except Exception:
                    pass

        page.on("response", on_response)
        page.goto(DEMO, wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(2500)
        for frame in page.frames:
            if "anchor" in (frame.url or ""):
                try:
                    frame.click("#recaptcha-anchor", timeout=8000)
                except Exception:
                    pass
                break
        page.wait_for_timeout(6000)

        bf = next((f for f in page.frames if "/bframe" in (f.url or "")), None)
        print("bframe present:", bf is not None)
        if payloads:
            p = payloads[0]
            print("payload bytes:", len(p), "head:", p[:16])
            open(os.path.join(OUT, "payload.bin"), "wb").write(p)
        if bf is not None:
            html = bf.content()
            open(os.path.join(OUT, "bframe.html"), "w", encoding="utf-8").write(html)
            info = bf.evaluate("""() => {
                const tiles = [...document.querySelectorAll('td[role=button], .rc-imageselect-tile')];
                const imgs = [...document.querySelectorAll('img')].map(i => i.src).slice(0, 20);
                const hdr = document.querySelector('.rc-imageselect-desc, .rc-imageselect-desc-no-canonical');
                const dyn = document.querySelector('#rc-imageselect');
                return JSON.stringify({
                  tiles: tiles.length,
                  imgs: imgs.slice(0, 12),
                  header: hdr ? hdr.innerText.slice(0, 160) : null,
                  dyn: dyn ? dyn.outerHTML.slice(0, 400) : null,
                  tables: document.querySelectorAll('table').length,
                  buttons: [...document.querySelectorAll('button')].map(b => (b.id || b.className || '').slice(0, 40))
                });
            }""")
            print("bframe info:", info[:1200])
        print("payloads captured:", len(payloads))
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())