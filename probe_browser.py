"""reCAPTCHA v2 feasibility probe: can we get a token from a headless browser, and
what does the network flow look like?

The pure-HTTP route is closed (api2/reload returns `["rresp",null,...]`), so the only
question that matters first is whether a local headless Chromium can produce a token at
all, and which calls carry the challenge.

Run:
  C:/Users/EAS/Documents/Projects VisualStudioCode/captchachalange/.venv/Scripts/python.exe probe_browser.py
"""
from __future__ import annotations

import json
import sys
import time

from playwright.sync_api import sync_playwright

DEMO = "https://www.google.com/recaptcha/api2/demo"
INTERESTING = ("/recaptcha/api2/reload", "/recaptcha/api2/payload",
               "/recaptcha/api2/userverify", "/recaptcha/enterprise/reload")


def main() -> int:
    calls: list[dict] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled",
                  "--no-sandbox"],
        )
        ctx = browser.new_context(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36"),
            locale="en-US",
            viewport={"width": 1280, "height": 900},
        )
        page = ctx.new_page()

        def on_request(req):
            if any(s in req.url for s in INTERESTING):
                body = None
                try:
                    body = req.post_data
                except Exception:
                    pass
                calls.append({"method": req.method, "url": req.url[:160],
                              "post": (body or "")[:400]})

        def on_response(resp):
            if any(s in resp.url for s in INTERESTING):
                try:
                    txt = resp.text()[:400]
                except Exception:
                    txt = "<unreadable>"
                calls.append({"resp": resp.status, "url": resp.url[:160], "body": txt})

        page.on("request", on_request)
        page.on("response", on_response)
        page.goto(DEMO, wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(2500)

        # the checkbox lives in the anchor iframe
        clicked = False
        for frame in page.frames:
            if "anchor" in (frame.url or ""):
                try:
                    frame.click("#recaptcha-anchor", timeout=8000)
                    clicked = True
                except Exception as exc:  # noqa: BLE001
                    print("anchor click failed:", type(exc).__name__, str(exc)[:100])
                break
        print("checkbox clicked:", clicked)

        token = None
        for _ in range(40):
            page.wait_for_timeout(500)
            try:
                token = page.evaluate(
                    "() => document.querySelector('#g-recaptcha-response')?.value || ''")
            except Exception:
                token = None
            if token:
                break
        print("token:", (token[:40] + "...") if token else None, "len", len(token or ""))
        print("frames:", [f.url[:90] for f in page.frames][:6])
        print("calls captured:", len(calls))
        for c in calls[:12]:
            print(json.dumps(c)[:300])
        browser.close()
    return 0 if token else 1


if __name__ == "__main__":
    sys.exit(main())