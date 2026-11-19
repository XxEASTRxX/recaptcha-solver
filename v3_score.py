"""reCAPTCHA v3 score harness.

v3 has no puzzle: `grecaptcha.execute()` returns a token and Google attaches a score
0.0-1.0 that the SITE's backend reads via siteverify. So the only thing worth measuring is
the score. Google's own demo page echoes the siteverify response, which gives us a real
score without owning a sitekey/secret pair.

    python v3_score.py --trials 5 --headed
    python v3_score.py --trials 5 --headless
    python v3_score.py --trials 5 --proxies-file <pool>

Each trial uses a fresh browser context (v3 scores the session, so reusing one contaminates
the next). Prints every score plus the mean and the fraction >= 0.7, which is the usual
"pass" threshold a site would apply.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from playwright.sync_api import sync_playwright

URL = "https://recaptcha-demo.appspot.com/recaptcha-v3-request-scores.php"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36")


def parse_proxy(url: str) -> dict | None:
    from urllib.parse import urlparse
    u = urlparse(url)
    if not u.hostname:
        return None
    d = {"server": f"http://{u.hostname}:{u.port or 80}"}
    if u.username:
        d["username"] = u.username
    if u.password:
        d["password"] = u.password
    return d


def one_trial(pw, headed: bool, proxy: str | None) -> dict:
    kw = {"headless": not headed,
          "args": ["--disable-blink-features=AutomationControlled"]}
    pd = parse_proxy(proxy) if proxy else None
    if pd:
        kw["proxy"] = pd
    browser = pw.chromium.launch(**kw)
    try:
        ctx = browser.new_context(user_agent=UA, locale="en-US",
                                  viewport={"width": 1280, "height": 900})
        page = ctx.new_page()
        page.goto(URL, wait_until="domcontentloaded", timeout=45000)
        # the page executes v3 on load and writes the siteverify JSON into .response
        for _ in range(30):
            txt = page.inner_text(".response") if page.query_selector(".response") else ""
            if "score" in txt:
                break
            page.wait_for_timeout(500)
        txt = page.inner_text(".response") if page.query_selector(".response") else ""
        try:
            return json.loads(txt)
        except Exception:
            return {"raw": txt[:200]}
    finally:
        browser.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--proxies-file", default=None)
    a = ap.parse_args()
    headed = a.headed or not a.headless

    pool = []
    if a.proxies_file:
        pool = [ln.strip() for ln in open(a.proxies_file, encoding="utf-8")
                if ln.strip().startswith("http")]
        print(f"proxy pool: {len(pool)} exits", flush=True)

    scores = []
    with sync_playwright() as pw:
        for i in range(a.trials):
            px = pool[i % len(pool)] if pool else None
            t = time.time()
            try:
                r = one_trial(pw, headed, px)
            except Exception as exc:  # noqa: BLE001
                print(f"trial {i+1}: ERROR {type(exc).__name__}: {str(exc)[:80]}")
                continue
            s = r.get("score")
            if s is None:
                print(f"trial {i+1}: no score ({str(r)[:120]}) {time.time()-t:.1f}s")
                continue
            scores.append(float(s))
            print(f"trial {i+1}: score={s} success={r.get('success')} "
                  f"action={r.get('action')} {time.time()-t:.1f}s", flush=True)

    if not scores:
        print("\nno scores collected")
        return 1
    mean = sum(scores) / len(scores)
    ge7 = sum(1 for s in scores if s >= 0.7)
    print(f"\nmode={'headed' if headed else 'headless'} n={len(scores)} "
          f"mean={mean:.3f} min={min(scores)} max={max(scores)} >=0.7: {ge7}/{len(scores)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
