#!/usr/bin/env python3
"""
Consolidation-Breakout Scanner
------------------------------
Two-phase detection over the full US equity market:

  Phase 1  CONSOLIDATION  the base is unusually quiet vs the stock's own
                          long-term normal, pressed under a repeatedly-tested
                          ceiling
  Phase 2  BREAKOUT       today's CLOSE clears the base ceiling on expanded volume

The scanner reports TWO states so you can watch a setup before it triggers:

  BREAKOUT  today's close cleared the ceiling on expanded volume  (signal fires)
  COILING   still inside a quiet base, pressed up under the         (building the
            repeatedly-tested ceiling and poised to break            range)

Consolidation is measured as volatility (ATR%) ranking in the quiet tail of the
stock's OWN trailing-year distribution — not a fixed range width. A base counts
only if it is calmer than usual AND its ceiling has been tapped multiple times.

Universe : Nasdaq Trader symbol files (nasdaqlisted.txt + otherlisted.txt)
Bars     : Yahoo Finance v8 chart endpoint (key-free)
Output   : results.json  (consumed by index.html)

Signal fires on the close of the breakout candle.
"""

import io
import json
import time
import random
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# TUNABLE PARAMETERS  (also surfaced in the dashboard header for reference)
# ----------------------------------------------------------------------------
PARAMS = {
    "base_len":            20,     # trading days in the consolidation base (pre-breakout)

    # ---- consolidation: the base must be QUIET vs the stock's OWN long-term normal ----
    "vol_baseline_bars":   252,    # trailing window (~1yr) that defines "normal" volatility
    "max_vol_pctile":      0.25,   # base ATR% must rank in the quietest this-fraction of the
                                    # baseline: 0.25 = bottom quartile of its own year
    "min_vol_baseline_bars": 120,  # need at least this much history to judge "normal"

    # ---- breakout trigger + coil zone ----
    "vol_mult":            1.5,     # breakout volume >= vol_mult * avg base volume
    "near_ceiling_pct":    4.0,     # COILING: how far below the ceiling still counts as
                                    # "pressed up under resistance" (building the range)

    # ---- structure: a real, repeatedly-tested ceiling ----
    "ceiling_prox_pct":    2.0,     # a bar "tests" the ceiling if its high is within this %
    "min_ceiling_tests":   2,       # resistance must be tapped at least this many times
    "ceiling_setup_frac":  0.6,     # first ceiling test must land within this fraction of
                                    # the base, so resistance is retested — not first reached
                                    # on the final push up (keeps "repeatedly" honest)

    "min_price":           10.0,    # liquidity floor: last close
    "min_avg_vol":         300_000, # liquidity floor: avg base volume (shares)
    "lookback_days":       420,     # calendar history pulled per symbol (~285 bars, enough
                                    # for a 252-bar volatility baseline)
    "chart_days_out":      70,      # bars of OHLCV embedded per hit for the mini-chart
}

# ----------------------------------------------------------------------------
# NETWORK
# ----------------------------------------------------------------------------
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})

NASDAQ_FILES = [
    ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "Symbol"),
    ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",  "ACT Symbol"),
]


def get_universe():
    """Return a clean list of common-stock tickers (ETFs / test / derivative
    classes removed), formatted for Yahoo (. -> -)."""
    symbols = set()
    for url, sym_col in NASDAQ_FILES:
        r = SESSION.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep="|")
        df = df[df.iloc[:, 0].notna()]
        # drop footer row "File Creation Time..."
        df = df[~df[sym_col].astype(str).str.contains("File Creation", na=False)]
        if "Test Issue" in df.columns:
            df = df[df["Test Issue"] != "Y"]
        if "ETF" in df.columns:
            df = df[df["ETF"] != "Y"]
        for s in df[sym_col].astype(str):
            s = s.strip()
            # skip warrants/units/preferred (suffix markers) and empty
            if not s or any(c in s for c in ["$", " "]):
                continue
            if len(s) > 6:
                continue
            symbols.add(s.replace(".", "-"))
    return sorted(symbols)


def fetch_bars(symbol, lookback_days, retries=3):
    """Fetch daily OHLCV from Yahoo. Returns a DataFrame or None."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": f"{lookback_days}d", "interval": "1d"}
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(1.5 * (attempt + 1) + random.random())
                continue
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            ts = res.get("timestamp")
            if not ts:
                return None
            q = res["indicators"]["quote"][0]
            df = pd.DataFrame({
                "date":   pd.to_datetime(ts, unit="s").tz_localize(None).normalize(),
                "open":   q["open"],
                "high":   q["high"],
                "low":    q["low"],
                "close":  q["close"],
                "volume": q["volume"],
            }).dropna()
            return df.reset_index(drop=True)
        except Exception:
            time.sleep(0.6 * (attempt + 1) + random.random())
    return None


# ----------------------------------------------------------------------------
# DETECTION
# ----------------------------------------------------------------------------
def atr_pct(df):
    """Average True Range as % of close, per row."""
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=7).mean()
    return atr / c * 100.0


def evaluate(symbol, df, p):
    """Run the two-phase test on the LAST bar.

    Returns a hit dict tagged with a ``status`` of either ``"breakout"`` (the
    signal fired today) or ``"coiling"`` (still building a quiet base under the
    ceiling), or None when the base is not a real consolidation.
    """
    n = p["base_len"]
    if df is None or len(df) < n + p["min_vol_baseline_bars"] + 20:
        return None

    df = df.copy()
    df["atrp"] = atr_pct(df)

    today = df.iloc[-1]                      # candidate breakout / latest bar
    base  = df.iloc[-(n + 1):-1]             # the n bars BEFORE today

    base_high = float(base["high"].max())
    if base_high <= 0:
        return None

    avg_base_vol   = base["volume"].mean()
    last_close     = float(today["close"])
    last_vol       = float(today["volume"])

    # ---- liquidity gates ----
    if last_close < p["min_price"]:                 return None
    if avg_base_vol < p["min_avg_vol"]:             return None

    # ---- consolidation: base volatility quiet vs the stock's OWN normal ----
    #   Measure = ATR% (price-normalized, so it's comparable across time). We
    #   rank the base window's average ATR% against the distribution of every
    #   base-length window over the trailing ~year. A low percentile means the
    #   stock is calmer than it usually is — a genuine volatility contraction,
    #   not merely a range that happens to look tight.
    vol_now = float(base["atrp"].mean())
    if not np.isfinite(vol_now):                    return None
    hist = df["atrp"].iloc[:-1].rolling(n).mean().dropna()
    hist = hist.iloc[-p["vol_baseline_bars"]:]
    if len(hist) < p["min_vol_baseline_bars"]:      return None
    vol_pctile = float((hist < vol_now).mean())     # 0 = quietest in its baseline
    if vol_pctile > p["max_vol_pctile"]:            return None

    # ---- structure: a real, retested ceiling (established early, tapped >=2x) ----
    prox_line   = base_high * (1 - p["ceiling_prox_pct"] / 100.0)
    tests_mask  = base["high"].values >= prox_line
    n_ceiling_tests = int(tests_mask.sum())
    first_test  = int(tests_mask.argmax()) if n_ceiling_tests else n
    if n_ceiling_tests < p["min_ceiling_tests"]:                    return None
    if first_test > (n - 1) * p["ceiling_setup_frac"]:             return None

    vol_mult            = last_vol / avg_base_vol if avg_base_vol else 0.0
    breakout_pct        = (last_close - base_high) / base_high * 100.0
    dist_to_ceiling_pct = (base_high - last_close) / base_high * 100.0

    # ---- classify: firing breakout vs still-coiling base ----
    if last_close > base_high and vol_mult >= p["vol_mult"]:
        status = "breakout"          # Phase 2: cleared ceiling on expanded volume
    elif last_close <= base_high and dist_to_ceiling_pct <= p["near_ceiling_pct"]:
        status = "coiling"           # Phase 1: pressed up under the ceiling, no trigger yet
    else:
        return None

    tail = df.tail(p["chart_days_out"])
    return {
        "symbol":          symbol,
        "status":          status,
        "date":            today["date"].strftime("%Y-%m-%d"),
        "close":           round(last_close, 2),
        "base_high":       round(base_high, 2),          # the tested ceiling
        # Part 1 — volatility quiet vs the stock's own normal
        "vol_pctile":      round(vol_pctile * 100, 1),
        # Part 2 — repeatedly-tested resistance
        "ceiling_tests":   n_ceiling_tests,
        # Breakout on volume
        "vol_mult":        round(vol_mult, 2),
        "breakout_pct":    round(breakout_pct, 2),
        "dist_to_ceiling_pct": round(dist_to_ceiling_pct, 2),
        "chart": {
            "dates":  [d.strftime("%Y-%m-%d") for d in tail["date"]],
            "close":  [round(float(x), 2) for x in tail["close"]],
            "volume": [int(x) for x in tail["volume"]],
            "high":   [round(float(x), 2) for x in tail["high"]],
            "low":    [round(float(x), 2) for x in tail["low"]],
            "base_high": round(base_high, 2),
        },
    }


def scan_symbol(symbol, p):
    df = fetch_bars(symbol, p["lookback_days"])
    try:
        return evaluate(symbol, df, p)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main(limit=None, workers=10):
    p = PARAMS
    universe = get_universe()
    if limit:
        universe = universe[:limit]
    print(f"Universe: {len(universe)} symbols")

    hits, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(scan_symbol, s, p): s for s in universe}
        for fut in as_completed(futs):
            done += 1
            h = fut.result()
            if h:
                hits.append(h)
            if done % 250 == 0:
                print(f"  {done}/{len(universe)} scanned, {len(hits)} hits")

    # Firing breakouts first (loudest volume on top), then coiling bases
    # ordered by how close they are pressed under the ceiling.
    def sort_key(h):
        if h.get("status") == "breakout":
            return (0, -h["vol_mult"])
        return (1, h.get("dist_to_ceiling_pct", 999))
    hits.sort(key=sort_key)

    breakout_count = sum(1 for h in hits if h.get("status") == "breakout")
    coiling_count  = len(hits) - breakout_count

    out = {
        "generated_utc": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "params": p,
        "universe_size": len(universe),
        "hit_count": len(hits),
        "breakout_count": breakout_count,
        "coiling_count": coiling_count,
        "hits": hits,
    }
    with open("results.json", "w") as f:
        json.dump(out, f)
    print(f"Done. {len(hits)} hits ({breakout_count} breakout, "
          f"{coiling_count} coiling) written to results.json")
    return out


if __name__ == "__main__":
    import sys
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=lim)
