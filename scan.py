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
stock's OWN trailing-year distribution — not a fixed range width. Resistance is
a price level where several DISTINCT swing highs cluster (price hit it, was
rejected, and came back) — found by pivot clustering, never the rolling max, so
a one-way uptrend produces no level and is rejected. A hit needs BOTH: calmer
than usual AND pressed under a level it has repeatedly failed to break.

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

    # ---- structure: a real, repeatedly-tested resistance level ----
    "res_lookback":        60,      # bars searched for a tested resistance level
    "pivot_k":             3,       # a swing high needs this many lower highs each side
    "min_reject_pct":      3.0,     # ...and price must fall >=this % off the high afterward
                                    # (a real rejection, not a micro-wiggle in the coil)
    "reject_win":          6,       # bars after a peak in which that drop must occur
    "ceiling_band_pct":    2.5,     # swing highs within this % of each other are the
                                    # same resistance zone (a horizontal band / mild slope)
    "min_ceiling_tests":   2,       # zone needs at least this many DISTINCT rejections
    "min_pivot_gap":       5,       # rejections must be >=this many bars apart to count
                                    # separately (not two touches from one push)
    "ceiling_setup_frac":  0.5,     # first rejection must land in the first fraction of the
                                    # window, so the level pre-exists today's approach
    "min_test_span_frac":  0.33,    # first->last rejection must span >=this fraction of the
                                    # window, so the level was tested OVER TIME, not at once
    "max_closes_above":    1,       # price must have FAILED to break: at most this many
                                    # closes above the level within the window

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


def swing_highs(highs, lows, p):
    """Indices of swing-high pivots that represent a genuine REJECTION.

    A pivot is a bar whose high is a strict local peak with ``pivot_k`` lower
    highs on each side AND from which price then fell at least ``min_reject_pct``
    within the next ``reject_win`` bars. The rejection-depth requirement is what
    separates a real test-and-fail from the shallow micro-wiggles inside a quiet
    coil (which would otherwise masquerade as a repeatedly-tested level)."""
    n = len(highs)
    k = p["pivot_k"]
    mr = p["min_reject_pct"] / 100.0
    rw = p["reject_win"]
    piv = []
    for i in range(k, n - k):
        h = highs[i]
        if not (h >= highs[i - k:i + k + 1].max() and h > highs[i - 1] and h > highs[i + 1]):
            continue
        fwd = lows[i + 1:min(n, i + 1 + rw)]
        if len(fwd) and (h - fwd.min()) / h >= mr:      # price was rejected down from here
            piv.append(i)
    return piv


def _distinct(indices, min_gap):
    """Collapse pivots that sit within ``min_gap`` bars of each other so two
    touches from the SAME push aren't counted as two separate rejections."""
    out = []
    for i in sorted(indices):
        if not out or i - out[-1] >= min_gap:
            out.append(i)
    return out


def resistance_zones(highs, lows, p):
    """Find horizontal resistance zones — price levels where several DISTINCT
    swing highs (genuine rejections) cluster within ``ceiling_band_pct``.

    This is deliberately NOT the rolling max of a window: an uptrend's swing
    highs ascend and never cluster, so a one-way grind produces no zone and is
    rejected. A genuine ceiling is a level price hit, got rejected at, pulled
    back from, and hit again — exactly what a cluster of separated pivots is.

    Returns a list of ``{level, tests, first, last}`` dicts, most-tested first.
    """
    piv = swing_highs(highs, lows, p)
    if len(piv) < p["min_ceiling_tests"]:
        return []
    band = p["ceiling_band_pct"] / 100.0
    zones = []
    for a in piv:
        lvl = highs[a]
        members = _distinct([b for b in piv if abs(highs[b] - lvl) <= band * lvl],
                            p["min_pivot_gap"])
        if len(members) >= p["min_ceiling_tests"]:
            top = max(float(highs[b]) for b in members)
            zones.append({"level": top, "tests": len(members),
                          "first": members[0], "last": members[-1]})
    # dedup zones whose levels coincide, keeping the most-tested
    uniq = []
    for z in sorted(zones, key=lambda z: (-z["tests"], -z["level"])):
        if not any(abs(z["level"] - u["level"]) <= band * u["level"] for u in uniq):
            uniq.append(z)
    return uniq


def evaluate(symbol, df, p):
    """Run the two-phase test on the LAST bar.

    Returns a hit dict tagged with a ``status`` of either ``"breakout"`` (the
    signal fired today) or ``"coiling"`` (still building a quiet base under the
    ceiling), or None when the base is not a real consolidation.
    """
    n = p["base_len"]
    if df is None or len(df) < max(n, p["res_lookback"]) + p["min_vol_baseline_bars"] + 20:
        return None

    df = df.copy()
    df["atrp"] = atr_pct(df)

    today = df.iloc[-1]                      # candidate breakout / latest bar
    prev  = df.iloc[-2]                       # yesterday (for a fresh-cross test)
    base  = df.iloc[-(n + 1):-1]             # the n bars BEFORE today

    avg_base_vol   = base["volume"].mean()
    last_close     = float(today["close"])
    prev_close     = float(prev["close"])
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

    # ---- structure: a real, repeatedly-tested resistance level ----
    #   Look back over res_lookback bars for a price level where several DISTINCT
    #   swing highs cluster — a level price hit and was rejected at more than
    #   once. The level is fixed by those historical rejections, so it can NOT
    #   drift up with the trend the way a rolling max does.
    m       = p["res_lookback"]
    res_win = df.iloc[-(m + 1):-1]           # window before today
    highs   = res_win["high"].values
    lows    = res_win["low"].values
    closes  = res_win["close"].values
    span    = len(highs)

    def qualifies(z):
        # established early, and the rejections span time (not clustered at the end)
        if z["first"] > span * p["ceiling_setup_frac"]:          return False
        if (z["last"] - z["first"]) < span * p["min_test_span_frac"]: return False
        # price FAILED to break: it never closed above the level within the window
        if int((closes > z["level"]).sum()) > p["max_closes_above"]: return False
        return True

    zones = [z for z in resistance_zones(highs, lows, p) if qualifies(z)]
    if not zones:
        return None

    vol_mult = last_vol / avg_base_vol if avg_base_vol else 0.0

    # ---- classify: firing breakout vs still-coiling under the resistance ----
    #   BREAKOUT : today's close clears a tested level that yesterday's close was
    #              still below (a fresh cross), on expanded volume. Requiring the
    #              cross to be TODAY rejects names that broke out days ago.
    #   COILING  : price is pressed just under the nearest overhead tested level.
    fresh = [z for z in zones
             if prev_close <= z["level"] < last_close and vol_mult >= p["vol_mult"]]
    overhead = [z for z in zones if z["level"] >= last_close]

    if fresh:
        chosen = max(fresh, key=lambda z: z["level"])
        status = "breakout"
    elif overhead:
        chosen = min(overhead, key=lambda z: z["level"])   # nearest resistance above
        if (chosen["level"] - last_close) / chosen["level"] * 100.0 > p["near_ceiling_pct"]:
            return None
        status = "coiling"
    else:
        return None

    base_high           = chosen["level"]
    n_ceiling_tests     = chosen["tests"]
    breakout_pct        = (last_close - base_high) / base_high * 100.0
    dist_to_ceiling_pct = (base_high - last_close) / base_high * 100.0

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
