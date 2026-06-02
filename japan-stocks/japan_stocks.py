import sys
from datetime import datetime

import pandas as pd
import yfinance as yf

# ── Stock Universe ────────────────────────────────────────────────────────────

NIKKEI_225 = {
    "Toyota Motor":          "7203.T",
    "Sony Group":            "6758.T",
    "Keyence":               "6861.T",
    "Fast Retailing":        "9983.T",
    "Shin-Etsu Chemical":    "4063.T",
    "Recruit Holdings":      "6098.T",
    "Hoya":                  "7741.T",
    "Daiichi Sankyo":        "4568.T",
    "Mitsubishi UFJ FG":     "8306.T",
    "KDDI":                  "9433.T",
    "NTT":                   "9432.T",
    "Sumitomo Mitsui FG":    "8316.T",
    "Mizuho Financial":      "8411.T",
    "Honda Motor":           "7267.T",
    "Daikin Industries":     "6367.T",
    "Komatsu":               "6301.T",
    "Canon":                 "7751.T",
    "Nippon Steel":          "5401.T",
    "Nintendo":              "7974.T",
    "Takeda Pharmaceutical": "4502.T",
    "Chugai Pharmaceutical": "4519.T",
    "Oriental Land":         "4661.T",
    "Eisai":                 "4523.T",
    "Kubota":                "6326.T",
    "Nidec":                 "6594.T",
    "Murata Manufacturing":  "6981.T",
    "Denso":                 "6902.T",
    "Japan Post Holdings":   "6178.T",
    "Seven & i Holdings":    "3382.T",
    "Olympus":               "7733.T",
    "Nomura Holdings":       "8604.T",
    "Shimano":               "7309.T",
    "Fanuc":                 "6954.T",
}

AI_GROWTH = {
    "SoftBank Group":        "9984.T",   # ARM / AI investment
    "Tokyo Electron":        "8035.T",   # semiconductor equipment
    "Advantest":             "6857.T",   # semiconductor testing
    "Lasertec":              "6920.T",   # EUV inspection
    "Disco":                 "6146.T",   # semiconductor precision cutting
    "Renesas Electronics":   "6723.T",   # microcontrollers / edge AI
    "Fujitsu":               "6702.T",   # enterprise AI / cloud
    "Hitachi":               "6501.T",   # AI / IoT / digital
    "NEC":                   "6701.T",   # facial recognition / AI solutions
    "SoftBank Corp":         "9434.T",   # 5G / AI infrastructure
    "M3":                    "2413.T",   # healthcare AI
    "CyberAgent":            "4751.T",   # AI advertising / media
    "PKSHA Technology":      "3993.T",   # pure-play AI software
    "Mercari":               "4385.T",   # AI-powered marketplace
    "Panasonic Holdings":    "6752.T",   # AI / EV battery
    "Zozo":                  "3092.T",   # AI fashion tech
}

ALL_STOCKS = {**NIKKEI_225, **AI_GROWTH}


# ── Technical helpers ─────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def fetch_stock(name: str, ticker: str) -> dict | None:
    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M JST")
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period="90d", interval="1d", auto_adjust=True)
        if df.empty or len(df) < 20:
            return None

        close = df["Close"].squeeze()
        high  = df["High"].squeeze()
        low   = df["Low"].squeeze()

        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        rsi   = compute_rsi(close)

        current   = float(close.iloc[-1])
        rsi_val   = float(rsi.iloc[-1])
        ema20_val = float(ema20.iloc[-1])
        ema50_val = float(ema50.iloc[-1])
        support   = float(low.tail(20).min())
        resistance = float(high.tail(20).max())

        # Fundamentals
        info = tk.info
        pe_ratio   = info.get("trailingPE") or info.get("forwardPE")
        div_yield  = (info.get("dividendYield") or 0) * 100
        sector     = info.get("sector", "—")
        mkt_cap    = info.get("marketCap")
        mkt_cap_b  = f"{mkt_cap / 1e12:.1f}T" if mkt_cap and mkt_cap >= 1e12 else (f"{mkt_cap / 1e9:.0f}B" if mkt_cap else "—")

        is_ai = ticker in AI_GROWTH.values()

        # Dip strength: RSI below 45 = recent dip
        dip_score = max(0, 45 - rsi_val)

        # Fundamental score
        fund_score = 0
        if pe_ratio and pe_ratio < 20:  fund_score += 3
        elif pe_ratio and pe_ratio < 30: fund_score += 2
        elif pe_ratio and pe_ratio < 40: fund_score += 1
        if div_yield > 2:  fund_score += 2
        elif div_yield > 1: fund_score += 1

        return {
            "name":       name,
            "ticker":     ticker,
            "price":      current,
            "rsi":        rsi_val,
            "ema20":      ema20_val,
            "ema50":      ema50_val,
            "support":    support,
            "resistance": resistance,
            "pe":         pe_ratio,
            "div_yield":  div_yield,
            "sector":     sector,
            "mkt_cap":    mkt_cap_b,
            "is_ai":      is_ai,
            "dip_score":  dip_score,
            "fund_score": fund_score,
            "fetched_at": fetched_at,
        }
    except Exception as e:
        print(f"  Warning: {name} ({ticker}) — {e}", file=sys.stderr)
        return None


# ── Screening ─────────────────────────────────────────────────────────────────

def screen(stocks: list[dict]) -> dict:
    nisa_dip    = []
    nisa_ai     = []
    tokutei_dip = []

    for s in stocks:
        rsi = s["rsi"]
        pe  = s["pe"]

        # NISA — dip + strong fundamentals
        if rsi < 45 and s["fund_score"] >= 2 and (pe is None or pe < 35):
            nisa_dip.append(s)

        # NISA — AI/growth theme (not overbought)
        if s["is_ai"] and rsi < 55:
            nisa_ai.append(s)

        # Tokutei — sharp dip, any quality stock
        if rsi < 40:
            tokutei_dip.append(s)

    # Sort: dip score for dip plays, rsi distance for AI
    nisa_dip    = sorted(nisa_dip,    key=lambda x: x["dip_score"] + x["fund_score"], reverse=True)[:5]
    nisa_ai     = sorted(nisa_ai,     key=lambda x: abs(x["rsi"] - 30), reverse=True)[:5]
    tokutei_dip = sorted(tokutei_dip, key=lambda x: x["dip_score"] + x["fund_score"], reverse=True)[:5]

    return {"nisa_dip": nisa_dip, "nisa_ai": nisa_ai, "tokutei_dip": tokutei_dip}


# ── Report ────────────────────────────────────────────────────────────────────

def fmt_pe(v) -> str:
    return f"{v:.1f}x" if v else "—"

def fmt_div(v: float) -> str:
    return f"{v:.2f}%" if v else "—"

def fmt_price(v: float) -> str:
    return f"¥{v:,.0f}"

def print_section(title: str, stocks: list[dict], show_div: bool = True) -> None:
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    if not stocks:
        print("  No qualifying stocks today.")
        return
    header = f"{'Stock':<24} {'Code':<8} {'Price':>9} {'RSI':>5} {'P/E':>7} {'Div':>6}  {'Support':>9} {'Resist':>9}  {'Data Pulled (JST)':<20}"
    print(f"\n{header}")
    print("-" * 90)
    for s in stocks:
        ai_tag = " [AI]" if s["is_ai"] else ""
        name   = s["name"][:23] + ai_tag
        print(
            f"{name:<24} {s['ticker']:<8} {fmt_price(s['price']):>9} {s['rsi']:>5.1f} "
            f"{fmt_pe(s['pe']):>7} {fmt_div(s['div_yield']):>6}  "
            f"{fmt_price(s['support']):>9} {fmt_price(s['resistance']):>9}  {s['fetched_at']:<20}"
        )
    print()
    for s in stocks:
        ai_tag = " [AI]" if s["is_ai"] else ""
        above_ema20 = "(above)" if s["price"] > s["ema20"] else "(below)"
        above_ema50 = "(above)" if s["price"] > s["ema50"] else "(below)"
        spread = s["resistance"] - s["support"]
        entry  = fmt_price(s["price"] * 0.999)
        target = fmt_price(s["price"] + spread * 0.4)
        stop   = fmt_price(s["support"])
        print(f"  {s['name']}{ai_tag} ({s['ticker']})  |  Sector: {s['sector']}  |  Mkt Cap: {s['mkt_cap']}")
        print(f"  {'-'*50}")
        print(f"  Price      : {fmt_price(s['price'])}   RSI: {s['rsi']:.1f}{'  (oversold)' if s['rsi'] < 30 else ''}")
        print(f"  EMA 20     : {fmt_price(s['ema20'])}  {above_ema20}")
        print(f"  EMA 50     : {fmt_price(s['ema50'])}  {above_ema50}")
        print(f"  P/E        : {fmt_pe(s['pe'])}   Dividend: {fmt_div(s['div_yield'])}")
        print(f"  Support    : {fmt_price(s['support'])}   Resistance: {fmt_price(s['resistance'])}")
        print(f"  Entry      : {entry}   Target: {target}   Stop: {stop}")
        print()


def print_report(results: dict, all_stocks: list[dict]) -> None:
    today = datetime.now().strftime("%B %d, %Y  %H:%M JST")

    print(f"\n{'='*90}")
    print(f"  JAPAN STOCK DAILY PICKS - {today}")
    print(f"  Universe: Nikkei 225 + TOPIX / AI-Growth Theme")
    print(f"{'='*90}")

    print_section("NISA (Growth) -- Dip + Strong Fundamentals", results["nisa_dip"])
    print_section("NISA (Growth) -- AI / Growth Theme",         results["nisa_ai"], show_div=False)
    print_section("TOKUTEI -- Sharp Dip Recovery Plays",        results["tokutei_dip"])

    # Top 3 overall
    combined = results["nisa_dip"] + results["nisa_ai"] + results["tokutei_dip"]
    seen     = set()
    unique   = [s for s in combined if not (s["ticker"] in seen or seen.add(s["ticker"]))]
    top3     = sorted(unique, key=lambda x: x["dip_score"] + x["fund_score"] + (5 if x["is_ai"] else 0), reverse=True)[:3]

    print(f"\n{'='*90}")
    print("  TOP 3 PICKS TODAY")
    print(f"{'='*90}")
    acct = {True: "NISA + Tokutei", False: "Tokutei"}
    for i, s in enumerate(top3, 1):
        tag  = " [AI]" if s["is_ai"] else ""
        acct_label = "NISA + Tokutei" if s in results["nisa_dip"] or s in results["nisa_ai"] else "Tokutei"
        spread = s["resistance"] - s["support"]
        target = fmt_price(s["price"] + spread * 0.4)
        print(f"  {i}. {s['name']}{tag} ({s['ticker']})  |  {acct_label}  |  {fmt_price(s['price'])}  RSI:{s['rsi']:.1f}  Target:{target}  Stop:{fmt_price(s['support'])}")

    print(f"\n{'='*90}")
    print("  Note: Not financial advice. Verify with your broker before trading.")
    print(f"{'='*90}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    stocks = []
    total  = len(ALL_STOCKS)
    for i, (name, ticker) in enumerate(ALL_STOCKS.items(), 1):
        print(f"  Fetching {name} ({ticker})... [{i}/{total}]", end="\r", flush=True)
        result = fetch_stock(name, ticker)
        if result:
            stocks.append(result)

    print(" " * 60, end="\r")

    if not stocks:
        print("Error: could not fetch any stock data.", file=sys.stderr)
        sys.exit(1)

    results = screen(stocks)
    print_report(results, stocks)


if __name__ == "__main__":
    main()
