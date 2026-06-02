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
    "Mitsubishi Corp":       "8058.T",
    "Mitsui & Co":           "8031.T",
    "Itochu":                "8001.T",
    "Sumitomo Corp":         "8053.T",
    "Tokio Marine":          "8766.T",
    "Orix":                  "8591.T",
    "Fujifilm Holdings":     "4901.T",
    "Asahi Group":           "2502.T",
    "Ajinomoto":             "2802.T",
    "Kao":                   "4452.T",
    "Shiseido":              "4911.T",
    "TDK":                   "6762.T",
    "Kyocera":               "6971.T",
    "Secom":                 "9735.T",
    "Nikon":                 "7731.T",
    "MS&AD Insurance":       "8725.T",
    "Japan Exchange Group":  "8697.T",
    "SBI Holdings":          "8473.T",
    "Kakaku.com":            "2371.T",
    "Trend Micro":           "4704.T",
}

AI_GROWTH = {
    "SoftBank Group":        "9984.T",   # ARM / AI investment
    "Tokyo Electron":        "8035.T",   # semiconductor equipment
    "Advantest":             "6857.T",   # semiconductor testing
    "Lasertec":              "6920.T",   # EUV inspection
    "Disco":                 "6146.T",   # semiconductor dicing
    "Renesas Electronics":   "6723.T",   # microcontrollers / edge AI
    "Fujitsu":               "6702.T",   # enterprise AI / cloud
    "Hitachi":               "6501.T",   # AI / IoT / digital
    "NEC":                   "6701.T",   # facial recognition / AI
    "SoftBank Corp":         "9434.T",   # 5G / AI infrastructure
    "M3":                    "2413.T",   # healthcare AI
    "CyberAgent":            "4751.T",   # AI advertising / media
    "PKSHA Technology":      "3993.T",   # pure-play AI software
    "Mercari":               "4385.T",   # AI-powered marketplace
    "Panasonic Holdings":    "6752.T",   # AI / EV battery
    "Zozo":                  "3092.T",   # AI fashion tech
    "Money Forward":         "3994.T",   # fintech AI
    "Appier Group":          "4180.T",   # AI marketing
    "Sansan":                "4443.T",   # AI business card/CRM
    "Freee K.K.":            "4478.T",   # cloud ERP / AI
}

ALL_STOCKS = {**NIKKEI_225, **AI_GROWTH}

AI_TICKERS = set(AI_GROWTH.values())


# ── Technical helpers ─────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def fix_div_yield(raw: float) -> float:
    """yfinance returns dividend yield inconsistently for JP stocks.
    If value > 0.20 it is already in % form (e.g. 3.44 means 3.44%).
    Otherwise it is a decimal (e.g. 0.0344 means 3.44%).
    Cap at 20% to filter bad data."""
    if raw is None:
        return 0.0
    pct = raw if raw > 0.20 else raw * 100
    return min(round(pct, 2), 20.0)


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
        ema5  = close.ewm(span=5,  adjust=False).mean()
        rsi   = compute_rsi(close)

        # Use live price if available (real-time during TSE hours), else last close
        try:
            live = float(tk.fast_info.last_price)
            current = live if live and live > 0 else float(close.iloc[-1])
            price_label = "LIVE"
        except Exception:
            current = float(close.iloc[-1])
            price_label = "CLOSE"
        rsi_val    = float(rsi.iloc[-1])
        ema5_val   = float(ema5.iloc[-1])
        ema20_val  = float(ema20.iloc[-1])
        ema50_val  = float(ema50.iloc[-1])
        support    = float(low.tail(20).min())
        resistance = float(high.tail(20).max())
        high_90d   = float(high.max())

        # 5-day and 20-day price change
        ret5  = (current / float(close.iloc[-6])  - 1) * 100 if len(close) >= 6  else 0.0
        ret20 = (current / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0.0

        # Drawdown from 90-day high (recovery potential)
        drawdown = (current / high_90d - 1) * 100

        # Fundamentals
        info      = tk.info
        pe_ratio  = info.get("trailingPE") or info.get("forwardPE")
        div_yield = fix_div_yield(info.get("dividendYield") or 0)
        sector    = info.get("sector", "—")
        mkt_cap   = info.get("marketCap")
        mkt_cap_s = (f"{mkt_cap/1e12:.1f}T" if mkt_cap and mkt_cap >= 1e12
                     else f"{mkt_cap/1e9:.0f}B" if mkt_cap else "—")

        is_ai = ticker in AI_TICKERS

        # Fundamental quality score (0-5)
        fund_score = 0
        if pe_ratio:
            if pe_ratio < 15:   fund_score += 3
            elif pe_ratio < 25: fund_score += 2
            elif pe_ratio < 40: fund_score += 1
        if div_yield > 3:   fund_score += 2
        elif div_yield > 1: fund_score += 1

        # Dip score: how oversold (0+ points)
        dip_score = max(0.0, 45.0 - rsi_val)

        # Tokutei momentum score: rewards short-term bounce + dip depth
        momentum_score = (
            dip_score * 1.5 +
            fund_score +
            (ret5 * 0.5 if ret5 > 0 else ret5 * 0.2) +   # reward recovery, penalise continued fall
            abs(drawdown) * 0.1                             # more upside room = better
        )

        return {
            "name":           name,
            "ticker":         ticker,
            "price":          current,
            "price_label":    price_label,
            "rsi":            rsi_val,
            "ema5":           ema5_val,
            "ema20":          ema20_val,
            "ema50":          ema50_val,
            "support":        support,
            "resistance":     resistance,
            "high_90d":       high_90d,
            "drawdown":       drawdown,
            "ret5":           ret5,
            "ret20":          ret20,
            "pe":             pe_ratio,
            "div_yield":      div_yield,
            "sector":         sector,
            "mkt_cap":        mkt_cap_s,
            "is_ai":          is_ai,
            "dip_score":      dip_score,
            "fund_score":     fund_score,
            "momentum_score": momentum_score,
            "fetched_at":     fetched_at,
        }
    except Exception as e:
        print(f"  Warning: {name} ({ticker}) — {e}", file=sys.stderr)
        return None


# ── Screening ─────────────────────────────────────────────────────────────────

def screen(stocks: list[dict], n: int = 10) -> dict:
    nisa_dip    = []
    nisa_ai     = []
    tokutei_mom = []

    for s in stocks:
        rsi = s["rsi"]
        pe  = s["pe"]

        # NISA — dip + strong fundamentals (quality long-term holds)
        if rsi < 50 and s["fund_score"] >= 1 and (pe is None or pe < 40):
            nisa_dip.append(s)

        # NISA — AI/growth theme (not overbought)
        if s["is_ai"] and rsi < 65:
            nisa_ai.append(s)

        # Tokutei — short-term momentum + recovery (RSI < 65, ranked by momentum score)
        if rsi < 65:
            tokutei_mom.append(s)

    nisa_dip    = sorted(nisa_dip,    key=lambda x: x["dip_score"] + x["fund_score"] * 1.5, reverse=True)[:n]
    nisa_ai     = sorted(nisa_ai,     key=lambda x: x["momentum_score"],                    reverse=True)[:n]
    tokutei_mom = sorted(tokutei_mom, key=lambda x: x["momentum_score"],                    reverse=True)[:n]

    # Top 20 dividend stocks: yield >= 1.5%, P/E < 30, RSI < 65
    # Score: yield * 2 + fund_score + dip bonus (good entry) + growth bonus (positive 20d return)
    dividends = [
        s for s in stocks
        if s["div_yield"] >= 1.5
        and (s["pe"] is None or s["pe"] < 30)
        and s["rsi"] < 65
    ]
    def div_score(s: dict) -> float:
        return (
            s["div_yield"] * 2.0
            + s["fund_score"] * 1.5
            + (3.0 if s["rsi"] < 45 else 1.5 if s["rsi"] < 55 else 0.0)  # dip bonus
            + (2.0 if s["ret20"] > 5 else 1.0 if s["ret20"] > 0 else 0.0)  # growth bonus
            + (1.0 if s["div_yield"] > 3 else 0.0)                          # high yield bonus
        )
    dividends = sorted(dividends, key=div_score, reverse=True)[:20]

    return {"nisa_dip": nisa_dip, "nisa_ai": nisa_ai, "tokutei_mom": tokutei_mom, "dividends": dividends}


# ── Report helpers ────────────────────────────────────────────────────────────

def fmt_pe(v) -> str:
    return f"{v:.1f}x" if v else "—"

def fmt_div(v: float) -> str:
    return f"{v:.2f}%" if v else "—"

def fmt_price(v: float) -> str:
    return f"\xa5{v:,.0f}"

def fmt_pct(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1f}%"


def print_section(title: str, label: str, stocks: list[dict]) -> None:
    W = 100
    print(f"\n{'='*W}")
    print(f"  {title}")
    print(f"{'='*W}")
    if not stocks:
        print("  No qualifying stocks today.")
        return

    hdr = (f"{'#':<3} {'Stock':<26} {'Code':<8} {'Price':>9} {'Px':>5} {'RSI':>5} "
           f"{'5d':>6} {'P/E':>7} {'Div':>6}  {'Drawdown':>9}  {'Data Pulled (JST)':<20}")
    print(f"\n{hdr}")
    print("-" * W)

    for i, s in enumerate(stocks, 1):
        ai_tag = "[AI] " if s["is_ai"] else "     "
        name   = (s["name"][:25]).ljust(26)
        print(
            f"{i:<3} {ai_tag}{name[5:] if s['is_ai'] else name:<26} {s['ticker']:<8} "
            f"{fmt_price(s['price']):>9} {s.get('price_label','—'):>5} {s['rsi']:>5.1f} "
            f"{fmt_pct(s['ret5']):>6} {fmt_pe(s['pe']):>7} {fmt_div(s['div_yield']):>6}  "
            f"{fmt_pct(s['drawdown']):>9}  {s['fetched_at']:<20}"
        )

    print(f"\n{'-'*W}")
    print("  DETAIL")
    print(f"{'-'*W}")

    for i, s in enumerate(stocks, 1):
        spread  = s["resistance"] - s["support"]
        entry   = fmt_price(s["price"] * 0.999)
        target  = fmt_price(s["price"] + spread * 0.4)
        stop    = fmt_price(s["support"])
        ai_tag  = " [AI]" if s["is_ai"] else ""
        a20     = "(above)" if s["price"] > s["ema20"] else "(below)"
        a50     = "(above)" if s["price"] > s["ema50"] else "(below)"
        print(f"\n  {i}. {s['name']}{ai_tag} ({s['ticker']})  [{label}]  |  {s['sector']}  |  Cap: {s['mkt_cap']}")
        print(f"     Price: {fmt_price(s['price'])} [{s.get('price_label','—')}]  RSI: {s['rsi']:.1f}  "
              f"5d: {fmt_pct(s['ret5'])}  20d: {fmt_pct(s['ret20'])}  "
              f"Drawdown from 90d-high: {fmt_pct(s['drawdown'])}")
        print(f"     EMA20: {fmt_price(s['ema20'])} {a20}  |  EMA50: {fmt_price(s['ema50'])} {a50}")
        print(f"     P/E: {fmt_pe(s['pe'])}  Div: {fmt_div(s['div_yield'])}  |  "
              f"Support: {fmt_price(s['support'])}  Resistance: {fmt_price(s['resistance'])}")
        print(f"     Entry: {entry}  Target: {target}  Stop: {stop}")


def print_dividend_section(stocks: list[dict]) -> None:
    W = 100
    print(f"\n{'='*W}")
    print("  TOP 20 DIVIDEND STOCKS  [Quality + Growth Potential]")
    print(f"  Criteria: Yield >= 1.5%  |  P/E < 30  |  RSI < 65  |  Ranked by yield + quality + dip entry")
    print(f"{'='*W}")
    if not stocks:
        print("  No qualifying dividend stocks today.")
        return

    hdr = (f"{'#':<3} {'Stock':<26} {'Code':<8} {'Price':>9} {'Px':>5} {'Div Yield':>10} "
           f"{'RSI':>5} {'P/E':>7} {'5d':>6} {'20d':>6}  {'Drawdown':>9}  {'Data Pulled (JST)':<20}")
    print(f"\n{hdr}")
    print("-" * W)

    for i, s in enumerate(stocks, 1):
        ai_tag = "[AI]" if s["is_ai"] else "    "
        name   = s["name"][:25].ljust(26)
        print(
            f"{i:<3} {ai_tag} {name:<26} {s['ticker']:<8} "
            f"{fmt_price(s['price']):>9} {s.get('price_label','—'):>5} {fmt_div(s['div_yield']):>10} "
            f"{s['rsi']:>5.1f} {fmt_pe(s['pe']):>7} "
            f"{fmt_pct(s['ret5']):>6} {fmt_pct(s['ret20']):>6}  "
            f"{fmt_pct(s['drawdown']):>9}  {s['fetched_at']:<20}"
        )

    print(f"\n{'-'*W}")
    print("  DETAIL")
    print(f"{'-'*W}")

    for i, s in enumerate(stocks, 1):
        spread = s["resistance"] - s["support"]
        entry  = fmt_price(s["price"] * 0.999)
        target = fmt_price(s["price"] + spread * 0.4)
        stop   = fmt_price(s["support"])
        a20    = "(above)" if s["price"] > s["ema20"] else "(below)"
        a50    = "(above)" if s["price"] > s["ema50"] else "(below)"
        ai_tag = " [AI]" if s["is_ai"] else ""
        growth = ("Growing" if s["ret20"] > 5 else "Stable" if s["ret20"] > 0 else "Under pressure")
        print(f"\n  {i}. {s['name']}{ai_tag} ({s['ticker']})  |  {s['sector']}  |  Cap: {s['mkt_cap']}  |  Trend: {growth}")
        print(f"     Price: {fmt_price(s['price'])} [{s.get('price_label','—')}]  Div: {fmt_div(s['div_yield'])}  P/E: {fmt_pe(s['pe'])}  RSI: {s['rsi']:.1f}")
        print(f"     5d: {fmt_pct(s['ret5'])}  20d: {fmt_pct(s['ret20'])}  Drawdown: {fmt_pct(s['drawdown'])}")
        print(f"     EMA20: {fmt_price(s['ema20'])} {a20}  |  EMA50: {fmt_price(s['ema50'])} {a50}")
        print(f"     Support: {fmt_price(s['support'])}  Resistance: {fmt_price(s['resistance'])}")
        print(f"     Entry: {entry}  Target: {target}  Stop: {stop}")


def print_report(results: dict) -> None:
    today = datetime.now().strftime("%B %d, %Y  %H:%M JST")
    W = 100

    print(f"\n{'='*W}")
    print(f"  JAPAN STOCK DAILY PICKS - {today}")
    print(f"  Universe: Nikkei 225 + TOPIX + AI/Growth  |  10 picks per account type")
    print(f"{'='*W}")

    print_section(
        "NISA (Growth) -- Dip + Strong Fundamentals  [Long-term quality holds]",
        "NISA", results["nisa_dip"]
    )
    print_section(
        "NISA (Growth) -- AI / High-Growth Theme  [Long-term growth potential]",
        "NISA", results["nisa_ai"]
    )
    print_section(
        "TOKUTEI -- Short-term Momentum + Recovery Plays  [Loss offset / active trading]",
        "TOKUTEI", results["tokutei_mom"]
    )

    print_dividend_section(results["dividends"])

    # Top 5 overall
    seen    = set()
    combined = results["nisa_dip"] + results["tokutei_mom"] + results["nisa_ai"]
    unique  = [s for s in combined if not (s["ticker"] in seen or seen.add(s["ticker"]))]
    top5    = sorted(unique, key=lambda x: x["momentum_score"] + (3 if x["is_ai"] else 0), reverse=True)[:5]

    print(f"\n{'='*W}")
    print("  TOP 5 PICKS TODAY  (best risk/reward across all categories)")
    print(f"{'='*W}")
    for i, s in enumerate(top5, 1):
        ai_tag = " [AI]" if s["is_ai"] else ""
        in_nisa = s in results["nisa_dip"] or s in results["nisa_ai"]
        in_tok  = s in results["tokutei_mom"]
        acct    = ("NISA + Tokutei" if in_nisa and in_tok
                   else "NISA" if in_nisa else "Tokutei")
        spread  = s["resistance"] - s["support"]
        target  = fmt_price(s["price"] + spread * 0.4)
        print(f"  {i}. {s['name']}{ai_tag} ({s['ticker']})  [{acct}]  "
              f"{fmt_price(s['price'])}  RSI:{s['rsi']:.1f}  "
              f"5d:{fmt_pct(s['ret5'])}  Drawdown:{fmt_pct(s['drawdown'])}  "
              f"Target:{target}  Stop:{fmt_price(s['support'])}")

    print(f"\n{'='*W}")
    print("  Note: Not financial advice. Verify with your broker before trading.")
    print(f"{'='*W}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    stocks = []
    total  = len(ALL_STOCKS)
    for i, (name, ticker) in enumerate(ALL_STOCKS.items(), 1):
        print(f"  Fetching {name} ({ticker})... [{i}/{total}]", end="\r", flush=True)
        result = fetch_stock(name, ticker)
        if result:
            stocks.append(result)

    print(" " * 70, end="\r")

    if not stocks:
        print("Error: could not fetch any stock data.", file=sys.stderr)
        sys.exit(1)

    results = screen(stocks, n=10)
    print_report(results)


if __name__ == "__main__":
    main()
