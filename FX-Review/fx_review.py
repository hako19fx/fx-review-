import sys
from datetime import datetime, timezone, timedelta

import pandas as pd
import yfinance as yf


def get_fx_live_price(ticker: str, last_close: float) -> tuple[float, str]:
    """Fetch latest 5-min intraday bar for FX (24/5 market).
    Falls back to last daily close on weekends or on error."""
    now_utc = datetime.now(timezone.utc)
    if now_utc.weekday() >= 5:          # weekend — forex closed
        return last_close, "CLOSE"
    try:
        df5 = yf.Ticker(ticker).history(period="1d", interval="5m", auto_adjust=True)
        if not df5.empty:
            p = float(df5["Close"].iloc[-1])
            if p > 0:
                return p, "LIVE"
    except Exception:
        pass
    return last_close, "CLOSE"

PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "NZD/USD": "NZDUSD=X",
}

# Central bank benchmark rates (%) — verify and update when rates change
CB_RATES = {
    "USD": 5.25,  # Federal Reserve
    "EUR": 3.65,  # ECB
    "GBP": 5.00,  # Bank of England
    "JPY": 0.25,  # Bank of Japan
    "CHF": 1.00,  # SNB
    "AUD": 4.35,  # RBA
    "CAD": 4.50,  # Bank of Canada
    "NZD": 5.25,  # RBNZ
}


def carry_positive(pair_name: str, bias: str) -> tuple[bool, float]:
    """Return (is_positive, differential) for the given pair and direction."""
    base, quote = pair_name.split("/")
    base_rate = CB_RATES.get(base, 0.0)
    quote_rate = CB_RATES.get(quote, 0.0)
    if bias == "BUY":
        diff = base_rate - quote_rate
    elif bias == "SELL":
        diff = quote_rate - base_rate
    else:
        diff = 0.0
    return diff > 0, diff


def fetch_data(ticker: str) -> tuple[pd.DataFrame, str]:
    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M JST")
    df = yf.download(ticker, period="60d", interval="1d", progress=False, auto_adjust=True)
    return df, fetched_at


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def analyze(pair_name: str, ticker: str) -> dict | None:
    df, fetched_at = fetch_data(ticker)
    if df.empty or len(df) < 20:
        return None

    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    rsi = compute_rsi(close)

    last_close = float(close.iloc[-1])
    current, price_label = get_fx_live_price(ticker, last_close)
    rsi_val = float(rsi.iloc[-1])
    ema20_val = float(ema20.iloc[-1])
    ema50_val = float(ema50.iloc[-1])

    support = float(low.tail(20).min())
    resistance = float(high.tail(20).max())
    spread = resistance - support

    above_ema20 = current > ema20_val
    above_ema50 = current > ema50_val

    if above_ema20 and above_ema50 and rsi_val > 52:
        bias = "BUY"
    elif not above_ema20 and not above_ema50 and rsi_val < 48:
        bias = "SELL"
    else:
        bias = "NEUTRAL"

    is_positive, carry_diff = carry_positive(pair_name, bias)

    decimals = 2 if "JPY" in pair_name else 4

    def fmt(v: float) -> str:
        return f"{v:.{decimals}f}"

    if bias == "BUY":
        entry = f"{fmt(current)}-{fmt(current * 1.001)}"
        target = fmt(current + spread * 0.4)
        stop = fmt(support)
    elif bias == "SELL":
        entry = f"{fmt(current * 0.999)}-{fmt(current)}"
        target = fmt(current - spread * 0.4)
        stop = fmt(resistance)
    else:
        entry = f"{fmt(support)}-{fmt(resistance)}"
        target = "-"
        stop = "-"

    return {
        "pair": pair_name,
        "price": current,
        "bias": bias,
        "rsi": rsi_val,
        "ema20": ema20_val,
        "ema50": ema50_val,
        "support": support,
        "resistance": resistance,
        "entry": entry,
        "target": target,
        "stop": stop,
        "decimals": decimals,
        "carry_positive": is_positive,
        "carry_diff": carry_diff,
        "fetched_at":   fetched_at,
        "price_label":  price_label,
    }


def print_report(results: list[dict]) -> None:
    today = datetime.now().strftime("%B %d, %Y")

    actionable = [r for r in results if r["bias"] != "NEUTRAL" and r["carry_positive"]]
    skipped = [r for r in results if r["bias"] != "NEUTRAL" and not r["carry_positive"]]
    neutral = [r for r in results if r["bias"] == "NEUTRAL"]

    print(f"\n{'='*96}")
    print(f"  FX DAILY REVIEW - {today}  (positive carry only)")
    print(f"{'='*96}")

    # Actionable summary table
    print(f"\n{'Pair':<10} {'Price':<10} {'Px':>5} {'Bias':<8} {'RSI':>5}  {'Carry':>7}  {'Entry Zone':<20} {'Target':<10} {'Stop':<10} {'Data Pulled (JST)':<20}")
    print("-" * 102)

    if actionable:
        for r in actionable:
            d = r["decimals"]
            price_str = f"{r['price']:.{d}f}"
            rsi_str = f"{r['rsi']:.1f}"
            carry_str = f"+{r['carry_diff']:.2f}%"
            print(f"{r['pair']:<10} {price_str:<10} {r.get('price_label','—'):>5} {r['bias']:<8} {rsi_str:>5}  {carry_str:>7}  {r['entry']:<20} {r['target']:<10} {r['stop']:<10} {r['fetched_at']:<20}")
    else:
        print("  No trades with positive carry today.")

    # Detailed breakdown — actionable only
    if actionable:
        print(f"\n{'='*96}")
        print("  DETAILED ANALYSIS  (positive carry trades)")
        print(f"{'='*96}")
        for r in actionable:
            d = r["decimals"]
            bias_label = {"BUY": "[BUY]", "SELL": "[SELL]"}[r["bias"]]
            base, quote = r["pair"].split("/")
            rate_base = CB_RATES.get(base, 0)
            rate_quote = CB_RATES.get(quote, 0)
            print(f"\n  {r['pair']}  {bias_label}  | Carry: +{r['carry_diff']:.2f}%  ({base} {rate_base}% vs {quote} {rate_quote}%)  | Data pulled: {r['fetched_at']}")
            print(f"  {'-'*60}")
            print(f"  Current Price  : {r['price']:.{d}f}")
            print(f"  RSI (14)       : {r['rsi']:.1f}  {'(overbought)' if r['rsi'] > 70 else '(oversold)' if r['rsi'] < 30 else ''}")
            print(f"  EMA 20         : {r['ema20']:.{d}f}  {'(above)' if r['price'] > r['ema20'] else '(below)'}")
            print(f"  EMA 50         : {r['ema50']:.{d}f}  {'(above)' if r['price'] > r['ema50'] else '(below)'}")
            print(f"  Support (20d)  : {r['support']:.{d}f}")
            print(f"  Resistance (20d): {r['resistance']:.{d}f}")
            print(f"  Entry Zone     : {r['entry']}")
            print(f"  Target         : {r['target']}")
            print(f"  Stop Loss      : {r['stop']}")

    # Skipped — negative carry
    if skipped:
        print(f"\n{'='*96}")
        print("  SKIPPED  (signal exists but carry is negative)")
        print(f"{'='*96}")
        for r in skipped:
            d = r["decimals"]
            base, quote = r["pair"].split("/")
            rate_base = CB_RATES.get(base, 0)
            rate_quote = CB_RATES.get(quote, 0)
            carry_str = f"{r['carry_diff']:.2f}%"
            print(f"  {r['pair']:<10} {r['bias']:<6}  Carry: {carry_str}  ({base} {rate_base}% vs {quote} {rate_quote}%)")

    # Neutral
    if neutral:
        print(f"\n{'='*96}")
        print("  NEUTRAL  (no clear directional signal)")
        print(f"{'='*96}")
        for r in neutral:
            d = r["decimals"]
            print(f"  {r['pair']:<10} Price: {r['price']:.{d}f}  RSI: {r['rsi']:.1f}")

    # Top 2 from positive carry trades
    top2 = sorted(actionable, key=lambda x: abs(x["rsi"] - 50), reverse=True)[:2]
    print(f"\n{'='*96}")
    print("  TOP 2 TRADE IDEAS  (positive carry)")
    print(f"{'='*96}")
    if top2:
        for i, r in enumerate(top2, 1):
            print(f"  {i}. {r['pair']} {r['bias']}  | Carry: +{r['carry_diff']:.2f}%  | Entry: {r['entry']}  | Target: {r['target']}  | Stop: {r['stop']}")
    else:
        print("  No positive carry setups today.")

    print(f"\n{'='*96}\n")


def main() -> None:
    results = []
    for pair_name, ticker in PAIRS.items():
        print(f"  Fetching {pair_name}...", end="\r", flush=True)
        result = analyze(pair_name, ticker)
        if result:
            results.append(result)
        else:
            print(f"  Warning: could not fetch data for {pair_name}", file=sys.stderr)

    print(" " * 40, end="\r")
    print_report(results)


if __name__ == "__main__":
    main()
