Run the Japan stock screener and provide daily NISA and Tokutei picks.

## Step 1 — Run the screener

```
uv run python japan-stocks/japan_stocks.py
```

## Step 2 — Present the output

Print the full report as returned by the script.

## Step 3 — Add brief market context

After the output, add:

### Japan Market Context
- 2-3 sentences on Nikkei 225 / TOPIX direction today
- Any relevant macro events (BoJ policy, USD/JPY level, US market impact on Japanese equities)
- Flag if any AI/semiconductor theme stocks have specific news today

### Risk Reminder
*"These are screening hints only — not financial advice. Always verify with your broker and check the latest news before placing trades in your NISA or Tokutei account."*
