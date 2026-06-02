Perform a live daily FX review by running the data fetcher, then present and interpret the results.

## Step 1 — Run the live data script

Execute the following command and capture its output:

```
uv run python FX-Review/fx_review.py
```

Wait for it to complete. The script fetches 60 days of OHLCV data for 7 major pairs from Yahoo Finance and computes RSI(14), EMA20, EMA50, support/resistance, and a BUY/SELL/NEUTRAL bias.

## Step 2 — Present the output

Print the full script output exactly as returned.

## Step 3 — Add qualitative commentary

After the output, add a short section:

### Market Context
- 2-3 sentences on the current macro backdrop (USD direction, risk sentiment, any major central bank themes)
- Flag any known high-impact events today (NFP, CPI, FOMC, ECB, BoE, RBA decisions) that could override the technical bias
- Note if any pair has conflicting signals (e.g., BUY bias but RSI > 70 = caution)

### Risk Reminder
Always remind the user: *"These are technical hints only — not financial advice. Always use proper risk management and confirm with your own analysis before trading."*
