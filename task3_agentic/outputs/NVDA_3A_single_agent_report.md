# NVDA research brief (2026-10-11)

## 1. Financial Health Summary

NVDA has rallied +13.19% over the past three months and is trading comfortably above both its 50‑day (221.94) and 200‑day (201.83) moving averages, indicating a sustained uptrend. Technical momentum remains positive with RSI at 53 (neutral) and a modestly positive MACD histogram (+0.12). Volatility is low for the year – 30‑day annualised volatility is 29.45%, placing it in the 6th percentile of the past 12 months, while the 1‑day 95 % VaR is –3.88%.

- 3‑month return: +13.19%
- Last close: $229.28
- SMA‑50: $221.94
- SMA‑200: $201.83
- RSI‑14: 53.0
- MACD histogram: 0.1185
- Bollinger %B: 0.5811
- 30‑day annualised volatility: 29.45%
- 1‑day 95 % VaR: –3.88%
- 1‑year max drawdown: –20.21%

**Market sentiment:** LLM sentiment analysis of recent headlines yields a modestly positive score (+0.192) with 7 bullish versus 5 bearish mentions, reflecting overall optimism tempered by visible institutional selling.

## 2. Top Three Risks (next 90 days)

### 1. Earnings surprise risk (high)

NVDA’s next earnings report is slated for 26 Aug 2026 (Q2 FY27) with analysts forecasting $93‑$95 B revenue; a miss or weaker guidance could trigger a sharp sell‑off in the next 90 days.

Evidence:
- Earnings date: 26 Aug 2026
- Revenue forecast $93‑$95 B
Sources: web_search (https://alphio.ai/blog/nvda-earnings-august-2026)

### 2. Institutional sell pressure (medium)

Multiple large investors have disclosed recent sales of NVDA shares, which could add downward pressure on the stock if the selling continues.

Evidence:
- Osterweis Capital Management Inc. Sells 6,347 Shares
- NVIDIA Corporation $NVDA Stock Sold by Primecap Management Co.
- 268,305 NVIDIA Shares Sold by The Manufacturers Life Insurance Company
Sources: get_news

### 3. Supply‑chain concentration risk (medium)

NVDA’s heavy reliance on TSMC and a few key suppliers makes it vulnerable to geopolitical tensions and fab bottlenecks that could delay deliveries and compress margins.

Evidence:
- NVIDIA Geopolitical Supply Chain Risk Assessment – highlights geopolitical fragmentation and Taiwan concentration
Sources: web_search (https://nova.kapualabs.com/nvda-stock_panel_c/nvidias-geopolitical-supply-chain-a-definitive-risk-assessment/)

## 3. Hedge Strategy Recommendation

**Buy a protective put option** using 3‑month NVDA put, strike ~5% OTM (~$217).

With 30‑day annualised volatility at 29.45% (low) and a 1‑day 95 % VaR of –3.88%, a 5% OTM put caps downside from earnings miss or sell‑off while preserving upside from the current uptrend.

Data points:
- 30‑day annualised volatility: 29.45%
- 1‑day 95 % VaR: –3.88%
Sizing / trigger: Buy 1 contract (100 shares) per 1000 shares held; strike set at $217 (≈5% below current $229.28) with expiration aligned to the earnings date (late Aug)

Sources: get_price_data; calculate_volatility; get_news; llm_sentiment; web_search

*Generated automatically for educational purposes. Not investment advice. Data may be delayed or incomplete, and LLM output can be wrong.*