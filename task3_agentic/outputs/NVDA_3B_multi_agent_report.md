# NVDA research brief (2026-10-11)

## 1. Financial Health Summary

NVDA’s share price closed at $229.28, up 13.2% over the past three months, and remains above both the 50‑ and 200‑day moving averages, indicating a sustained uptrend. The 90‑day annualised volatility of 38.5% sits at the 65th percentile of the past year, signalling moderate risk. The 1‑day 95% VaR of 3.88% translates to a potential $8.9 loss per share. Overall sentiment is neutral, with mixed institutional activity.

- price 229.28
- 3mo return 13.19%
- 52w high 243.37
- 52w low 163.90
- pct from 52w high -5.79%
- 90‑day vol 38.51%
- 1‑day 95% VaR -3.88%

**Market sentiment:** Neutral sentiment from recent headlines; institutional selling and buying are roughly balanced, suggesting no strong directional bias.

## 2. Top Three Risks (next 90 days)

### 1. Intensifying competition in the AI‑accelerator market (high)

New competitors such as Google TPUs, AMD MI300, and Intel Habana chips are narrowing performance gaps and increasing price sensitivity, potentially eroding NVDA’s market share and margins within 90 days.

Evidence:
- Google TPUs gain traction (MarketBeat forecast)
- Competition for talent and rising costs (TipRanks risk factor)
Sources: https://www.tipranks.com/stocks/nvda/risk-factors, https://www.marketbeat.com/stocks/NASDAQ/NVDA/forecast/

### 2. Export‑control and geopolitical risk (high)

U.S. export restrictions on China and other high‑tech jurisdictions could limit NVDA’s ability to sell GPUs and data‑center chips, and may disrupt supply chains, impacting revenue and earnings in the near term.

Evidence:
- Export controls and geopolitical risks listed in FY 2026 10‑K
- Potential revenue impact noted in risk factors
Sources: https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm

### 3. Supply‑chain and manufacturing risk (medium)

NVDA relies heavily on TSMC for advanced process nodes; any disruption—natural disaster, geopolitical tension, or capacity constraints—could delay product launches and hurt earnings over the next 90 days.

Evidence:
- Supply‑chain disruptions cited in 10‑K
- Foundry capacity constraints noted as risk
Sources: https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm

## 3. Hedge Strategy Recommendation

**3‑month put option collar** using Buy 3‑month put with strike $210 (≈9% below current price) and sell 3‑month call with strike $240.

The put limits downside exposure from competition, export‑control, and supply‑chain risks, while the call generates premium to offset the put cost. The strike levels are chosen to cap loss at roughly (229.28‑210)/229.28 ≈ 8.3% plus premium, aligning with the 1‑day 95% VaR of 3.88%.

Data points:
- current price 229.28
- 90‑day vol 38.51%
- 1‑day 95% VaR -3.88%
Sizing / trigger: Hold one put per share; trigger if price falls below $210 within 90 days

**How the analyst's clarification was used:** Agent A’s sentiment analysis confirmed a neutral overall tone (score 0.074) and balanced institutional activity, supporting the neutral market sentiment assessment in the report.

Sources: calculate_volatility; get_price_data; llm_sentiment; web_search

*Generated automatically for educational purposes. Not investment advice. Data may be delayed or incomplete, and LLM output can be wrong.*