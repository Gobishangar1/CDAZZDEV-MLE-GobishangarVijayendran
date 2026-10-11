# Sample outputs

Each run writes one folder here: `<TICKER>_<as-of date>/`

| File | Content |
|---|---|
| `brief.html` | the one-page research brief, self-contained (chart embedded) |
| `brief.md` | the same brief in Markdown (links `chart.png`) |
| `chart.png` | price + SMA-50/200 + Bollinger, RSI-14, MACD (last 12 months) |
| `run.json` | the full result: summary, headlines, per-headline JSON, sentiment, facts, conflicts, signal, validation history, warnings |

The committed folder comes from the executed Colab notebook run.
