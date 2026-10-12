"""One-page equity research brief (bonus): Markdown -> styled HTML with an embedded matplotlib chart.

    make_chart(df, ticker, path)     3 stacked panels, last 12 months: price + SMAs + Bollinger, RSI, MACD
    build_context(final)             all report values, pre-formatted as text
    write_report(final, df, run_dir) writes chart.png, brief.md, brief.html (self-contained)

The brief contains what the bonus asks for: company snapshot, technical outlook, news sentiment
with the top three headlines, the recommendation with its reasoning, and a risk disclaimer.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import markdown
import matplotlib

matplotlib.use("Agg")   # no display needed (Colab, servers, tests)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from jinja2 import Environment, FileSystemLoader  # noqa: E402

from equity_research.config import (  # noqa: E402
    CHART_DAYS,
    CHART_DPI,
    DISCLAIMER,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    RSI_PERIOD,
    SMA_LONG,
    SMA_SHORT,
    TOP_HEADLINES,
)
from equity_research.sentiment import SENTIMENT_VALUE  # noqa: E402

log = logging.getLogger(__name__)
TEMPLATES = Path(__file__).parent / "templates"

# Colours: validated categorical palette (blue, orange, aqua; CVD-safe) + neutral chrome.
INK, MUTED, GRID, BAND = "#0b0b0b", "#898781", "#e1e0d9", "#f0efec"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"

CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #1f1f1d;
       background: #f9f9f7; margin: 0; padding: 24px 16px; line-height: 1.5; font-size: 14px; }
main { max-width: 880px; margin: 0 auto; background: #fcfcfb; padding: 28px 36px;
       border: 1px solid rgba(11,11,11,0.10); border-radius: 8px; }
h1 { font-size: 22px; margin: 0 0 6px; } h2 { font-size: 16px; margin: 22px 0 8px;
     border-bottom: 1px solid #e1e0d9; padding-bottom: 4px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 6px 0; }
th, td { border-bottom: 1px solid #e1e0d9; padding: 5px 8px; text-align: left; vertical-align: top; }
th { color: #52514e; font-weight: 600; }
img { width: 100%; height: auto; border: 1px solid #e1e0d9; border-radius: 6px; }
ul { padding-left: 20px; margin: 6px 0; } li { margin: 2px 0; }
.badge { display: inline-block; padding: 1px 10px; border-radius: 999px; font-weight: 700; color: #fff; }
.badge.buy { background: #006300; } .badge.hold { background: #52514e; } .badge.sell { background: #b3261e; }
hr { border: none; border-top: 1px solid #e1e0d9; margin: 20px 0 10px; }
p:last-child { color: #52514e; font-size: 12px; }
@media (max-width: 640px) { main { padding: 18px 16px; } table { display: block; overflow-x: auto; }
                            td, th { padding: 4px 6px; } }
@media print { body { background: #fff; padding: 0; } main { border: none; } }
"""


# -----------------------------------------------------------------------------
# Chart
# -----------------------------------------------------------------------------
def make_chart(df: pd.DataFrame, ticker: str, path: Path) -> Path:
    """Three stacked panels with a shared date axis, last CHART_DAYS trading days."""
    d = df.iloc[-CHART_DAYS:]
    sma_s, sma_l, rsi_col = f"SMA_{SMA_SHORT}", f"SMA_{SMA_LONG}", f"RSI_{RSI_PERIOD}"

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7.2), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1.2, 1.4], "hspace": 0.12})
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")

    # Panel 1: price, moving averages, Bollinger Bands, crosses
    ax1.fill_between(d.index, d["BB_lower"], d["BB_upper"], color=BAND, label="Bollinger (20, 2σ)")
    ax1.plot(d.index, d["Close"], color=INK, linewidth=1.4, label="Close")
    ax1.plot(d.index, d[sma_s], color=ORANGE, linewidth=1.6, label=f"SMA-{SMA_SHORT}")
    ax1.plot(d.index, d[sma_l], color=AQUA, linewidth=1.6, linestyle="--", label=f"SMA-{SMA_LONG}")
    above = (d[sma_s] > d[sma_l]).astype(float).where(d[[sma_s, sma_l]].notna().all(axis=1))
    crosses = above.diff().dropna()
    for date, change in crosses[crosses != 0].items():
        ax1.scatter([date], [d.loc[date, sma_s]], s=46, zorder=5, color=ORANGE,
                    edgecolor="#fcfcfb", linewidth=1.5, marker="^" if change > 0 else "v")
        ax1.annotate("golden cross" if change > 0 else "death cross", (date, d.loc[date, sma_s]),
                     xytext=(6, 8 if change > 0 else -14), textcoords="offset points", fontsize=8, color="#52514e")
    ax1.set_title(f"{ticker}: price, moving averages and Bollinger Bands (last 12 months)",
                  loc="left", fontsize=10, color=INK)
    ax1.legend(loc="upper left", fontsize=8, frameon=False, ncol=4)

    # Panel 2: RSI with 30/70 levels
    ax2.plot(d.index, d[rsi_col], color=BLUE, linewidth=1.4)
    for level in (RSI_OVERSOLD, RSI_OVERBOUGHT):
        ax2.axhline(level, color=MUTED, linewidth=0.8, linestyle="--")
    ax2.set_ylim(0, 100)
    ax2.set_yticks([RSI_OVERSOLD, 50, RSI_OVERBOUGHT])
    ax2.set_title(f"RSI-{RSI_PERIOD} (Wilder): dashed lines at {RSI_OVERSOLD:.0f} and {RSI_OVERBOUGHT:.0f}",
                  loc="left", fontsize=9, color=INK)

    # Panel 3: MACD (bars: positive blue, negative red; lines on top)
    hist = d["MACD_hist"]
    ax3.bar(d.index, hist, width=1.0, color=[BLUE if v >= 0 else RED for v in hist], alpha=0.35)
    line_macd, = ax3.plot(d.index, d["MACD"], color=BLUE, linewidth=1.4)
    line_signal, = ax3.plot(d.index, d["MACD_signal"], color=ORANGE, linewidth=1.4)
    ax3.axhline(0, color="#c3c2b7", linewidth=0.8)
    ax3.set_title("MACD (12, 26, 9)", loc="left", fontsize=9, color=INK)
    ax3.legend([line_macd, line_signal, Patch(color=BLUE, alpha=0.35), Patch(color=RED, alpha=0.35)],
               ["MACD", "Signal", "Histogram > 0", "Histogram < 0"],
               loc="upper left", fontsize=8, frameon=False, ncol=4)

    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return path


# -----------------------------------------------------------------------------
# Text values
# -----------------------------------------------------------------------------
def _money(value: float | None, currency: str | None) -> str:
    if value is None:
        return "n/a"
    prefix = "$" if (currency or "USD") == "USD" else f"{currency} "
    for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(value) >= size:
            return f"{prefix}{value / size:.2f}{suffix}"
    return f"{prefix}{value:,.2f}"


def _short(text: str, limit: int) -> str:
    """Cut at a word boundary and add an ellipsis, so notes never end mid-word."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def _cell(text: object) -> str:
    """Safe inside a Markdown table cell."""
    return str(text if text is not None else "n/a").replace("|", "/").replace("\n", " ")


def top_headlines(results: list[dict], n: int = TOP_HEADLINES) -> list[dict]:
    """Most informative headlines: strongest |sentiment x confidence|, newest first on ties."""
    ranked = sorted(results, key=lambda r: (abs(SENTIMENT_VALUE[r["sentiment"]] * r["confidence"]),
                                            str(r.get("published_at") or "")), reverse=True)
    return ranked[:n]


def build_context(final: dict, chart_src: str) -> dict:
    s, sig, sent = final["summary"], final["signal"], final.get("sentiment")
    info, mom, cur = final.get("info") or {}, s["momentum"], s.get("currency")
    pe = "n/a (not reported)" if s["pe_ratio"] is None else f"{s['pe_ratio']:.1f} ({s['pe_source']})"
    sentiment_line = ("Unavailable (no headline could be classified)" if not sent else
                      f"{sent['label'].capitalize()} ({sent['score']:+.2f}) from {sent['n_valid']} headlines: "
                      f"{sent['counts']['positive']} positive, {sent['counts']['neutral']} neutral, "
                      f"{sent['counts']['negative']} negative · agreement {sent['agreement']:.0%} · "
                      f"coverage {sent['coverage']:.0%}")
    notes = list(final.get("warnings") or [])
    notes += [f"Headline not classified after a repair retry: \"{_short(f['headline'], 90)}\" "
              f"(reason: {_short(f['error'], 140)})" for f in final.get("headline_failures") or []]
    notes += [f"{e['node']}: {e['error']}" for e in final.get("errors") or [] if not e.get("critical")]
    if final.get("signal_source") == "rule_based_fallback":
        notes.append("The LLM signal failed validation on every attempt; the recommendation is rule-based.")
    return {
        "company": _cell(s.get("company_name") or s["ticker"]), "ticker": s["ticker"], "as_of": s["as_of"],
        "price": _money(s["current_price"], cur), "signal": sig["signal"], "signal_class": sig["signal"].lower(),
        "confidence": f"{sig['confidence']:.2f}",
        "signal_source": "LLM" if final.get("signal_source") == "llm" else "rule-based fallback",
        "sector": _cell(info.get("sector")), "market_cap": _money(info.get("marketCap"), cur), "pe": pe,
        "high_52w": _money(s["high_52w"], cur), "low_52w": _money(s["low_52w"], cur),
        "pct_from_high": f"{s['pct_from_52w_high']:+.1f}%",
        "ytd": "n/a" if s["ytd_return_pct"] is None else f"{s['ytd_return_pct']:+.1f}%",
        "chart_src": chart_src,
        "momentum_label": mom["label"].replace("_", " "), "momentum_score": f"{mom['score']:+.2f}",
        "momentum_parts": ", ".join(f"{k.replace('_', ' ')} {v:+.2f}" for k, v in mom["components"].items()),
        "facts": final["facts"], "conflicts": final["conflicts"],
        "sentiment_line": sentiment_line,
        "top_headlines": [{"headline": _cell(h["headline"]), "publisher": _cell(h.get("publisher")),
                           "sentiment": h["sentiment"], "confidence": f"{h['confidence']:.2f}",
                           "brief_reason": _cell(h["brief_reason"])}
                          for h in top_headlines(final.get("headline_results") or [])],
        "justification": sig["justification"],
        "key_drivers": "; ".join(sig["key_drivers"]),
        "conflicting_evidence": "; ".join(sig["conflicting_evidence"]) or "none",
        "notes": notes, "disclaimer": DISCLAIMER,
    }


def render_markdown(context: dict) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=False, trim_blocks=False)
    return env.get_template("brief.md.j2").render(**context)


def markdown_to_html(md_text: str, title: str) -> str:
    body = markdown.markdown(md_text, extensions=["tables"])
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{title}</title><style>{CSS}</style></head><body><main>{body}</main></body></html>")


def write_report(final: dict, df: pd.DataFrame, run_dir: Path) -> dict[str, str]:
    """chart.png + brief.md (links the PNG) + brief.html (chart embedded, one self-contained file)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    ticker = final["summary"]["ticker"]
    chart_path = make_chart(df, ticker, run_dir / "chart.png")

    md_text = render_markdown(build_context(final, chart_src="chart.png"))
    (run_dir / "brief.md").write_text(md_text, encoding="utf-8")

    data_uri = "data:image/png;base64," + base64.b64encode(chart_path.read_bytes()).decode("ascii")
    html = markdown_to_html(render_markdown(build_context(final, chart_src=data_uri)),
                            title=f"{ticker} equity research brief")
    (run_dir / "brief.html").write_text(html, encoding="utf-8")
    log.info("report_written dir=%s", run_dir)
    return {"chart": str(chart_path), "markdown": str(run_dir / "brief.md"), "html": str(run_dir / "brief.html")}
