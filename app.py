"""
FundaScore — Warren Buffett-style stock analyzer
Run with: streamlit run app.py
"""

import json
import random
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime
from fpdf import FPDF

BUFFETT_QUOTES = [
    ("Price is what you pay. Value is what you get.", "The Essays of Warren Buffett"),
    ("It's far better to buy a wonderful company at a fair price than a fair company at a wonderful price.", "1989 Shareholder Letter"),
    ("The stock market is a device for transferring money from the impatient to the patient.", "1990 Shareholder Letter"),
    ("Never invest in a business you cannot understand.", "1994 Shareholder Letter"),
    ("Risk comes from not knowing what you're doing.", "University of Georgia, 2001"),
    ("Our favourite holding period is forever.", "1988 Shareholder Letter"),
    ("Be fearful when others are greedy, and greedy when others are fearful.", "2004 Shareholder Letter"),
    ("The best investment you can make is in yourself.", "HBO Documentary, 2017"),
    ("Wide diversification is only required when investors do not understand what they are doing.", "2008 Shareholder Letter"),
    ("Only buy something that you'd be perfectly happy to hold if the market shut down for 10 years.", "2000 Shareholder Letter"),
    ("It takes 20 years to build a reputation and five minutes to ruin it.", "Various speeches"),
    ("In the short run, the market is a voting machine. In the long run, it is a weighing machine.", "The Intelligent Investor foreword"),
    ("The most important quality for an investor is temperament, not intellect.", "CNBC, 2009"),
    ("Opportunities come infrequently. When it rains gold, put out the bucket, not the thimble.", "1998 Shareholder Letter"),
    ("Someone's sitting in the shade today because someone planted a tree a long time ago.", "Various"),
    ("If a business does well, the stock eventually follows.", "Various interviews"),
    ("I try to buy stock in businesses that are so wonderful that an idiot can run them.", "Various"),
    ("Rule No. 1: Never lose money. Rule No. 2: Never forget Rule No. 1.", "Various"),
    ("The difference between successful people and really successful people is that really successful people say no to almost everything.", "Various"),
    ("You don't need to be a rocket scientist. Investing is not a game where the guy with the 160 IQ beats the guy with 130 IQ.", "1994 Shareholder Letter"),
]

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FundaScore",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────
# GOOGLE ANALYTICS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-5P3G6CKKPW"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-5P3G6CKKPW');
</script>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 1100px; }
    .stButton > button { border-radius: 8px; font-weight: 600; }
    .stTextInput > div > div > input {
        font-size: 1.1rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 2px; border-radius: 8px;
    }
    .metric-card {
        background: white; border: 1px solid #E5E7EB;
        border-radius: 10px; padding: 14px 18px;
    }
    div[data-testid="stMetric"] { background: white; border-radius:8px; padding:10px 14px; border:1px solid #E5E7EB; }
    .stAlert { border-radius: 8px; }
    hr { margin: 1rem 0; border-color: #E5E7EB; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SECTOR AVERAGE P/E  (approximate 2024–25 values)
# ─────────────────────────────────────────────────────────────────
SECTOR_PE = {
    "Technology": 29, "Healthcare": 22, "Financials": 14,
    "Financial Services": 14, "Consumer Cyclical": 26,
    "Consumer Defensive": 21, "Industrials": 22, "Energy": 13,
    "Utilities": 18, "Basic Materials": 18, "Real Estate": 36,
    "Communication Services": 22,
}
WEIGHTS = {
    "balance_sheet": 0.15, "profitability": 0.25,
    "cashflow": 0.15,      "capital_efficiency": 0.10,
    "valuation": 0.15,     "moat": 0.10, "management": 0.10,
}

# ─────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        if not info or "longName" not in info:
            return None, "Ticker not found. Please check the symbol and try again."
        return {
            "info":          info,
            "financials":    stock.financials,
            "cashflow":      stock.cashflow,
            "balance_sheet": stock.balance_sheet,
        }, None
    except Exception as exc:
        return None, f"Error fetching data: {exc}"

@st.cache_data(ttl=300, show_spinner=False)
def search_company_name(query: str):
    """Search Yahoo Finance by company name; return top equity matches."""
    try:
        results = yf.Search(query, max_results=6).quotes
        return [r for r in results if r.get("quoteType") == "EQUITY"][:3]
    except Exception:
        return []

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def sg(info, key, default=None):
    """Safe-get from info dict, treating NaN as missing."""
    v = info.get(key, default)
    if v is None:
        return default
    try:
        if np.isnan(v):
            return default
    except TypeError:
        pass
    return v

def get_row(df, names):
    """Try a list of row names in a DataFrame; return the first match."""
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            row = df.loc[n].dropna()
            return row if len(row) else None
    return None

def cagr(series, years=3):
    if series is None or len(series) < 2:
        return None
    n   = min(years, len(series) - 1)
    end = series.iloc[0]
    beg = series.iloc[n]
    if beg <= 0 or end <= 0:
        return None
    return (end / beg) ** (1 / n) - 1

def fp(v, sign=True, decimals=1):
    if v is None: return "N/A"
    pfx = "+" if (sign and v > 0) else ""
    return f"{pfx}{v*100:.{decimals}f}%"

def fn(v, d=2):
    return "N/A" if v is None else f"{v:.{d}f}"

def fl(v):
    if v is None: return "N/A"
    a = abs(v)
    if a >= 1e12: return f"${v/1e12:.2f}T"
    if a >= 1e9:  return f"${v/1e9:.2f}B"
    if a >= 1e6:  return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"

def flag(score):
    if score is None: return "⬜ N/A"
    if score >= 7:    return "🟢 STRONG"
    if score >= 5:    return "🟡 OK"
    return "🔴 WEAK"

def color(score):
    if score is None: return "#9CA3AF"
    if score >= 7:    return "#22C55E"
    if score >= 5:    return "#F59E0B"
    return "#EF4444"

# ─────────────────────────────────────────────────────────────────
# ROIC CALCULATION
# ─────────────────────────────────────────────────────────────────
def calc_roic(financials, balance_sheet):
    try:
        op = get_row(financials, ["Operating Income", "EBIT", "Ebit",
                                  "Total Operating Income As Reported"])
        if op is None: return None
        op_val = op.iloc[0]

        tax_rate = 0.21
        pretax = get_row(financials, ["Pretax Income", "Income Before Tax"])
        taxes  = get_row(financials, ["Tax Provision", "Income Tax Expense"])
        if pretax is not None and taxes is not None:
            pt, tx = pretax.iloc[0], abs(taxes.iloc[0])
            if pt != 0:
                tax_rate = min(max(tx / abs(pt), 0), 0.5)

        nopat = op_val * (1 - tax_rate)

        eq = get_row(balance_sheet, [
            "Total Stockholder Equity", "Stockholders Equity",
            "Total Equity Gross Minority Interest", "Common Stock Equity",
        ])
        ltd = get_row(balance_sheet, [
            "Long Term Debt", "Long-Term Debt",
            "Long Term Debt And Capital Lease Obligation",
        ])
        if eq is None: return None

        ic = eq.iloc[0] + (ltd.iloc[0] if ltd is not None else 0)
        return None if ic <= 0 else nopat / ic
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────
# SCORING FUNCTIONS  (each returns: score | None, list of detail tuples)
# ─────────────────────────────────────────────────────────────────
def s_balance(info, _bs):
    scores, details = [], []

    cr = sg(info, "currentRatio")
    if cr is not None:
        s = 10 if cr>=2 else 8 if cr>=1.5 else 6 if cr>=1 else 3 if cr>=0.7 else 1
        scores.append(s)
        lbl = "✅ Strong" if cr>=1.5 else "✅ Adequate" if cr>=1 else "⚠️ Tight" if cr>=0.7 else "❌ Critical"
        details.append(("Current Ratio", fn(cr), lbl))

    de_raw = sg(info, "debtToEquity")
    if de_raw is not None:
        de = de_raw / 100
        s  = 10 if de<0.3 else 9 if de<0.5 else 7 if de<1 else 4 if de<2 else 1
        scores.append(s)
        lbl = "✅ Conservative" if de<0.5 else "✅ Manageable" if de<1 else "⚠️ Elevated" if de<2 else "❌ High leverage"
        details.append(("Debt / Equity", f"{de:.2f}x", lbl))

    cash = sg(info, "totalCash", 0) or 0
    debt = sg(info, "totalDebt", 0) or 0
    if debt > 0:
        r = cash / debt
        s = 10 if r>=1 else 7 if r>=0.5 else 4 if r>=0.25 else 1
        scores.append(s)
        lbl = "✅ Cash > Debt" if r>=1 else "✅ Good coverage" if r>=0.5 else "⚠️ Partial" if r>=0.25 else "❌ Low coverage"
        details.append(("Cash / Total Debt", f"{r:.2f}x", lbl))

    return (round(float(np.mean(scores)), 1) if scores else None), details


def s_profit(info, fin):
    scores, details = [], []

    margin = sg(info, "profitMargins")
    if margin is not None:
        s = 10 if margin>=0.25 else 8 if margin>=0.20 else 6 if margin>=0.15 else 4 if margin>=0.10 else 2 if margin>=0 else 0
        scores.append(s)
        lbl = "✅ Excellent (≥20%)" if margin>=0.20 else "✅ Good (≥10%)" if margin>=0.10 else "⚠️ Thin" if margin>=0 else "❌ Negative"
        details.append(("Net Profit Margin", fp(margin, sign=False), lbl))

    roe = sg(info, "returnOnEquity")
    if roe is not None:
        s = 10 if roe>=0.25 else 8 if roe>=0.20 else 6 if roe>=0.15 else 4 if roe>=0.10 else 1
        scores.append(s)
        lbl = "✅ Excellent (Buffett ≥20%)" if roe>=0.20 else "✅ Good (≥15%)" if roe>=0.15 else "⚠️ Average" if roe>=0.10 else "❌ Below par"
        details.append(("Return on Equity (ROE)", fp(roe, sign=False), lbl))

    # EPS / earnings growth — try financials first, fallback to info
    eps_growth = None
    eps_row = get_row(fin, ["Diluted EPS", "Basic EPS", "EPS"])
    if eps_row is not None and len(eps_row) >= 2:
        eps_growth = cagr(eps_row)
    if eps_growth is None:
        ni_row = get_row(fin, ["Net Income", "Net Income Common Stockholders"])
        if ni_row is not None and len(ni_row) >= 2:
            eps_growth = cagr(ni_row)
    if eps_growth is None:
        eps_growth = sg(info, "earningsGrowth")

    if eps_growth is not None:
        s = 10 if eps_growth>=0.20 else 8 if eps_growth>=0.15 else 6 if eps_growth>=0.10 else 4 if eps_growth>=0.05 else 2 if eps_growth>=0 else 0
        scores.append(s)
        lbl = "✅ Excellent" if eps_growth>=0.15 else "✅ Good (Buffett ≥10%)" if eps_growth>=0.10 else "⚠️ Slow" if eps_growth>=0 else "❌ Shrinking"
        details.append(("EPS Growth (3-yr CAGR)", fp(eps_growth), lbl))

    return (round(float(np.mean(scores)), 1) if scores else None), details


def s_cashflow(info, cf, _fin):
    scores, details = [], []

    fcf_val  = sg(info, "freeCashflow")
    fcf_ser  = get_row(cf, ["Free Cash Flow", "FreeCashFlow"])

    if fcf_val is None and fcf_ser is not None and len(fcf_ser):
        fcf_val = float(fcf_ser.iloc[0])

    if fcf_val is not None:
        s = 10 if fcf_val > 0 else 0
        scores.append(s)
        details.append(("Free Cash Flow (TTM)", fl(fcf_val),
                         "✅ Positive" if fcf_val > 0 else "❌ Negative (burning cash)"))

    if fcf_ser is not None and len(fcf_ser) >= 2:
        pos = fcf_ser[fcf_ser > 0]
        if len(pos) >= 2:
            g = cagr(fcf_ser.dropna())
            if g is not None:
                s = 10 if g>=0.20 else 8 if g>=0.10 else 6 if g>=0.05 else 4 if g>=0 else 1
                scores.append(s)
                lbl = "✅ Strong growth" if g>=0.15 else "✅ Growing" if g>=0.05 else "⚠️ Flat" if g>=0 else "❌ Declining"
                details.append(("FCF Growth (3-yr CAGR)", fp(g), lbl))

    op_row = get_row(cf, ["Operating Cash Flow", "Total Cash From Operating Activities",
                           "Cash From Operating Activities"])
    if op_row is not None and len(op_row):
        ov = float(op_row.iloc[0])
        scores.append(10 if ov > 0 else 1)
        details.append(("Operating Cash Flow", fl(ov),
                         "✅ Operations generating cash" if ov > 0 else "❌ Operations burning cash"))

    return (round(float(np.mean(scores)), 1) if scores else None), details


def s_capeff(info, fin, bs):
    scores, details = [], []

    roic = calc_roic(fin, bs)
    if roic is not None:
        s = 10 if roic>=0.25 else 9 if roic>=0.20 else 7 if roic>=0.15 else 5 if roic>=0.10 else 3 if roic>=0.05 else 1
        scores.append(s)
        lbl = "✅ Excellent (>25%)" if roic>=0.25 else "✅ Strong (>15%)" if roic>=0.15 else "✅ Adequate" if roic>=0.10 else "⚠️ Below avg" if roic>=0.05 else "❌ Poor"
        details.append(("ROIC (Return on Invested Capital)", fp(roic, sign=False), lbl))

    roa = sg(info, "returnOnAssets")
    if roa is not None:
        s = 10 if roa>=0.15 else 8 if roa>=0.10 else 5 if roa>=0.05 else 3 if roa>=0 else 1
        scores.append(s)
        lbl = "✅ Excellent (>15%)" if roa>=0.15 else "✅ Good (>10%)" if roa>=0.10 else "✅ Average" if roa>=0.05 else "⚠️ Low" if roa>=0 else "❌ Negative"
        details.append(("Return on Assets (ROA)", fp(roa, sign=False), lbl))

    return (round(float(np.mean(scores)), 1) if scores else None), details


def s_valuation(info, sector):
    scores, details = [], []
    sec_pe = SECTOR_PE.get(sector, 20)

    pe = sg(info, "trailingPE")
    if pe is not None and 0 < pe < 500:
        r = pe / sec_pe
        s = 10 if r<=0.70 else 8 if r<=0.85 else 6 if r<=1.0 else 4 if r<=1.2 else 2 if r<=1.5 else 1
        scores.append(s)
        lbl = "🟢 Undervalued vs sector" if r<=0.85 else "🟡 Fair value" if r<=1.10 else "🟠 Premium" if r<=1.5 else "🔴 Expensive"
        details.append(("P/E Ratio", f"{pe:.1f}x  (sector avg: {sec_pe}x)", lbl))

    peg = sg(info, "pegRatio")
    if peg is not None and peg > 0:
        s = 10 if peg<0.5 else 8 if peg<1.0 else 6 if peg<1.5 else 3 if peg<2.0 else 1
        scores.append(s)
        lbl = "🟢 Undervalued (<1)" if peg<1.0 else "🟡 Fair (1–1.5)" if peg<1.5 else "🟠 Pricey (1.5–2)" if peg<2.0 else "🔴 Expensive (>2)"
        details.append(("PEG Ratio  (P/E ÷ growth)", fn(peg), lbl))

    fpe = sg(info, "forwardPE")
    if fpe is not None and 0 < fpe < 500:
        details.append(("Forward P/E", f"{fpe:.1f}x", "📊 Context only"))

    return (round(float(np.mean(scores)), 1) if scores else None), details, sec_pe


def s_moat(gross_margin, moat_rating):
    scores, details = [], []

    if moat_rating is not None:
        s = moat_rating * 2
        scores.append(s)
        labels = {1:"❌ No clear advantage",2:"⚠️ Weak",3:"🟡 Some advantage",4:"✅ Clear moat",5:"✅ Dominant moat"}
        details.append(("Your Moat Assessment", f"{moat_rating}/5", labels.get(moat_rating,"")))

    if gross_margin is not None:
        s = 10 if gross_margin>=0.60 else 8 if gross_margin>=0.40 else 5 if gross_margin>=0.25 else 3 if gross_margin>=0.10 else 1
        scores.append(s)
        lbl = "✅ Pricing power (>60%)" if gross_margin>=0.60 else "✅ Strong (>40%)" if gross_margin>=0.40 else "🟡 Average" if gross_margin>=0.25 else "⚠️ Thin margins"
        details.append(("Gross Margin  (moat proxy)", fp(gross_margin, sign=False), lbl))

    return (round(float(np.mean(scores)), 1) if scores else None), details


def s_mgmt(mgmt_rating, insider_pct):
    scores, details = [], []

    if mgmt_rating is not None:
        s = mgmt_rating * 2
        scores.append(s)
        labels = {1:"❌ Poor",2:"⚠️ Below avg",3:"🟡 Average",4:"✅ Good",5:"✅ Excellent"}
        details.append(("Your Management Rating", f"{mgmt_rating}/5", labels.get(mgmt_rating,"")))

    if insider_pct is not None:
        s = 10 if insider_pct>=0.10 else 7 if insider_pct>=0.05 else 5 if insider_pct>=0.02 else 3
        scores.append(s)
        lbl = "✅ High (skin in the game)" if insider_pct>=0.05 else "🟡 Moderate" if insider_pct>=0.02 else "⚠️ Low"
        details.append(("Insider Ownership", fp(insider_pct, sign=False), lbl))

    return (round(float(np.mean(scores)), 1) if scores else None), details


# ─────────────────────────────────────────────────────────────────
# WEIGHTED TOTAL & VERDICT
# ─────────────────────────────────────────────────────────────────
def weighted_score(scores_dict):
    wsum, wtot = 0.0, 0.0
    for k, w in WEIGHTS.items():
        v = scores_dict.get(k)
        if v is not None:
            wsum += v * w
            wtot += w
    return round(wsum / wtot, 1) if wtot else None


def verdict(score):
    if score is None:
        return "⬜ INSUFFICIENT DATA", "#9CA3AF", "Not enough data to produce a verdict."
    if score >= 7.5:
        return "🟢 STRONG BUY",  "#16A34A", "Exceptional fundamentals across most criteria — hallmark of a quality long-term holding."
    if score >= 6.5:
        return "🟢 BUY",         "#22C55E", "Solid fundamentals. Consistent with a good long-term investment."
    if score >= 5.5:
        return "🟡 HOLD",        "#D97706", "Decent company but with notable weaknesses. Hold if you own it; wait for a better price if you don't."
    if score >= 4.0:
        return "🟡 WEAK HOLD",   "#F59E0B", "Mixed picture. High risk. Only consider for a small speculative position."
    return     "🔴 AVOID",       "#DC2626", "Significant fundamental weaknesses — not aligned with a value investing approach."


# ─────────────────────────────────────────────────────────────────
# CRITERION CARD
# ─────────────────────────────────────────────────────────────────
def criterion_card(title, score, details, weight_pct, note=""):
    sc   = score
    clr  = color(sc)
    flg  = flag(sc)
    sdsp = fn(sc, 1) if sc is not None else "N/A"

    rows = "".join(f"""
        <tr>
          <td style="padding:4px 8px;color:#6B7280;font-size:12px;width:44%;">{lbl}</td>
          <td style="padding:4px 8px;font-weight:600;font-size:12px;width:22%;">{val}</td>
          <td style="padding:4px 8px;font-size:12px;">{asmt}</td>
        </tr>""" for lbl, val, asmt in details)

    note_html = (f'<p style="margin:8px 0 0;padding:8px 10px;background:#F0F9FF;'
                 f'border-radius:6px;font-size:11px;color:#0369A1;line-height:1.5;">'
                 f'💡 <strong>What makes a good score?</strong> {note}</p>') if note else ""

    st.markdown(f"""
    <div style="border:1px solid #E5E7EB;border-left:5px solid {clr};
                border-radius:10px;padding:14px 16px;margin-bottom:12px;background:#FAFAFA;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <div>
          <span style="font-weight:700;font-size:14px;color:#111827;">{title}</span>
          <span style="margin-left:8px;font-size:10px;color:#9CA3AF;font-weight:500;">WEIGHT: {weight_pct}%</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="font-size:11px;color:#6B7280;">{flg}</span>
          <span style="background:{clr};color:#fff;font-weight:700;
                       font-size:13px;padding:3px 12px;border-radius:20px;">{sdsp}/10</span>
        </div>
      </div>
      <table style="width:100%;border-collapse:collapse;">{rows}</table>
      {note_html}
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────────────────────────────
def _p(text: str) -> str:
    """Strip emojis / unicode symbols so fpdf Helvetica doesn't crash."""
    replacements = {
        "✅": "PASS", "❌": "FAIL", "⚠️": "WARN", "⚠": "WARN",
        "🟢": "GOOD", "🟡": "FAIR", "🔴": "POOR", "🟠": "FAIR",
        "📊": "", "📖": "", "📈": "", "📉": "",
        "①": "1.", "②": "2.", "③": "3.", "④": "4.",
        "⑤": "5.", "⑥": "6.", "⑦": "7.",
        "·": "-", "—": "-", "​": "",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Drop any remaining non-latin-1 characters
    return text.encode("latin-1", errors="ignore").decode("latin-1")

def make_pdf(company, ticker, vtext, vscore, criteria_results, ts):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "FundaScore - Stock Analysis Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, _p(f"{company} ({ticker.upper()})  -  {ts}"), ln=True, align="C")
    pdf.ln(4)

    pdf.set_fill_color(30, 58, 138)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 12, _p(f"Verdict: {vtext}   |   Weighted Score: {vscore} / 10"),
             ln=True, align="C", fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5, "DISCLAIMER: For educational purposes only. Not financial advice. "
                         "Always consult a qualified financial advisor before investing.")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    for cname, (cscore, cdetails) in criteria_results.items():
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_fill_color(46, 117, 182)
        pdf.set_text_color(255, 255, 255)
        score_txt = fn(cscore, 1) if cscore is not None else "N/A"
        pdf.cell(0, 8, _p(f"  {cname}   -   Score: {score_txt} / 10"), ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 10)
        for lbl, val, asmt in cdetails:
            pdf.cell(70, 6, _p(lbl)[:38], border=0)
            pdf.cell(55, 6, _p(val)[:30], border=0)
            pdf.cell(0,  6, _p(asmt)[:36], ln=True, border=0)
        pdf.ln(3)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, f"FundaScore  -  {ts}  -  Not financial advice", align="C", ln=True)
    return bytes(pdf.output())


# ─────────────────────────────────────────────────────────────────
# MARKDOWN REPORT
# ─────────────────────────────────────────────────────────────────
def make_md(company, ticker, vtext, vscore, criteria_results, ts):
    lines = [
        f"# 📊 FundaScore: {company} ({ticker.upper()})",
        f"*{ts}*", "",
        "---", "",
        f"## Verdict: {vtext}",
        f"**Weighted Score: {vscore} / 10**", "",
        "> ⚠️ *Educational purposes only — not financial advice.*", "", "---", "",
    ]
    for cname, (cscore, cdetails) in criteria_results.items():
        lines += [
            f"## {cname}",
            f"**Score: {fn(cscore,1) if cscore is not None else 'N/A'} / 10**", "",
            "| Metric | Value | Assessment |",
            "|--------|-------|------------|",
        ]
        for lbl, val, asmt in cdetails:
            lines.append(f"| {lbl} | {val} | {asmt} |")
        lines.append("")
    lines += ["---", "*FundaScore · Warren Buffett-style value investing framework*"]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────
def main():
    # ── Header ──────────────────────────────────────────────────
    if "header_quote" not in st.session_state:
        st.session_state["header_quote"] = random.choice(BUFFETT_QUOTES)
    hq, hsrc = st.session_state["header_quote"]
    st.markdown(f"""
    <div style="text-align:center;padding:10px 0 6px;">
      <h1 style="font-size:2.6rem;margin:0;letter-spacing:-1px;">📊 FundaScore</h1>
      <p style="color:#6B7280;font-size:1rem;margin:4px 0 0;">
        Warren Buffett-style analysis · Type a ticker · Get a verdict
      </p>
      <p style="color:#9CA3AF;font-size:0.85rem;font-style:italic;margin:10px auto 0;max-width:600px;">
        "{hq}"
      </p>
      <p style="color:#CBD5E1;font-size:0.75rem;margin:2px 0 0;">— Warren Buffett · {hsrc}</p>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Ticker input ─────────────────────────────────────────────
    # on_change fires when user presses Enter or blurs the field
    def _request_analyze():
        val = st.session_state.get("_ticker_box", "").strip().upper()
        if val:
            st.session_state["active_ticker"] = val
            st.session_state["do_analyze"] = True

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.text_input("Ticker", placeholder="e.g. AAPL  ·  MSFT  ·  NVDA",
                      label_visibility="collapsed",
                      key="_ticker_box",
                      on_change=_request_analyze)
        if st.button("🔍  Analyze", use_container_width=True, type="primary"):
            val = st.session_state.get("_ticker_box", "").strip().upper()
            if val:
                st.session_state["active_ticker"] = val
                st.session_state["do_analyze"] = True

    # The single source of truth for what to analyze
    ticker = st.session_state.get("active_ticker", "")
    do_analyze = st.session_state.get("do_analyze", False)

    if not ticker:
        st.markdown("""
        <div style="text-align:center;color:#9CA3AF;padding:50px 0;">
          <p style="font-size:3.5rem;margin:0;">📈</p>
          <p style="font-size:1rem;">Enter a ticker or company name above and hit Analyze</p>
          <p style="font-size:11px;">Data from Yahoo Finance · Not financial advice</p>
        </div>""", unsafe_allow_html=True)
        return

    # Don't show stale results from a previous search while user hasn't re-analyzed
    if not do_analyze and f"data_{ticker}" not in st.session_state:
        return

    # Consume the flag so it doesn't re-trigger on slider moves etc.
    st.session_state["do_analyze"] = False

    # ── Fetch ────────────────────────────────────────────────────
    if f"data_{ticker}" not in st.session_state:
        with st.spinner(f"Fetching data for {ticker}…"):
            data, err = fetch_stock_data(ticker)
        if err or not data:
            # Ticker not found — try as a company name, auto-pick best match
            with st.spinner("Searching by company name…"):
                suggestions = search_company_name(ticker)
            if suggestions:
                best  = suggestions[0]
                sym   = best.get("symbol", "")
                sname = best.get("longname") or best.get("shortname", sym)
                others = [r.get("symbol","") for r in suggestions[1:] if r.get("symbol")]
                # Fetch the best match directly — no button needed
                with st.spinner(f"Loading {sym}…"):
                    data, err = fetch_stock_data(sym)
                if err or not data:
                    st.error(f"❌ Could not load data for {sym}. Try typing the ticker directly (e.g. MSFT).")
                    return
                ticker = sym
                st.session_state["active_ticker"] = sym
                st.session_state[f"data_{sym}"] = data
                # Show a small banner so the user knows what was matched
                other_txt = f"  ·  Other matches: {', '.join(others)}" if others else ""
                st.info(f"🔍 **'{st.session_state.get('_ticker_box','').upper()}'** matched → **{sym}** ({sname}){other_txt}")
            else:
                st.error(f"❌ Could not find **'{ticker}'**. Try the ticker symbol (e.g. AAPL, MSFT).")
                return
        st.session_state[f"data_{ticker}"] = data

    data   = st.session_state[f"data_{ticker}"]
    info   = data["info"]
    fin    = data["financials"]
    cf     = data["cashflow"]
    bs     = data["balance_sheet"]

    name   = sg(info, "longName", ticker)
    sector = sg(info, "sector", "Unknown")
    indust = sg(info, "industry", "Unknown")
    descr  = sg(info, "longBusinessSummary", "")
    price  = sg(info, "currentPrice") or sg(info, "regularMarketPrice")
    mcap   = sg(info, "marketCap")
    gm     = sg(info, "grossMargins")
    ins    = sg(info, "heldPercentInsiders")

    # ── ETF / Fund warning ───────────────────────────────────────
    quote_type = sg(info, "quoteType", "")
    if quote_type in ("ETF", "MUTUALFUND"):
        fund_label = "ETF" if quote_type == "ETF" else "Mutual Fund"
        st.warning(
            f"⚠️ **{name} is an {fund_label}** — FundaScore is built for individual stocks "
            f"using Warren Buffett's framework. Metrics like ROE, ROIC, and Debt/Equity "
            f"don't apply to funds, so the score below will be misleading.\n\n"
            f"**For ETFs, look at:** expense ratio, long-term returns vs benchmark, "
            f"AUM size, and dividend yield instead."
        )

    # ── Prerequisite gate ────────────────────────────────────────
    gate_key    = f"gate_{ticker}"
    proceed_key = f"proceed_{ticker}"

    gate_quote, gate_src = random.choice(BUFFETT_QUOTES)
    st.markdown(f"""
    <div style="background:#F0F9FF;border:1px solid #BAE6FD;border-radius:10px;
                padding:16px 20px;margin:10px 0;">
      <p style="margin:0;font-weight:700;font-size:15px;">
        🧠 Buffett's Rule #1 — Do you understand what <em>{name}</em> does?
      </p>
      <p style="margin:6px 0 0;color:#0369A1;font-size:13px;font-style:italic;">
        "{gate_quote}"
      </p>
      <p style="margin:3px 0 0;color:#7DD3FC;font-size:11px;">— Warren Buffett · {gate_src}</p>
    </div>""", unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    if g1.button("✅  Yes, I know it well",     key=f"yes_{ticker}", use_container_width=True):
        st.session_state[gate_key] = "yes"
    if g2.button("🤔  Somewhat familiar",        key=f"smt_{ticker}", use_container_width=True):
        st.session_state[gate_key] = "somewhat"
    if g3.button("❓  No, not really",           key=f"no_{ticker}",  use_container_width=True):
        st.session_state[gate_key] = "no"

    gate = st.session_state.get(gate_key)

    if gate is None:
        st.info("👆  Answer the question above to unlock the analysis.")
        return

    # ── No / Don't understand ────────────────────────────────────
    if gate == "no":
        st.warning(
            "⚠️  **High risk flag.** Buffett says unfamiliar businesses should be skipped. "
            "You can still run the analysis, but factor this in when deciding."
        )
        if descr:
            with st.expander(f"📖  What does {name} do? — click to read"):
                st.markdown(descr)

        if st.session_state.get(proceed_key) != "yes":
            p1, p2 = st.columns(2)
            if p1.button("🔍  Proceed with analysis anyway", use_container_width=True, key=f"proc_{ticker}"):
                st.session_state[proceed_key] = "yes"
                st.rerun()
            if p2.button("← Change ticker", use_container_width=True, key=f"back_{ticker}"):
                for k in [gate_key, proceed_key, f"data_{ticker}"]:
                    st.session_state.pop(k, None)
                st.rerun()
            return

    # ── Company overview ─────────────────────────────────────────
    st.divider()
    st.markdown(f"### 🏢 {name}  `{ticker}`")
    price_str  = f"${price:.2f}" if price else "N/A"
    mcap_str   = fl(mcap)
    indust_str = (indust[:28] + "…") if indust and len(indust) > 28 else (indust or "N/A")
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:8px 0;">
      <div style="background:#fff;border:1px solid #E5E7EB;border-radius:10px;padding:12px 16px;">
        <p style="margin:0;font-size:11px;color:#6B7280;font-weight:600;text-transform:uppercase;letter-spacing:.5px;">Price</p>
        <p style="margin:4px 0 0;font-size:1.4rem;font-weight:700;color:#111827;">{price_str}</p>
      </div>
      <div style="background:#fff;border:1px solid #E5E7EB;border-radius:10px;padding:12px 16px;">
        <p style="margin:0;font-size:11px;color:#6B7280;font-weight:600;text-transform:uppercase;letter-spacing:.5px;">Market Cap</p>
        <p style="margin:4px 0 0;font-size:1.4rem;font-weight:700;color:#111827;">{mcap_str}</p>
      </div>
      <div style="background:#fff;border:1px solid #E5E7EB;border-radius:10px;padding:12px 16px;">
        <p style="margin:0;font-size:11px;color:#6B7280;font-weight:600;text-transform:uppercase;letter-spacing:.5px;">Sector</p>
        <p style="margin:4px 0 0;font-size:1rem;font-weight:700;color:#111827;">{sector}</p>
      </div>
      <div style="background:#fff;border:1px solid #E5E7EB;border-radius:10px;padding:12px 16px;">
        <p style="margin:0;font-size:11px;color:#6B7280;font-weight:600;text-transform:uppercase;letter-spacing:.5px;">Industry</p>
        <p style="margin:4px 0 0;font-size:1rem;font-weight:700;color:#111827;">{indust_str}</p>
      </div>
    </div>""", unsafe_allow_html=True)

    if gate == "somewhat" and descr:
        with st.expander("📖  Company description"):
            st.markdown(descr)

    st.divider()

    # ── Qualitative sliders ──────────────────────────────────────
    st.markdown("#### ✍️  Your Assessment  *(only you can judge these)*")
    q1, q2 = st.columns(2)
    moat_r = q1.slider("⑥  Competitive Moat  (1 = none · 5 = dominant)",  1, 5, 3, key=f"moat_{ticker}")
    mgmt_r = q2.slider("⑦  Management Quality  (1 = poor · 5 = excellent)", 1, 5, 3, key=f"mgmt_{ticker}")

    st.divider()

    # ── Score ────────────────────────────────────────────────────
    with st.spinner("Calculating…"):
        sc1, d1 = s_balance(info, bs)
        sc2, d2 = s_profit(info, fin)
        sc3, d3 = s_cashflow(info, cf, fin)
        sc4, d4 = s_capeff(info, fin, bs)
        sc5, d5, _spe = s_valuation(info, sector)
        sc6, d6 = s_moat(gm, moat_r)
        sc7, d7 = s_mgmt(mgmt_r, ins)

    scores = dict(balance_sheet=sc1, profitability=sc2, cashflow=sc3,
                  capital_efficiency=sc4, valuation=sc5, moat=sc6, management=sc7)
    wscore = weighted_score(scores)
    vtext, vcolor, vdesc = verdict(wscore)
    wdsp = fn(wscore, 1) if wscore is not None else "N/A"

    # ── Verdict (top) ─────────────────────────────────────────────
    st.divider()
    vc_display = "#4ADE80" if vcolor in ("#16A34A","#22C55E") else vcolor
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1F3864 0%,#2E75B6 100%);
                border-radius:14px;padding:28px 30px;text-align:center;color:#fff;margin:4px 0 6px;">
      <p style="margin:0;font-size:11px;opacity:.6;letter-spacing:3px;text-transform:uppercase;">Weighted Score</p>
      <p style="margin:4px 0;font-size:3rem;font-weight:800;line-height:1.1;">
        {wdsp}<span style="font-size:1.4rem;opacity:.5;"> / 10</span>
      </p>
      <p style="margin:8px 0 4px;font-size:1.8rem;font-weight:700;color:{vc_display};">{vtext}</p>
      <p style="margin:0;font-size:13px;opacity:.8;max-width:520px;margin-left:auto;margin-right:auto;">{vdesc}</p>
    </div>
    <p style="text-align:center;color:#9CA3AF;font-size:11px;margin:4px 0 0;">
      ⚠️ Educational purposes only · Not financial advice · Consult a qualified advisor before investing
    </p>""", unsafe_allow_html=True)

    # ── Criteria cards ───────────────────────────────────────────
    st.divider()
    st.markdown("### 📊  Score Breakdown  *(scroll to see details)*")

    NOTES = {
        "① Balance Sheet": (
            "Current Ratio >1.5 (can cover short-term debts), Debt/Equity <0.5 (low leverage). "
            "Buffett prefers companies that could retire all debt within a few years of earnings."
        ),
        "② Profitability & Returns": (
            "ROE >20% is Buffett's key threshold — it means the company generates strong returns without relying on debt. "
            "Net margin >20% and EPS growing >10%/yr are also hallmarks of a quality business."
        ),
        "③ Cash Flow": (
            "Positive and growing free cash flow every year. Buffett calls FCF 'owner earnings' — "
            "the real cash a business generates for shareholders after maintaining its operations."
        ),
        "④ Capital Efficiency": (
            "ROIC >15% means the company creates significant value above its cost of capital. "
            "This is one of the strongest long-term predictors of stock performance."
        ),
        "⑤ Valuation": (
            "P/E at or below your sector average means you're not overpaying. "
            "PEG <1 means growth is underpriced. Even great companies are bad investments if the price is too high."
        ),
        "⑥ Competitive Moat": (
            "Rate 4–5 if the company has durable advantages: brand loyalty, patents, network effects, or high switching costs. "
            "Gross margin >40% is a quantitative sign of pricing power."
        ),
        "⑦ Management": (
            "Rate 4–5 for honest, shareholder-focused leadership. "
            "Insider ownership >5% means management has real skin in the game alongside you."
        ),
    }

    criteria = [
        ("① Balance Sheet",            sc1, d1, 15),
        ("② Profitability & Returns",  sc2, d2, 25),
        ("③ Cash Flow",                sc3, d3, 15),
        ("④ Capital Efficiency",       sc4, d4, 10),
        ("⑤ Valuation",               sc5, d5, 15),
        ("⑥ Competitive Moat",        sc6, d6, 10),
        ("⑦ Management",              sc7, d7, 10),
    ]
    left, right = st.columns(2)
    for i, (title, sc, det, wt) in enumerate(criteria):
        with (left if i % 2 == 0 else right):
            criterion_card(title, sc, det or [], wt, note=NOTES.get(title, ""))

    # ── Exports ──────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📥  Export Report")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    cr_results = {
        "① Balance Sheet (15%)":           (sc1, d1 or []),
        "② Profitability & Returns (25%)": (sc2, d2 or []),
        "③ Cash Flow (15%)":               (sc3, d3 or []),
        "④ Capital Efficiency (10%)":      (sc4, d4 or []),
        "⑤ Valuation (15%)":              (sc5, d5 or []),
        "⑥ Competitive Moat (10%)":       (sc6, d6 or []),
        "⑦ Management (10%)":             (sc7, d7 or []),
    }

    e1, e2, e3 = st.columns(3)
    with e1:
        try:
            pdf_bytes = make_pdf(name, ticker, vtext, wdsp, cr_results, ts)
            st.download_button("📄  PDF Report", data=pdf_bytes,
                               file_name=f"FundaScore_{ticker}_{datetime.now():%Y%m%d}.pdf",
                               mime="application/pdf", use_container_width=True)
        except Exception:
            st.button("📄  PDF (unavailable)", disabled=True, use_container_width=True)

    with e2:
        md = make_md(name, ticker, vtext, wdsp, cr_results, ts)
        st.download_button("📝  Markdown Report", data=md,
                           file_name=f"FundaScore_{ticker}_{datetime.now():%Y%m%d}.md",
                           mime="text/markdown", use_container_width=True)

    with e3:
        summary = {"ticker": ticker, "company": name, "date": ts,
                   "score": wscore, "verdict": vtext,
                   "criteria": {k: v[0] for k, v in cr_results.items()}}
        st.download_button("📊  JSON Summary", data=json.dumps(summary, indent=2),
                           file_name=f"FundaScore_{ticker}_{datetime.now():%Y%m%d}.json",
                           mime="application/json", use_container_width=True)


if __name__ == "__main__":
    main()
