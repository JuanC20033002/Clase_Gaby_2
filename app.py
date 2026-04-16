import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Risk Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://api.fontshare.com/v2/css?f[]=satoshi@400,500,700&display=swap');

    html, body, [class*="css"] { font-family: 'Satoshi', sans-serif; }

    .main { background-color: #0e1117; }

    .metric-card {
        background: #1c1f26;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-label {
        font-size: 12px;
        color: #7a7984;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #e8e8ea;
    }
    .metric-sub {
        font-size: 12px;
        color: #7a7984;
        margin-top: 4px;
    }

    .alert-red {
        background: rgba(220, 38, 38, 0.15);
        border: 1px solid rgba(220, 38, 38, 0.5);
        border-radius: 10px;
        padding: 16px 20px;
        color: #f87171;
        font-weight: 600;
        margin: 12px 0;
    }
    .alert-green {
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.4);
        border-radius: 10px;
        padding: 16px 20px;
        color: #4ade80;
        font-weight: 600;
        margin: 12px 0;
    }
    .alert-yellow {
        background: rgba(234, 179, 8, 0.12);
        border: 1px solid rgba(234, 179, 8, 0.4);
        border-radius: 10px;
        padding: 16px 20px;
        color: #facc15;
        font-weight: 600;
        margin: 12px 0;
    }
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #e8e8ea;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .report-box {
        background: #1c1f26;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 28px 32px;
        line-height: 1.8;
        color: #c8c8ca;
        font-size: 15px;
    }
    .report-box h4 {
        color: #4f98a3;
        font-size: 16px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 6px;
    }
    .report-box ul {
        padding-left: 20px;
        margin-top: 6px;
    }
    .report-box li {
        margin-bottom: 6px;
    }
    .tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 99px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .tag-red   { background: rgba(220,38,38,0.2);  color: #f87171; }
    .tag-green { background: rgba(34,197,94,0.2);  color: #4ade80; }
    .tag-blue  { background: rgba(79,152,163,0.2); color: #4f98a3; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
ASSETS = {
    "GOOGL — Alphabet (Acción)":         "GOOGL",
    "ADA-USD — Cardano (Criptomoneda)":  "ADA-USD",
    "NG=F — Gas Natural (Refugio)":      "NG=F",
}
ASSET_CLASS = {
    "GOOGL":   ("Acción",        "#4f98a3"),
    "ADA-USD": ("Criptomoneda",  "#a86fdf"),
    "NG=F":    ("Materia Prima", "#e8af34"),
}
RISK_FREE_RATE = 0.0525
VAR_ALERT_THRESHOLD = 0.03

# ── Helper functions ──────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def download_data(ticker: str, period_years: int) -> pd.Series:
    end   = datetime.today()
    start = end - timedelta(days=365 * period_years)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    close = df["Close"].squeeze().dropna()
    return close

def compute_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()

def annualised_volatility(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(252))

def var_parametric(returns: pd.Series, confidence: float) -> float:
    z = stats.norm.ppf(1 - confidence)
    return float(-(returns.mean() + z * returns.std()))

def max_drawdown(prices: pd.Series) -> float:
    roll_max = prices.cummax()
    dd       = (prices - roll_max) / roll_max
    return float(dd.min())

def sharpe_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    excess = returns.mean() * 252 - rf
    vol    = returns.std() * np.sqrt(252)
    return float(excess / vol) if vol != 0 else 0.0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parámetros")
    st.markdown("---")

    asset_label = st.selectbox("Activo", list(ASSETS.keys()), index=0)
    ticker      = ASSETS[asset_label]
    asset_class, accent_color = ASSET_CLASS[ticker]

    period_years = st.selectbox("Periodo histórico", [1, 2, 3], index=0,
                                format_func=lambda x: f"{x} año{'s' if x > 1 else ''}")

    confidence_label = st.selectbox("Nivel de confianza VaR",
                                    ["90%", "95%", "99%"], index=1)
    confidence       = float(confidence_label.strip("%")) / 100

    var_threshold = st.slider("Umbral de alerta VaR (%)", 1.0, 10.0,
                               VAR_ALERT_THRESHOLD * 100, 0.5) / 100

    st.markdown("---")
    st.markdown(f"**Clase:** {asset_class}")
    st.markdown(f"**Tasa libre de riesgo:** {RISK_FREE_RATE*100:.2f}%")
    st.markdown("---")
    st.markdown("<small style='color:#7a7984'>Datos via yfinance · Actualización horaria</small>",
                unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"# 📊 Risk Analytics Dashboard")
st.markdown(f"**Activo seleccionado:** `{ticker}` &nbsp;·&nbsp; {asset_class} &nbsp;·&nbsp; "
            f"Confianza VaR: **{confidence_label}** &nbsp;·&nbsp; Periodo: **{period_years} año(s)**")
st.markdown("---")

# ── Data load ─────────────────────────────────────────────────────────────────
with st.spinner(f"Descargando datos de {ticker}…"):
    try:
        prices  = download_data(ticker, period_years)
        returns = compute_returns(prices)
    except Exception as e:
        st.error(f"Error al descargar datos: {e}")
        st.stop()

if prices.empty or len(prices) < 50:
    st.error("No se obtuvieron suficientes datos. Intenta otro activo o periodo.")
    st.stop()

# ── Compute indicators ────────────────────────────────────────────────────────
vol     = annualised_volatility(returns)
var     = var_parametric(returns, confidence)
mdd     = max_drawdown(prices)
sharpe  = sharpe_ratio(returns)
ann_ret = float(returns.mean() * 252)

# ── VaR Alert ─────────────────────────────────────────────────────────────────
if var > var_threshold:
    st.markdown(
        f'<div class="alert-red">🚨 <strong>ALERTA DE RIESGO:</strong> '
        f'El VaR ({confidence_label}) es <strong>{var*100:.2f}%</strong>, '
        f'superando el umbral definido de {var_threshold*100:.1f}%. '
        f'Se recomienda revisar el tamaño de posición, implementar stop-loss o hedging.</div>',
        unsafe_allow_html=True)
elif var > var_threshold * 0.75:
    st.markdown(
        f'<div class="alert-yellow">⚠️ <strong>PRECAUCIÓN:</strong> '
        f'El VaR ({confidence_label}) es <strong>{var*100:.2f}%</strong>, '
        f'cercano al umbral de alerta de {var_threshold*100:.1f}%.</div>',
        unsafe_allow_html=True)
else:
    st.markdown(
        f'<div class="alert-green">✅ <strong>RIESGO CONTROLADO:</strong> '
        f'El VaR ({confidence_label}) es <strong>{var*100:.2f}%</strong>, '
        f'por debajo del umbral de {var_threshold*100:.1f}%.</div>',
        unsafe_allow_html=True)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">📐 Indicadores de Riesgo</p>', unsafe_allow_html=True)

def kpi(label, value, sub=""):
    return f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>"""

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(kpi("Volatilidad Anualizada", f"{vol*100:.2f}%",
                    "Dispersión de retornos"), unsafe_allow_html=True)
with c2:
    st.markdown(kpi(f"VaR {confidence_label}", f"{var*100:.2f}%",
                    "Pérdida máx. esperada"), unsafe_allow_html=True)
with c3:
    st.markdown(kpi("Máximo Drawdown", f"{mdd*100:.2f}%",
                    "Caída máx. del periodo"), unsafe_allow_html=True)
with c4:
    color_sharpe = "#4ade80" if sharpe > 1 else ("#facc15" if sharpe > 0 else "#f87171")
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Ratio de Sharpe</div>
        <div class="metric-value" style="color:{color_sharpe}">{sharpe:.3f}</div>
        <div class="metric-sub">Retorno ajustado al riesgo</div>
    </div>""", unsafe_allow_html=True)
with c5:
    color_ret = "#4ade80" if ann_ret > 0 else "#f87171"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Retorno Anualizado</div>
        <div class="metric-value" style="color:{color_ret}">{ann_ret*100:.2f}%</div>
        <div class="metric-sub">Media aritmética × 252</div>
    </div>""", unsafe_allow_html=True)

# ── Charts ────────────────────────────────────────────────────────────────────
CHART_SURF = "#1c1f26"
GRID_COLOR = "rgba(255,255,255,0.06)"
TEXT_COLOR = "#9a9aa0"

def base_layout(title=""):
    return dict(
        title=dict(text=title, font=dict(size=15, color="#e8e8ea"), x=0.01),
        paper_bgcolor=CHART_SURF,
        plot_bgcolor=CHART_SURF,
        font=dict(family="Satoshi, sans-serif", color=TEXT_COLOR, size=12),
        xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, zerolin
