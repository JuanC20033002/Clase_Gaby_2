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
        xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
    )

# ─── Comparativa: Precio + Histograma ────────────────────────────────────────
st.markdown('<p class="section-header">📈 Comparativa: Precio vs. Distribución de Retornos</p>',
            unsafe_allow_html=True)

fig_comp = make_subplots(
    rows=1, cols=2,
    column_widths=[0.62, 0.38],
    subplot_titles=["Precio de Cierre", f"Distribución de Retornos Diarios (VaR {confidence_label})"],
)

fig_comp.add_trace(
    go.Scatter(x=prices.index, y=prices.values,
               mode="lines", name="Precio",
               line=dict(color=accent_color, width=1.8),
               fill="tozeroy",
               fillcolor=f"rgba({int(accent_color[1:3],16)},{int(accent_color[3:5],16)},{int(accent_color[5:7],16)},0.08)"),
    row=1, col=1)

ret_vals  = returns.values
bin_count = min(80, max(40, len(ret_vals) // 10))
fig_comp.add_trace(
    go.Histogram(x=ret_vals, nbinsx=bin_count,
                 name="Retornos", marker_color=accent_color, opacity=0.7),
    row=1, col=2)

x_range  = np.linspace(ret_vals.min(), ret_vals.max(), 300)
pdf_vals = stats.norm.pdf(x_range, ret_vals.mean(), ret_vals.std())
scale    = len(ret_vals) * (ret_vals.max() - ret_vals.min()) / bin_count
fig_comp.add_trace(
    go.Scatter(x=x_range, y=pdf_vals * scale,
               mode="lines", name="Dist. Normal",
               line=dict(color="#e8af34", width=2, dash="dot")),
    row=1, col=2)

var_val = returns.mean() - stats.norm.ppf(confidence) * returns.std()
fig_comp.add_vline(x=var_val, line_width=2, line_dash="dash",
                   line_color="#ef4444",
                   annotation_text=f"VaR {confidence_label}",
                   annotation_font_color="#ef4444",
                   row=1, col=2)

layout = base_layout()
layout.update(height=380, showlegend=False,
              paper_bgcolor=CHART_SURF, plot_bgcolor=CHART_SURF)
fig_comp.update_layout(**layout)
fig_comp.update_xaxes(gridcolor=GRID_COLOR, linecolor=GRID_COLOR)
fig_comp.update_yaxes(gridcolor=GRID_COLOR, linecolor=GRID_COLOR)
st.plotly_chart(fig_comp, use_container_width=True)

# ─── Drawdown ─────────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">📉 Drawdown Histórico</p>', unsafe_allow_html=True)

roll_max        = prices.cummax()
drawdown_series = (prices - roll_max) / roll_max * 100

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=drawdown_series.index, y=drawdown_series.values,
    mode="lines", name="Drawdown",
    line=dict(color="#ef4444", width=1.5),
    fill="tozeroy", fillcolor="rgba(239,68,68,0.12)"
))
fig_dd.add_hline(y=mdd * 100, line_dash="dash",
                 line_color="#f87171", line_width=1.5,
                 annotation_text=f"Máx. Drawdown: {mdd*100:.2f}%",
                 annotation_font_color="#f87171",
                 annotation_position="bottom right")
dd_layout = base_layout("Drawdown (%)")
dd_layout.update(height=280, showlegend=False)
fig_dd.update_layout(**dd_layout)
st.plotly_chart(fig_dd, use_container_width=True)

# ─── Volatilidad Rodante + VaR por nivel ─────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<p class="section-header">🌡️ Volatilidad Rodante (30d)</p>',
                unsafe_allow_html=True)
    roll_vol = returns.rolling(30).std() * np.sqrt(252) * 100
    fig_rv = go.Figure()
    fig_rv.add_trace(go.Scatter(
        x=roll_vol.index, y=roll_vol.values,
        mode="lines", name="Vol 30d",
        line=dict(color="#a86fdf", width=1.8),
        fill="tozeroy", fillcolor="rgba(168,111,223,0.10)"
    ))
    fig_rv.add_hline(y=vol * 100, line_dash="dot",
                     line_color="#facc15", line_width=1.5,
                     annotation_text=f"Promedio: {vol*100:.2f}%",
                     annotation_font_color="#facc15")
    rv_layout = base_layout("Volatilidad Anualizada Rodante (%)")
    rv_layout.update(height=280, showlegend=False)
    fig_rv.update_layout(**rv_layout)
    st.plotly_chart(fig_rv, use_container_width=True)

with col_right:
    st.markdown('<p class="section-header">📊 VaR por Nivel de Confianza</p>',
                unsafe_allow_html=True)
    confidence_levels = [0.80, 0.85, 0.90, 0.95, 0.99]
    var_values  = [var_parametric(returns, c) * 100 for c in confidence_levels]
    colors_bar  = ["#4f98a3" if v < var_threshold * 100 else "#ef4444" for v in var_values]

    fig_var = go.Figure(go.Bar(
        x=[f"{int(c*100)}%" for c in confidence_levels],
        y=var_values,
        marker_color=colors_bar,
        text=[f"{v:.2f}%" for v in var_values],
        textposition="outside",
        textfont=dict(color="#e8e8ea", size=11)
    ))
    fig_var.add_hline(y=var_threshold * 100, line_dash="dash",
                      line_color="#facc15", line_width=1.5,
                      annotation_text=f"Umbral {var_threshold*100:.1f}%",
                      annotation_font_color="#facc15")
    var_layout = base_layout("VaR Paramétrico por Nivel de Confianza")
    var_layout.update(height=280, showlegend=False, yaxis_title="Pérdida Máxima (%)")
    fig_var.update_layout(**var_layout)
    st.plotly_chart(fig_var, use_container_width=True)

# ─── Reporte de Análisis ──────────────────────────────────────────────────────
st.markdown('<p class="section-header">📝 Reporte de Análisis Comparativo</p>',
            unsafe_allow_html=True)

st.markdown("""
<div class="report-box">

<p>
Este reporte compara los perfiles de riesgo de tres activos distintos:
<strong>GOOGL</strong> (acción tecnológica de gran capitalización),
<strong>ADA-USD</strong> (criptomoneda de mediana capitalización) y
<strong>NG=F</strong> (futuros de Gas Natural, materia prima con comportamiento de refugio parcial).
El análisis se fundamenta en cuatro métricas cuantitativas calculadas sobre datos históricos reales.
</p>

<h4>🔵 GOOGL — Alphabet Inc. (Acción Tecnológica)</h4>
<ul>
  <li><strong>Volatilidad:</strong> Moderada (~25–35% anualizada). Como acción de mega-cap en el S&P 500,
  GOOGL presenta volatilidad acotada comparada con activos especulativos.</li>
  <li><strong>VaR 95%:</strong> Típicamente entre 2–4% diario, manejable dentro de un portafolio diversificado.</li>
  <li><strong>Máximo Drawdown:</strong> Puede superar el −40% en correcciones tecnológicas (ej. 2022),
  representando un riesgo de recuperación importante.</li>
  <li><strong>Sharpe Ratio:</strong> Históricamente positivo (>1 en bull markets). El retorno compensa razonablemente el riesgo.</li>
  <li><strong>Conclusión:</strong> <span class="tag tag-blue">Riesgo Moderado</span>
  Adecuada para portafolios con horizonte largo y tolerancia media al riesgo.</li>
</ul>

<h4>🟣 ADA-USD — Cardano (Criptomoneda)</h4>
<ul>
  <li><strong>Volatilidad:</strong> Muy alta (>80–120% anualizada). Los swings extremos son amplificados
  por baja liquidez relativa y sentimiento especulativo.</li>
  <li><strong>VaR 95%:</strong> Frecuentemente supera el 8–12% diario — en 1 de cada 20 días
  se puede perder más del 10% del capital.</li>
  <li><strong>Máximo Drawdown:</strong> Histórico superior al −90% (máximos 2021 → mínimos 2022),
  el más severo entre los tres activos.</li>
  <li><strong>Sharpe Ratio:</strong> Altamente variable; negativo en ciclos bajistas.</li>
  <li><strong>Conclusión:</strong> <span class="tag tag-red">Mayor Riesgo</span>
  Solo apropiado para inversores con alta tolerancia al riesgo y asignación reducida del portafolio.</li>
</ul>

<h4>🟡 NG=F — Futuros de Gas Natural (Materia Prima)</h4>
<ul>
  <li><strong>Volatilidad:</strong> Alta y estacional (50–80% anualizada), impulsada por factores
  climáticos, geopolíticos y de oferta/demanda energética.</li>
  <li><strong>VaR 95%:</strong> Entre 4–8% diario. Los movimientos intradía pueden ser drásticos
  por eventos exógenos (clima, inventarios EIA, conflictos).</li>
  <li><strong>Máximo Drawdown:</strong> Caídas del −70% no son infrecuentes en ciclos de sobreoferta (ej. 2023–2024).</li>
  <li><strong>Sharpe Ratio:</strong> Generalmente bajo o negativo, aunque su baja correlación con renta
  variable lo hace valioso como cobertura táctica.</li>
  <li><strong>Conclusión:</strong> <span class="tag tag-green">Riesgo Alto / Diversificador</span>
  Perfil de riesgo elevado con poca compensación por retorno, útil como hedge táctico.</li>
</ul>

<h4>⚖️ Ranking de Riesgo (Mayor a Menor)</h4>
<ol>
  <li><strong>ADA-USD</strong> — Riesgo más alto en todas las métricas: volatilidad extrema,
  VaR máximo, drawdowns devastadores y Sharpe inconsistente.</li>
  <li><strong>NG=F</strong> — Alta volatilidad estructural con picos estacionales y bajo Sharpe sistemático.</li>
  <li><strong>GOOGL</strong> — Menor riesgo relativo: volatilidad manejable, Sharpe positivo sostenido
  y respaldo fundamental sólido.</li>
</ol>

<p style="margin-top:16px; color:#7a7984; font-size:13px;">
⚠️ <em>Este análisis tiene fines educativos. No constituye asesoría financiera.
Los rendimientos pasados no garantizan resultados futuros.</em>
</p>

</div>
""", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<small style='color:#4a4a52'>Datos descargados: "
    f"{prices.index[0].date()} → {prices.index[-1].date()} &nbsp;·&nbsp; "
    f"{len(prices):,} observaciones &nbsp;·&nbsp; "
    f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M')}</small>",
    unsafe_allow_html=True
)
