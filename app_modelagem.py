# =============================================================================
# CALCULADORA DE VaR — CARTEIRA DE AÇÕES E OPÇÕES
# Projeto Final: Modelagem Aplicada ao Mercado Financeiro
# =============================================================================
# COMO RODAR:
#   pip install -r requirements.txt
#   streamlit run app_modelagem.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="VaR Calculator | Modelagem Financeira",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Tema visual: teal/esmeralda — diferente do azul navy do projeto anterior
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0B1120; }
    [data-testid="stSidebar"]          { display: none; }
    [data-testid="collapsedControl"]   { display: none; }
    [data-testid="stMetric"]           { background-color: #0D1F2D; border: 1px solid #1E3A4A;
                                         border-radius: 8px; padding: 12px; }
    h1, h2, h3                         { color: #E2F8F0 !important; }
    p, li, label                       { color: #94A3B8; }
    .stDataFrame                       { border: 1px solid #1E3A4A; border-radius: 8px; }
    div[data-testid="stMetricValue"]   { color: #F1F5F9 !important; font-size: 1.6rem !important; }
    div[data-testid="stMetricLabel"]   { color: #64748B !important; }
    /* Estilo das abas */
    .stTabs [data-baseweb="tab-list"]  { background-color: #0D1627; border-radius: 8px;
                                         padding: 4px; gap: 4px; }
    .stTabs [data-baseweb="tab"]       { background-color: transparent; color: #64748B;
                                         border-radius: 6px; padding: 6px 16px; font-weight: 500; }
    .stTabs [aria-selected="true"]     { background-color: #1E3A8A !important; color: #FFFFFF !important;
                                         font-weight: 700; }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================
if "carteira" not in st.session_state:
    st.session_state["carteira"] = None      # DataFrame com posições

if "retornos" not in st.session_state:
    st.session_state["retornos"] = None      # DataFrame de retornos históricos

if "params" not in st.session_state:
    st.session_state["params"] = {
        "confianca":  0.95,
        "horizonte":  1,
        "janela":     252,
        "n_sim":      10_000,
    }

if "resultados" not in st.session_state:
    st.session_state["resultados"] = {}      # dict com VaR por metodologia


# =============================================================================
# FUNÇÕES — FINANÇAS
# =============================================================================

def baixar_retornos(tickers: list, janela: int = 252) -> pd.DataFrame:
    """Baixa retornos diários históricos do Yahoo Finance."""
    frames = {}
    for t in tickers:
        try:
            df = yf.download(t, period=f"{int(janela*1.5)}d",
                             progress=False, auto_adjust=True)
            if df.empty:
                continue
            # Compatível com yfinance >= 0.2.x que retorna MultiIndex
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.squeeze()
            ret = close.pct_change().dropna()
            if isinstance(ret, pd.Series) and len(ret) > 10:
                frames[t] = ret
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    # Alinha pelo índice antes de montar o DataFrame
    ret_df = pd.concat(frames, axis=1)
    ret_df.columns = list(frames.keys())
    ret_df = ret_df.dropna()
    return ret_df.tail(janela)


def black_scholes_preco(S, K, T, r, sigma, tipo):
    """Preço Black-Scholes para opção europeia (Call ou Put)."""
    if T <= 0:
        return max(0.0, S - K) if tipo == "Call" else max(0.0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if tipo == "Call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def black_scholes_delta(S, K, T, r, sigma, tipo):
    """Delta da opção pelo modelo Black-Scholes."""
    if T <= 0:
        return 1.0 if tipo == "Call" else -1.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if tipo == "Call" else norm.cdf(d1) - 1.0


def black_scholes_greeks(S, K, T, r, sigma, tipo):
    """Calcula Delta, Gamma, Vega, Theta para fins informativos."""
    if T <= 0:
        return {"Delta": 0, "Gamma": 0, "Vega": 0, "Theta": 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if tipo == "Call" else norm.cdf(d1) - 1.0
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100
    if tipo == "Call":
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 252
    else:
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 252
    return {"Delta": round(delta, 4), "Gamma": round(gamma, 6),
            "Vega": round(vega, 4), "Theta": round(theta, 4)}


def calcular_var_parametrico(retornos_df, carteira_df, confianca, horizonte):
    """
    VaR Paramétrico com matriz de covariância.
    VaR = z_α × σ_p × V × √T
    σ_p = √(w' Σ w)
    """
    tickers_acoes = carteira_df[carteira_df["Tipo"] == "Ação"]["Ativo"].tolist()
    ret_acoes = retornos_df[[t for t in tickers_acoes if t in retornos_df.columns]]

    if ret_acoes.empty:
        return 0.0, pd.DataFrame(), pd.DataFrame()

    valores = []
    for t in ret_acoes.columns:
        row = carteira_df[carteira_df["Ativo"] == t]
        if not row.empty:
            valores.append(float(row["Valor da Posição"].values[0]))
        else:
            valores.append(0.0)

    V_total = sum(valores)
    if V_total == 0:
        return 0.0, pd.DataFrame(), pd.DataFrame()

    w = np.array(valores) / V_total

    # Matriz de covariância e correlação
    cov_matrix  = ret_acoes.cov()
    corr_matrix = ret_acoes.corr()

    sigma_p = np.sqrt(w @ cov_matrix.values @ w)
    z       = norm.ppf(confianca)
    var     = z * sigma_p * V_total * np.sqrt(horizonte)

    return max(0.0, var), cov_matrix, corr_matrix


def calcular_var_historico(retornos_df, carteira_df, confianca, horizonte):
    """
    VaR Histórico: percentil empírico dos retornos da carteira ponderada.
    """
    tickers_acoes = carteira_df[carteira_df["Tipo"] == "Ação"]["Ativo"].tolist()
    ret_acoes = retornos_df[[t for t in tickers_acoes if t in retornos_df.columns]]

    if ret_acoes.empty:
        return 0.0, np.array([])

    valores = []
    for t in ret_acoes.columns:
        row = carteira_df[carteira_df["Ativo"] == t]
        valores.append(float(row["Valor da Posição"].values[0]) if not row.empty else 0.0)

    V_total = sum(valores)
    if V_total == 0:
        return 0.0, np.array([])

    w = np.array(valores) / V_total
    ret_carteira = ret_acoes.values @ w  # retorno diário da carteira

    alpha = 1.0 - confianca
    var_1d = -np.percentile(ret_carteira, alpha * 100) * V_total
    var    = max(0.0, var_1d * np.sqrt(horizonte))

    return var, ret_carteira


def calcular_var_full_valuation(retornos_df, carteira_df, confianca,
                                horizonte, n_sim, r=0.105):
    """
    VaR Full Valuation: reprecificação completa da carteira em cada cenário.

    Para ações: S_T = S_0 × exp[(μ - ½σ²)T + σ√T × Z]
    Para opções: reprecifica pelo Black-Scholes em cada cenário simulado.
    """
    np.random.seed(42)
    perdas_simuladas = np.zeros(n_sim)
    V_atual = 0.0

    for _, row in carteira_df.iterrows():
        ativo    = row["Ativo"]
        tipo     = row["Tipo"]
        qtd      = float(row["Quantidade"])
        S0       = float(row["Preço"])
        val_pos  = float(row["Valor da Posição"])
        V_atual += val_pos

        # Retornos do ativo para estimar μ e σ
        if ativo in retornos_df.columns:
            ret_hist = retornos_df[ativo].values
        else:
            ret_hist = np.random.normal(0.0004, 0.019, 252)

        mu    = np.mean(ret_hist)
        sigma = np.std(ret_hist, ddof=1)
        vol_anual = sigma * np.sqrt(252)

        Z  = np.random.standard_normal(n_sim)
        ST = S0 * np.exp((mu - 0.5 * sigma**2) * horizonte
                         + sigma * np.sqrt(horizonte) * Z)

        if tipo == "Ação":
            # P&L da ação
            pl = (ST - S0) * qtd
        elif tipo in ("Opção Call", "Opção Put"):
            K    = float(row.get("Strike", S0 * 1.05)  if pd.notna(row.get("Strike")) else S0 * 1.05)
            dias = float(row.get("Vencimento", 30)      if pd.notna(row.get("Vencimento")) else 30)
            T0   = dias / 252
            T1   = max(T0 - horizonte / 252, 0.001)
            tipo_bs = "Call" if tipo == "Opção Call" else "Put"

            # Reprecificação completa: P_0 e P_T para cada cenário
            P0 = black_scholes_preco(S0, K, T0, r, vol_anual, tipo_bs)
            P1 = np.array([black_scholes_preco(st, K, T1, r, vol_anual, tipo_bs)
                           for st in ST])
            pl = (P1 - P0) * qtd
        else:
            pl = np.zeros(n_sim)

        perdas_simuladas -= pl   # perda = ganho negativo

    alpha = 1.0 - confianca
    var   = np.percentile(perdas_simuladas, (1 - alpha) * 100)
    return max(0.0, var), perdas_simuladas, V_atual


def calcular_rolling_var(retornos_df, carteira_df, confianca,
                         janela_rolling=60, horizonte=1):
    """VaR histórico rolante para visualizar evolução no tempo."""
    tickers_acoes = carteira_df[carteira_df["Tipo"] == "Ação"]["Ativo"].tolist()
    ret_acoes = retornos_df[[t for t in tickers_acoes if t in retornos_df.columns]]
    if ret_acoes.empty or len(ret_acoes) < janela_rolling:
        return pd.Series(dtype=float)

    valores = []
    for t in ret_acoes.columns:
        row = carteira_df[carteira_df["Ativo"] == t]
        valores.append(float(row["Valor da Posição"].values[0]) if not row.empty else 0.0)

    V_total = sum(valores)
    if V_total == 0:
        return pd.Series(dtype=float)

    w = np.array(valores) / V_total
    ret_carteira = pd.Series(ret_acoes.values @ w, index=ret_acoes.index)
    alpha = 1.0 - confianca

    rolling_var = ret_carteira.rolling(janela_rolling).apply(
        lambda x: -np.percentile(x, alpha * 100) * V_total * np.sqrt(horizonte),
        raw=True
    )
    return rolling_var.dropna()


def stress_test(retornos_df, carteira_df):
    """
    Aplica choques de estresse na carteira e calcula P&L em cada cenário.
    Cenários: Crise 2008, COVID 2020, Alta de Juros, Choque Cambial.
    """
    tickers_acoes = carteira_df[carteira_df["Tipo"] == "Ação"]["Ativo"].tolist()
    ret_acoes = retornos_df[[t for t in tickers_acoes if t in retornos_df.columns]]

    cenarios = {
        "Crise 2008 (-40%)":    -0.40,
        "COVID Mar/2020 (-35%)": -0.35,
        "Alta Juros (-15%)":    -0.15,
        "Choque Cambial (-20%)": -0.20,
        "Cenario Leve (-10%)":  -0.10,
        "Cenario Positivo (+15%)": +0.15,
    }

    resultados = {}
    V_total = carteira_df["Valor da Posição"].sum()

    for nome, choque in cenarios.items():
        valores = []
        for t in ret_acoes.columns:
            row = carteira_df[carteira_df["Ativo"] == t]
            if not row.empty:
                valores.append(float(row["Valor da Posição"].values[0]))
        pl = sum(v * choque for v in valores)
        resultados[nome] = {"P&L (R$)": pl, "% Carteira": pl / V_total * 100}

    return pd.DataFrame(resultados).T


# =============================================================================
# HEADER + NAVEGAÇÃO POR ABAS NO TOPO
# =============================================================================
p = st.session_state["params"]
status_carteira = "✅ carregada" if st.session_state["carteira"] is not None else "❌"
status_var      = "✅ calculado" if st.session_state["resultados"]            else "❌"

col_title, col_status = st.columns([3, 2])
with col_title:
    st.markdown("## 📉 Calculadora de VaR — Modelagem Aplicada ao Mercado Financeiro")
with col_status:
    st.markdown(
        f"<div style='text-align:right; color:#64748B; font-size:0.82rem; padding-top:14px;'>"
        f"Confiança: <b style='color:#F1F5F9'>{p['confianca']*100:.0f}%</b> &nbsp;|&nbsp; "
        f"Horizonte: <b style='color:#F1F5F9'>{p['horizonte']}d</b> &nbsp;|&nbsp; "
        f"Janela: <b style='color:#F1F5F9'>{p['janela']}d</b> &nbsp;|&nbsp; "
        f"Carteira: <b style='color:#F1F5F9'>{status_carteira}</b> &nbsp;|&nbsp; "
        f"VaR: <b style='color:#F1F5F9'>{status_var}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.divider()

(tab_inicio, tab_carteira, tab_params,
 tab_param_var, tab_hist, tab_fv, tab_analise) = st.tabs([
    "🏠 Início",
    "📈 Carteira",
    "⚙️ Parâmetros",
    "📐 VaR Paramétrico",
    "📜 VaR Histórico",
    "🔄 Full Valuation",
    "📊 Análise de Risco",
])


# =============================================================================
# PÁGINA 1 — INÍCIO
# =============================================================================
with tab_inicio:
    st.title("📉 Calculadora de VaR — Carteira de Ações e Opções")
    st.markdown("""
    Aplicação para **cálculo e análise de risco de mercado** de carteiras compostas
    por ações e opções utilizando diferentes metodologias de Value at Risk.
    """)

    col1, col2, col3 = st.columns(3)
    col1.info("**📐 VaR Paramétrico**\nMatriz de covariância e distribuição normal dos retornos.")
    col2.info("**📜 VaR Histórico**\nDistribuição empírica real, sem hipóteses paramétricas.")
    col3.info("**🔄 Full Valuation**\nReprecificação completa via Black-Scholes + Monte Carlo.")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("O que é VaR?")
        st.markdown("""
        O **Value at Risk (VaR)** estima a perda máxima esperada de uma carteira
        em um horizonte de tempo, dado um nível de confiança:

        > *"Com 95% de confiança, a carteira não perderá mais de R$ X em 1 dia."*

        É amplamente utilizado em **bancos**, **assets**, **hedge funds** e
        **tesourarias** como principal métrica de risco de mercado.
        """)

    with col2:
        st.subheader("Aplicações Profissionais")
        st.markdown("""
        - **Bancos**: limite de risco por mesa de trading e capital regulatório (Basileia III)
        - **Hedge Funds**: monitoramento diário de portfólios alavancados
        - **Assets**: controle de risco de fundos de investimento
        - **Tesourarias**: gestão de risco cambial e de taxa de juros
        - **Regulatório**: exigência de capital mínimo pelo Banco Central
        """)

    st.divider()
    st.subheader("Fluxo de uso")
    cols = st.columns(5)
    etapas = ["1. Montar Carteira", "2. Configurar Parâmetros",
              "3. Calcular VaR", "4. Analisar Resultados", "5. Comparar Métodos"]
    for c, e in zip(cols, etapas):
        c.success(e)


# =============================================================================
# PÁGINA 2 — CONSTRUÇÃO DA CARTEIRA
# =============================================================================
with tab_carteira:
    st.title("📈 Construção da Carteira")
    st.markdown("Monte sua carteira escolhendo ativos, quantidades e preços — ou use o **exemplo padrão**.")

    TICKERS_SUGERIDOS = [
        "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "WEGE3.SA",
        "MGLU3.SA", "RENT3.SA", "ABEV3.SA", "AAPL", "MSFT", "NVDA",
    ]

    tab_manual, tab_exemplo = st.tabs(["✏️ Montar manualmente", "🗂️ Exemplo padrão"])

    with tab_exemplo:
        exemplo = pd.DataFrame({
            "Ativo":            ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA", "PETR4.SA", "VALE3.SA"],
            "Tipo":             ["Ação",     "Ação",     "Ação",     "Ação",     "Opção Call", "Opção Put"],
            "Quantidade":       [100_000,    80_000,     120_000,    75_000,     50_000,       30_000],
            "Preço":            [38.50,      68.20,      32.10,      45.30,      38.50,        68.20],
            "Valor da Posição": [3_850_000,  5_456_000,  3_852_000,  3_397_500,  1_925_000,    2_046_000],
            "Strike":           [None,       None,       None,       None,       42.0,         65.0],
            "Vencimento":       [None,       None,       None,       None,       30.0,         30.0],
        })

        if st.button("📥 Carregar exemplo padrão", type="primary"):
            st.session_state["carteira"] = exemplo
            st.success("✅ Carteira de exemplo carregada!")

        if st.session_state["carteira"] is not None:
            st.dataframe(st.session_state["carteira"], use_container_width=True, hide_index=True)

    with tab_manual:
        st.markdown("#### Adicionar posição")
        col1, col2, col3 = st.columns(3)
        with col1:
            ticker_input = st.text_input("Ticker", placeholder="Ex: PETR4.SA")
            tipo_input = st.selectbox("Tipo", ["Ação", "Opção Call", "Opção Put"])
        with col2:
            qtd_input   = st.number_input("Quantidade", min_value=1, value=10_000, step=1_000)
            preco_input = st.number_input("Preço (R$)", min_value=0.01, value=10.0, step=0.01)
        with col3:
            strike_input = st.number_input("Strike (só opções)", min_value=0.0, value=0.0) if tipo_input != "Ação" else None
            venc_input   = st.number_input("Vencimento (dias úteis)", min_value=1, value=30) if tipo_input != "Ação" else None

        if st.button("➕ Adicionar à carteira", type="primary"):
            nova = {
                "Ativo":            ticker_input.upper(),
                "Tipo":             tipo_input,
                "Quantidade":       qtd_input,
                "Preço":            preco_input,
                "Valor da Posição": qtd_input * preco_input,
                "Strike":           strike_input,
                "Vencimento":       venc_input,
            }
            df_atual = st.session_state["carteira"] if st.session_state["carteira"] is not None else pd.DataFrame()
            st.session_state["carteira"] = pd.concat([df_atual, pd.DataFrame([nova])], ignore_index=True)
            st.success(f"✅ {ticker_input.upper()} adicionado!")

        if st.session_state["carteira"] is not None and not st.session_state["carteira"].empty:
            st.divider()
            st.subheader("Carteira atual")
            df_show = st.session_state["carteira"].copy()
            st.dataframe(df_show, use_container_width=True, hide_index=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Posições", len(df_show))
            col2.metric("Valor Total", f"R$ {df_show['Valor da Posição'].sum():,.0f}")
            col3.metric("Ativos únicos", df_show["Ativo"].nunique())

            if st.button("🗑️ Limpar carteira", type="secondary"):
                st.session_state["carteira"] = None
                st.session_state["retornos"] = None
                st.session_state["resultados"] = {}
                st.rerun()


# =============================================================================
# PÁGINA 3 — PARÂMETROS
# =============================================================================
with tab_params:
    st.title("⚙️ Parâmetros do Modelo")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Nível de Confiança")
        confianca_opcao = st.radio(
            "Opções comuns:",
            options=[0.90, 0.95, 0.99],
            format_func=lambda x: f"{x*100:.0f}%",
            horizontal=True,
            index=[0.90, 0.95, 0.99].index(st.session_state["params"]["confianca"]),
        )
        confianca_custom = st.text_input(
            "Ou digite um valor entre 0.90 e 0.9999:",
            placeholder="Ex: 0.975",
        )
        try:
            confianca = float(confianca_custom) if confianca_custom.strip() else confianca_opcao
            confianca = max(0.80, min(0.9999, confianca))
        except ValueError:
            confianca = confianca_opcao
            st.caption("Valor inválido — usando opção selecionada.")

        msgs = {0.90: "1 excesso a cada 10 dias.", 0.95: "Padrão de mercado. 1 excesso a cada 20 dias.", 0.99: "Regulatório (Basileia). 1 excesso a cada 100 dias."}
        st.info(f"📌 {confianca*100:.2f}% — {msgs.get(round(confianca,2), 'nível personalizado.')}")

        st.subheader("Horizonte de Tempo")
        horizonte_opcao = st.radio(
            "Opções comuns:",
            options=[1, 5, 10, 21],
            format_func=lambda x: f"{x}d",
            horizontal=True,
            index=[1, 5, 10, 21].index(st.session_state["params"]["horizonte"]),
        )
        horizonte_custom = st.text_input(
            "Ou digite o horizonte em dias:",
            placeholder="Ex: 15",
        )
        try:
            horizonte = int(horizonte_custom) if horizonte_custom.strip() else horizonte_opcao
            horizonte = max(1, min(252, horizonte))
        except ValueError:
            horizonte = horizonte_opcao
            st.caption("Valor inválido — usando opção selecionada.")
        st.caption(f"Escalamento pela raiz do tempo: √{horizonte} ≈ {np.sqrt(horizonte):.3f}")

    with col2:
        st.subheader("Janela Histórica")
        janela_opcao = st.radio(
            "Opções comuns:",
            options=[126, 252, 504],
            format_func=lambda x: f"{x} dias (~{x//252}a)" if x >= 252 else f"{x} dias (~6m)",
            horizontal=True,
            index=1,
        )
        janela_custom = st.text_input(
            "Ou digite a janela em dias úteis:",
            placeholder="Ex: 180",
        )
        try:
            janela = int(janela_custom) if janela_custom.strip() else janela_opcao
            janela = max(30, min(1260, janela))
        except ValueError:
            janela = janela_opcao
            st.caption("Valor inválido — usando opção selecionada.")
        st.caption("252 = 1 ano útil. Janelas maiores capturam mais ciclos.")

        st.subheader("Simulações — Full Valuation")
        n_sim_opcao = st.radio(
            "Opções comuns:",
            options=[1_000, 5_000, 10_000, 50_000],
            format_func=lambda x: f"{x:,}",
            horizontal=True,
            index=2,
        )
        n_sim_custom = st.text_input(
            "Ou digite o número de simulações:",
            placeholder="Ex: 20000",
        )
        try:
            n_sim = int(n_sim_custom) if n_sim_custom.strip() else n_sim_opcao
            n_sim = max(100, min(100_000, n_sim))
        except ValueError:
            n_sim = n_sim_opcao
            st.caption("Valor inválido — usando opção selecionada.")
        st.caption("Mais simulações = mais preciso, porém mais lento.")

    st.divider()
    if st.button("💾 Salvar parâmetros", type="primary"):
        st.session_state["params"] = {
            "confianca": confianca,
            "horizonte": horizonte,
            "janela":    janela,
            "n_sim":     n_sim,
        }
        st.session_state["retornos"]   = None
        st.session_state["resultados"] = {}
        st.success(f"✅ Parâmetros salvos: {confianca*100:.2f}% | {horizonte} dia(s) | {janela} dias | {n_sim:,} sim.")


# =============================================================================
# HELPER: garante que retornos estão carregados
# =============================================================================
def garantir_retornos():
    if st.session_state["carteira"] is None:
        st.warning("⚠️ Monte a carteira primeiro em **📈 Construção da Carteira**.")
        return False
    if st.session_state["retornos"] is None:
        tickers = st.session_state["carteira"]["Ativo"].unique().tolist()
        janela  = st.session_state["params"]["janela"]
        with st.spinner("Baixando dados do Yahoo Finance..."):
            ret = baixar_retornos(tickers, janela)
        if ret.empty:
            st.error("Não foi possível baixar dados. Verifique os tickers.")
            return False
        st.session_state["retornos"] = ret
    return True


# =============================================================================
# PÁGINA 4 — VaR PARAMÉTRICO
# =============================================================================
with tab_param_var:
    st.title("📐 VaR Paramétrico")
    st.markdown("""
    Assume que os retornos seguem **distribuição normal**.
    Utiliza a **matriz de covariância** para capturar correlações entre ativos.

    $$VaR = Z_\\alpha \\cdot \\sigma_p \\cdot V \\cdot \\sqrt{T}$$
    $$\\sigma_p = \\sqrt{\\mathbf{w}^\\top \\Sigma \\mathbf{w}}$$
    """)

    if not garantir_retornos():
        st.info("⬅️ Vá para a aba **📈 Carteira** para montar a carteira primeiro.")
    else:
        p   = st.session_state["params"]
        ret = st.session_state["retornos"]
        cart = st.session_state["carteira"]

        var, cov_matrix, corr_matrix = calcular_var_parametrico(
            ret, cart, p["confianca"], p["horizonte"]
        )

        st.session_state["resultados"]["Paramétrico"] = var

        # KPIs
        V_total = cart["Valor da Posição"].sum()
        z       = norm.ppf(p["confianca"])
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💼 Valor da Carteira", f"R$ {V_total:,.0f}")
        col2.metric("📐 VaR Paramétrico",   f"R$ {var:,.0f}")
        col3.metric("📊 VaR / Carteira",    f"{var/V_total*100:.2f}%")
        col4.metric("Z Score", f"{z:.3f}")

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Matriz de Correlação")
            if not corr_matrix.empty:
                fig_corr = px.imshow(
                    corr_matrix.round(3),
                    color_continuous_scale="Teal",
                    text_auto=True,
                    aspect="auto",
                )
                fig_corr.update_layout(
                    paper_bgcolor="#0B1120", plot_bgcolor="#0B1120",
                    font_color="#E2F8F0", height=350,
                )
                st.plotly_chart(fig_corr, use_container_width=True)

        with col2:
            st.subheader("Volatilidade por Ativo")
            tickers_acoes = cart[cart["Tipo"].str.contains("Ação", na=False)]["Ativo"].tolist()
            ret_acoes = ret[[t for t in tickers_acoes if t in ret.columns]]
            if not ret_acoes.empty:
                vols = ret_acoes.std() * np.sqrt(252) * 100
                fig_vol = go.Figure(go.Bar(
                    x=vols.index, y=vols.values,
                    marker_color="#F1F5F9",
                    text=[f"{v:.1f}%" for v in vols.values],
                    textposition="outside",
                ))
                fig_vol.update_layout(
                    yaxis_title="Volatilidade Anual (%)",
                    paper_bgcolor="#0B1120", plot_bgcolor="#0B1120",
                    font_color="#E2F8F0", height=350,
                    yaxis=dict(gridcolor="#1E3A4A"),
                    xaxis=dict(gridcolor="#1E3A4A"),
                )
                st.plotly_chart(fig_vol, use_container_width=True)

        st.subheader("Matriz de Covariância")
        if not cov_matrix.empty:
            st.dataframe(
                cov_matrix.style.format("{:.6f}"),
                use_container_width=True,
            )


# =============================================================================
# PÁGINA 5 — VaR HISTÓRICO
# =============================================================================
with tab_hist:
    st.title("📜 VaR Histórico")
    st.markdown("""
    Usa a **distribuição empírica** dos retornos passados.
    Não assume nenhuma forma paramétrica — o percentil é extraído diretamente dos dados.

    $$VaR_\\alpha = -Percentil_\\alpha(R_p) \\times V \\times \\sqrt{T}$$
    """)

    if not garantir_retornos():
        st.info("⬅️ Vá para a aba **📈 Carteira** para montar a carteira primeiro.")
    else:
        p    = st.session_state["params"]
        ret  = st.session_state["retornos"]
        cart = st.session_state["carteira"]

        var, ret_carteira = calcular_var_historico(ret, cart, p["confianca"], p["horizonte"])
        st.session_state["resultados"]["Histórico"] = var

        V_total = cart["Valor da Posição"].sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("💼 Valor da Carteira", f"R$ {V_total:,.0f}")
        col2.metric("📜 VaR Histórico",     f"R$ {var:,.0f}")
        col3.metric("📊 VaR / Carteira",    f"{var/V_total*100:.2f}%")

        st.divider()

        if len(ret_carteira) > 0:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Distribuição de Retornos da Carteira")
                alpha   = 1.0 - p["confianca"]
                var_pct = np.percentile(ret_carteira, alpha * 100)
                mu      = np.mean(ret_carteira)
                sigma   = np.std(ret_carteira)

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=ret_carteira, nbinsx=60,
                    marker_color="#F1F5F9", opacity=0.7, name="Retornos",
                ))

                # Curva normal de referência
                x_range = np.linspace(ret_carteira.min(), ret_carteira.max(), 200)
                y_normal = norm.pdf(x_range, mu, sigma) * len(ret_carteira) * (ret_carteira.max() - ret_carteira.min()) / 60
                fig_hist.add_trace(go.Scatter(
                    x=x_range, y=y_normal,
                    mode="lines", name="Normal Ref.",
                    line=dict(color="#F59E0B", width=2, dash="dash"),
                ))

                fig_hist.add_vline(
                    x=var_pct, line_dash="dash", line_color="#EF4444",
                    annotation_text=f"VaR {p['confianca']*100:.0f}%",
                    annotation_position="top right",
                )
                fig_hist.update_layout(
                    xaxis_title="Retorno Diário", yaxis_title="Frequência",
                    paper_bgcolor="#0B1120", plot_bgcolor="#0B1120",
                    font_color="#E2F8F0", height=380,
                    legend=dict(bgcolor="#0D1F2D"),
                    yaxis=dict(gridcolor="#1E3A4A"),
                    xaxis=dict(gridcolor="#1E3A4A"),
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with col2:
                st.subheader("Ordenação das Perdas (Pior → Melhor)")
                perdas_ordenadas = np.sort(ret_carteira * V_total)
                n = len(perdas_ordenadas)
                idx_var = int(alpha * n)

                fig_sort = go.Figure()
                fig_sort.add_trace(go.Scatter(
                    x=list(range(n)), y=perdas_ordenadas,
                    mode="lines", fill="tozeroy",
                    fillcolor="rgba(45, 212, 191, 0.1)",
                    line=dict(color="#F1F5F9", width=1),
                    name="P&L histórico",
                ))
                fig_sort.add_vline(
                    x=idx_var, line_dash="dash", line_color="#EF4444",
                    annotation_text=f"Percentil {alpha*100:.0f}%",
                )
                fig_sort.update_layout(
                    xaxis_title="Observação (ordenada)",
                    yaxis_title="P&L (R$)",
                    paper_bgcolor="#0B1120", plot_bgcolor="#0B1120",
                    font_color="#E2F8F0", height=380,
                    yaxis=dict(gridcolor="#1E3A4A"),
                    xaxis=dict(gridcolor="#1E3A4A"),
                )
                st.plotly_chart(fig_sort, use_container_width=True)

            # Estatísticas descritivas
            st.subheader("Estatísticas dos Retornos")
            stats_df = pd.DataFrame({
                "Média Diária":    [f"{np.mean(ret_carteira)*100:.4f}%"],
                "Vol. Diária":     [f"{np.std(ret_carteira)*100:.4f}%"],
                "Vol. Anual":      [f"{np.std(ret_carteira)*np.sqrt(252)*100:.2f}%"],
                "Mínimo":          [f"{np.min(ret_carteira)*100:.4f}%"],
                "Máximo":          [f"{np.max(ret_carteira)*100:.4f}%"],
                "Assimetria":      [f"{float(pd.Series(ret_carteira).skew()):.4f}"],
                "Curtose":         [f"{float(pd.Series(ret_carteira).kurtosis()):.4f}"],
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)


# =============================================================================
# PÁGINA 6 — FULL VALUATION
# =============================================================================
with tab_fv:
    st.title("🔄 VaR Full Valuation")
    st.markdown("""
    **Reprecificação completa** da carteira em cada cenário simulado.

    - **Ações**: preço simulado pelo Movimento Browniano Geométrico (GBM)
    - **Opções**: reprecificação completa pelo **Black-Scholes** em cada cenário

    $$S_T = S_0 \\cdot e^{(\\mu - \\frac{1}{2}\\sigma^2)T + \\sigma\\sqrt{T}Z}$$
    """)

    if not garantir_retornos():
        st.info("⬅️ Vá para a aba **📈 Carteira** para montar a carteira primeiro.")
    else:
        p    = st.session_state["params"]
        ret  = st.session_state["retornos"]
        cart = st.session_state["carteira"]

        if st.button("🚀 Executar Full Valuation", type="primary", use_container_width=True):
            with st.spinner(f"Simulando {p['n_sim']:,} cenários..."):
                var, perdas, V_atual = calcular_var_full_valuation(
                    ret, cart, p["confianca"], p["horizonte"], p["n_sim"]
                )
            st.session_state["resultados"]["Full Valuation"] = var
            st.session_state["_perdas_fv"] = perdas
            st.session_state["_V_fv"]      = V_atual
            st.success("✅ Simulação concluída!")

        if "Full Valuation" in st.session_state["resultados"]:
            var    = st.session_state["resultados"]["Full Valuation"]
            perdas = st.session_state.get("_perdas_fv", np.array([]))
            V_atual = st.session_state.get("_V_fv", 1.0)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("💼 Valor da Carteira",   f"R$ {V_atual:,.0f}")
            col2.metric("🔄 VaR Full Valuation",  f"R$ {var:,.0f}")
            col3.metric("📊 VaR / Carteira",       f"{var/V_atual*100:.2f}%")
            col4.metric("🎲 Simulações",           f"{p['n_sim']:,}")

            st.divider()

            if len(perdas) > 0:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Distribuição das Perdas Simuladas")
                    alpha = 1.0 - p["confianca"]
                    fig_mc = go.Figure()
                    fig_mc.add_trace(go.Histogram(
                        x=perdas, nbinsx=80,
                        marker_color="#818CF8", opacity=0.75, name="Perdas simuladas",
                    ))
                    fig_mc.add_vline(
                        x=var, line_dash="dash", line_color="#EF4444",
                        annotation_text=f"VaR {p['confianca']*100:.0f}%",
                        annotation_position="top right",
                    )
                    fig_mc.update_layout(
                        xaxis_title="Perda (R$)", yaxis_title="Frequência",
                        paper_bgcolor="#0B1120", plot_bgcolor="#0B1120",
                        font_color="#E2F8F0", height=380,
                        yaxis=dict(gridcolor="#1E3A4A"),
                        xaxis=dict(gridcolor="#1E3A4A"),
                    )
                    st.plotly_chart(fig_mc, use_container_width=True)

                with col2:
                    st.subheader("Distribuição Acumulada das Perdas")
                    perdas_sorted = np.sort(perdas)
                    cdf = np.arange(1, len(perdas_sorted)+1) / len(perdas_sorted)
                    fig_cdf = go.Figure()
                    fig_cdf.add_trace(go.Scatter(
                        x=perdas_sorted, y=cdf * 100,
                        mode="lines", line=dict(color="#F1F5F9", width=2), name="CDF",
                    ))
                    fig_cdf.add_hline(
                        y=p["confianca"]*100, line_dash="dash", line_color="#EF4444",
                        annotation_text=f"{p['confianca']*100:.0f}%",
                    )
                    fig_cdf.update_layout(
                        xaxis_title="Perda (R$)", yaxis_title="Probabilidade acumulada (%)",
                        paper_bgcolor="#0B1120", plot_bgcolor="#0B1120",
                        font_color="#E2F8F0", height=380,
                        yaxis=dict(gridcolor="#1E3A4A"),
                        xaxis=dict(gridcolor="#1E3A4A"),
                    )
                    st.plotly_chart(fig_cdf, use_container_width=True)

            # Greeks das opções
            opcoes = cart[cart["Tipo"].isin(["Opção Call", "Opção Put"])]
            if not opcoes.empty:
                st.subheader("Greeks das Opções (Black-Scholes)")
                rows_greeks = []
                for _, row in opcoes.iterrows():
                    ativo = row["Ativo"]
                    sigma = ret[ativo].std() * np.sqrt(252) if ativo in ret.columns else 0.30
                    K     = float(row["Strike"])   if pd.notna(row.get("Strike"))     else row["Preço"]*1.05
                    T     = float(row["Vencimento"]) / 252 if pd.notna(row.get("Vencimento")) else 30/252
                    tipo_bs = "Call" if row["Tipo"] == "Opção Call" else "Put"
                    g = black_scholes_greeks(row["Preço"], K, T, 0.105, sigma, tipo_bs)
                    g["Ativo"] = ativo
                    g["Tipo"]  = row["Tipo"]
                    rows_greeks.append(g)
                df_greeks = pd.DataFrame(rows_greeks)[["Ativo","Tipo","Delta","Gamma","Vega","Theta"]]
                st.dataframe(df_greeks, use_container_width=True, hide_index=True)


    # =============================================================================
    # PÁGINA 7 — ANÁLISE DE RISCO
    # =============================================================================
    with tab_analise:
        st.title("📊 Análise de Risco")

        if not garantir_retornos():
            st.info("⬅️ Vá para a aba **📈 Carteira** para montar a carteira primeiro.")
        else:
            p    = st.session_state["params"]
            ret  = st.session_state["retornos"]
            cart = st.session_state["carteira"]
            res  = st.session_state["resultados"]

                # ── Comparação entre metodologias ─────────────────────────────────────
            if res:
                st.subheader("⚖️ Comparação entre Metodologias")
                df_comp = pd.DataFrame([
                    {"Metodologia": k, "VaR (R$)": v} for k, v in res.items()
                ])
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.dataframe(
                        df_comp.style.format({"VaR (R$)": "R$ {:,.0f}"}),
                        use_container_width=True, hide_index=True,
                    )
                with col2:
                    fig_comp = go.Figure(go.Bar(
                        x=df_comp["Metodologia"], y=df_comp["VaR (R$)"],
                        marker_color=["#F1F5F9", "#818CF8", "#F59E0B"][:len(df_comp)],
                        text=[f"R$ {v:,.0f}" for v in df_comp["VaR (R$)"]],
                        textposition="outside",
                    ))
                    fig_comp.update_layout(
                        yaxis_title="VaR (R$)",
                        paper_bgcolor="#0B1120", plot_bgcolor="#0B1120",
                        font_color="#E2F8F0", height=300,
                        yaxis=dict(gridcolor="#1E3A4A"),
                        xaxis=dict(gridcolor="#1E3A4A"),
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.info("ℹ️ Calcule pelo menos uma metodologia para ver a comparação.")

            st.divider()

            # ── VaR Rolante ───────────────────────────────────────────────────────────
            st.subheader("📈 Evolução do VaR ao Longo do Tempo")
            rolling_var = calcular_rolling_var(ret, cart, p["confianca"],
                                               janela_rolling=60, horizonte=p["horizonte"])
            if not rolling_var.empty:
                fig_roll = go.Figure()
                fig_roll.add_trace(go.Scatter(
                    x=rolling_var.index, y=rolling_var.values,
                    mode="lines", fill="tozeroy",
                    fillcolor="rgba(45, 212, 191, 0.08)",
                    line=dict(color="#F1F5F9", width=2),
                    name="VaR Rolante (60 dias)",
                ))
                fig_roll.update_layout(
                    xaxis_title="Data", yaxis_title="VaR (R$)",
                    paper_bgcolor="#0B1120", plot_bgcolor="#0B1120",
                    font_color="#E2F8F0", height=320,
                    yaxis=dict(gridcolor="#1E3A4A"),
                    xaxis=dict(gridcolor="#1E3A4A"),
                )
                st.plotly_chart(fig_roll, use_container_width=True)
            else:
                st.info("Dados insuficientes para VaR rolante (necessário > 60 obs.).")

            st.divider()

            # ── Stress Testing ────────────────────────────────────────────────────────
            st.subheader("💥 Stress Testing")
            st.markdown("Aplica choques históricos extremos na carteira e calcula o impacto financeiro.")

            df_stress = stress_test(ret, cart)
            V_total   = cart["Valor da Posição"].sum()

            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(
                    df_stress.style.format({
                        "P&L (R$)":    "R$ {:,.0f}",
                        "% Carteira":  "{:.1f}%",
                    }).map(
                        lambda v: "color: #EF4444" if isinstance(v, (int, float)) and v < 0
                        else "color: #22C55E",
                    ),
                    use_container_width=True,
                )
            with col2:
                cores_stress = ["#EF4444" if v < 0 else "#22C55E"
                                for v in df_stress["P&L (R$)"]]
                fig_stress = go.Figure(go.Bar(
                    x=df_stress.index, y=df_stress["P&L (R$)"],
                    marker_color=cores_stress,
                    text=[f"R$ {v:,.0f}" for v in df_stress["P&L (R$)"]],
                    textposition="outside",
                ))
                fig_stress.add_hline(y=0, line_color="#475569", line_width=1)
                fig_stress.update_layout(
                    yaxis_title="P&L (R$)",
                    paper_bgcolor="#0B1120", plot_bgcolor="#0B1120",
                    font_color="#E2F8F0", height=350,
                    yaxis=dict(gridcolor="#1E3A4A"),
                    xaxis=dict(gridcolor="#1E3A4A", tickangle=-20),
                )
                st.plotly_chart(fig_stress, use_container_width=True)

            st.divider()

            # ── Sensibilidade do VaR ──────────────────────────────────────────────────
            st.subheader("🎚️ Sensibilidade do VaR ao Nível de Confiança")
            _, ret_carteira = calcular_var_historico(ret, cart, 0.95, p["horizonte"])

            if len(ret_carteira) > 0:
                niveis = np.arange(0.90, 0.9999, 0.005)
                V_total = cart["Valor da Posição"].sum()
                vars_sens = [
                    -np.percentile(ret_carteira, (1-nc)*100) * V_total * np.sqrt(p["horizonte"])
                    for nc in niveis
                ]
                fig_sens = go.Figure(go.Scatter(
                    x=niveis*100, y=vars_sens,
                    mode="lines", line=dict(color="#F59E0B", width=2),
                    fill="tozeroy", fillcolor="rgba(245, 158, 11, 0.08)",
                ))
                fig_sens.update_layout(
                    xaxis_title="Nível de Confiança (%)",
                    yaxis_title="VaR (R$)",
                    paper_bgcolor="#0B1120", plot_bgcolor="#0B1120",
                    font_color="#E2F8F0", height=320,
                    yaxis=dict(gridcolor="#1E3A4A"),
                    xaxis=dict(gridcolor="#1E3A4A"),
                )
                st.plotly_chart(fig_sens, use_container_width=True)

            st.divider()

            # ── Evolução do portfólio ─────────────────────────────────────────────────
            st.subheader("📉 Evolução Histórica dos Preços")
            tickers_acoes = cart[cart["Tipo"] == "Ação"]["Ativo"].tolist()
            tickers_disp  = [t for t in tickers_acoes if t in ret.columns]

            if tickers_disp:
                ativo_sel = st.selectbox("Selecione o ativo:", tickers_disp)
                with st.spinner(f"Carregando {ativo_sel}..."):
                    try:
                        dados_hist = yf.download(ativo_sel, period="1y",
                                                 progress=False, auto_adjust=True)
                        fig_preco = go.Figure(go.Scatter(
                            x=dados_hist.index,
                            y=dados_hist["Close"].values.flatten(),
                            mode="lines", line=dict(color="#F1F5F9", width=2),
                            fill="tozeroy", fillcolor="rgba(45,212,191,0.07)",
                        ))
                        fig_preco.update_layout(
                            title=f"Preço — {ativo_sel}",
                            xaxis_title="Data", yaxis_title="Preço (R$)",
                            paper_bgcolor="#0B1120", plot_bgcolor="#0B1120",
                            font_color="#E2F8F0", height=320,
                            yaxis=dict(gridcolor="#1E3A4A"),
                            xaxis=dict(gridcolor="#1E3A4A"),
                        )
                        st.plotly_chart(fig_preco, use_container_width=True)
                    except Exception:
                        st.info("Dados de preço não disponíveis.")

            # ── Limitações do VaR ─────────────────────────────────────────────────────
            st.divider()
            st.subheader("⚠️ Limitações do VaR")
            col1, col2 = st.columns(2)
            with col1:
                st.error("**Não captura eventos extremos**\nO VaR mede perdas em condições normais. Em crises severas (2008, COVID), as perdas reais costumam superar o VaR estimado.")
                st.error("**Hipótese de normalidade**\nRetornos financeiros têm caudas mais pesadas do que a normal prevê — fenômeno chamado de *fat tails*.")
            with col2:
                st.warning("**Dependência da janela histórica**\nUma janela que não inclui períodos de crise subestima sistematicamente o risco.")
                st.warning("**Não é subadditivo**\nO VaR pode indicar que combinar duas carteiras é mais arriscado do que tê-las separadas — o que viola intuição financeira básica.")
