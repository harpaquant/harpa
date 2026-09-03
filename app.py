import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import sqlite3
import math
import io
from pathlib import Path
from datetime import datetime, timedelta, date
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

PASTA = Path(__file__).parent

LOGO_BARRA = next((PASTA / n for n in ("harpa-clara.png", "harpa.png", "harpa.jpg")
                   if (PASTA / n).exists()), None)
LOGO_ABA = next((PASTA / n for n in ("harpa.png", "harpa-clara.png", "harpa.jpg")
                 if (PASTA / n).exists()), None)

st.set_page_config(page_title="Harpa Quant", layout="wide",
                   page_icon=str(LOGO_ABA) if LOGO_ABA else None)

DB_COTACOES = PASTA / "zcotacoes.db"
DB_PCR = PASTA / "zpcr_historico.db"

INICIO_ACOES = "2015-01-01"
INICIO_INDICE = "2000-01-01"
MARGEM_REVISAO = 15
TOLERANCIA_AJUSTE = 0.005
MINUTOS_ENTRE_COLETAS = 30

ativos_ibov = ['ABEV3', 'ALOS3', 'ASAI3', 'AURE3', 'AXIA3', 'AXIA6', 'AXIA7', 'AZZA3',
               'B3SA3', 'BBAS3', 'BBDC3', 'BBDC4', 'BBSE3', 'BEEF3', 'BPAC11', 'BRAP4',
               'BRAV3', 'BRKM5', 'CEAB3', 'CMIG4', 'CMIN3', 'COGN3', 'CPFE3', 'CPLE3',
               'CSAN3', 'CSMG3', 'CSNA3', 'CURY3', 'CXSE3', 'CYRE3', 'CYRE4', 'DIRR3',
               'EGIE3', 'EMBJ3', 'ENEV3', 'ENGI11', 'EQTL3', 'FLRY3', 'GGBR4', 'GOAU4',
               'HAPV3', 'HYPE3', 'IGTI11', 'IRBR3', 'ISAE4', 'ITSA4', 'ITUB4', 'KLBN11',
               'LREN3', 'MBRF3', 'MGLU3', 'MOTV3', 'MRVE3', 'MULT3', 'NATU3', 'PCAR3',
               'PETR3', 'PETR4', 'POMO4', 'PRIO3', 'PSSA3', 'RADL3', 'RAIL3', 'RAIZ4',
               'RDOR3', 'RECV3', 'RENT3', 'RENT4', 'SANB11', 'SBSP3', 'SLCE3', 'SMFT3',
               'SUZB3', 'TAEE11', 'TIMS3', 'TOTS3', 'UGPA3', 'USIM5', 'VALE3', 'VAMO3',
               'VBBR3', 'VIVA3', 'VIVT3', 'WEGE3', 'YDUQ3']

ativos_yf = [t + '.SA' for t in ativos_ibov]
ibov = '^BVSP'
universo_padrao = tuple(ativos_yf + [ibov])


# ============================================================================
# BASE LOCAL DE COTACOES
# ============================================================================

def abrir_base():
    con = sqlite3.connect(DB_COTACOES)
    con.execute('''CREATE TABLE IF NOT EXISTS cotacoes (
                       data TEXT NOT NULL,
                       ticker TEXT NOT NULL,
                       abertura REAL, maxima REAL, minima REAL,
                       fechamento REAL, volume REAL,
                       PRIMARY KEY (data, ticker))''')
    con.execute('CREATE INDEX IF NOT EXISTS idx_cotacoes_ticker ON cotacoes (ticker, data)')
    con.execute('''CREATE TABLE IF NOT EXISTS controle (
                       ticker TEXT PRIMARY KEY,
                       primeiro_dado TEXT,
                       ultimo_dado TEXT,
                       ultima_coleta TEXT,
                       linhas INTEGER)''')
    return con


def baixar_do_yahoo(tickers, inicio, fim):
    bruto = yf.download(list(tickers), start=inicio, end=fim, interval='1d',
                        auto_adjust=True, progress=False, group_by='column')
    if bruto is None or bruto.empty:
        return pd.DataFrame()
    if not isinstance(bruto.columns, pd.MultiIndex):
        bruto.columns = pd.MultiIndex.from_product([bruto.columns, [list(tickers)[0]]])
    campos = {'Open': 'abertura', 'High': 'maxima', 'Low': 'minima',
              'Close': 'fechamento', 'Volume': 'volume'}
    presentes = [c for c in bruto.columns.get_level_values(1).unique() if c in tickers]
    pedacos = []
    for ticker in presentes:
        try:
            fatia = bruto.xs(ticker, axis=1, level=1).copy()
        except KeyError:
            continue
        fatia = fatia.rename(columns=campos)
        colunas_uteis = [c for c in campos.values() if c in fatia.columns]
        fatia = fatia[colunas_uteis].dropna(how='all')
        if 'fechamento' in fatia.columns:
            fatia = fatia[fatia['fechamento'].notna()]
        if fatia.empty:
            continue
        fatia['ticker'] = ticker
        fatia['data'] = fatia.index.strftime('%Y-%m-%d')
        pedacos.append(fatia)
    if not pedacos:
        return pd.DataFrame()
    return pd.concat(pedacos, ignore_index=True)


def gravar_cotacoes(con, tabela):
    if tabela.empty:
        return 0
    colunas = ['data', 'ticker', 'abertura', 'maxima', 'minima', 'fechamento', 'volume']
    for coluna in colunas:
        if coluna not in tabela.columns:
            tabela[coluna] = None
    linhas = list(tabela[colunas].itertuples(index=False, name=None))
    linhas = [tuple(None if (not isinstance(v, str) and pd.isna(v)) else v for v in linha)
              for linha in linhas]
    con.executemany('''INSERT INTO cotacoes (data, ticker, abertura, maxima, minima, fechamento, volume)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(data, ticker) DO UPDATE SET
                           abertura = excluded.abertura,
                           maxima = excluded.maxima,
                           minima = excluded.minima,
                           fechamento = excluded.fechamento,
                           volume = excluded.volume''', linhas)
    con.commit()
    return len(linhas)


def atualizar_controle(con, tickers):
    for ticker in tickers:
        primeiro, ultimo, linhas = con.execute(
            'SELECT MIN(data), MAX(data), COUNT(*) FROM cotacoes WHERE ticker = ?', (ticker,)).fetchone()
        con.execute('''INSERT INTO controle (ticker, primeiro_dado, ultimo_dado, ultima_coleta, linhas)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(ticker) DO UPDATE SET
                           primeiro_dado = excluded.primeiro_dado,
                           ultimo_dado = excluded.ultimo_dado,
                           ultima_coleta = excluded.ultima_coleta,
                           linhas = excluded.linhas''',
                    (ticker, primeiro, ultimo, datetime.now().isoformat(timespec='seconds'), linhas))
    con.commit()


@st.cache_data(ttl=900, show_spinner="Atualizando a base local de cotacoes...")
def atualizar_base(tickers_tupla, forcar_completo, chave_do_dia):
    con = abrir_base()
    con.execute('DELETE FROM cotacoes WHERE fechamento IS NULL')
    con.commit()
    tickers = list(tickers_tupla)
    limite_coleta = (datetime.now() - timedelta(minutes=MINUTOS_ENTRE_COLETAS)).isoformat(timespec='seconds')
    amanha = (date.today() + timedelta(days=1)).strftime('%Y-%m-%d')

    controle = pd.read_sql_query('SELECT * FROM controle', con)
    controle = controle.set_index('ticker') if not controle.empty else pd.DataFrame()

    completos = []
    incrementais = {}

    for ticker in tickers:
        if forcar_completo:
            completos.append(ticker)
            continue
        if controle.empty or ticker not in controle.index:
            completos.append(ticker)
            continue
        registro = controle.loc[ticker]
        recente = bool(registro['ultima_coleta']) and registro['ultima_coleta'] > limite_coleta
        if recente:
            continue
        if pd.isna(registro['ultimo_dado']) or not registro['linhas']:
            completos.append(ticker)
            continue
        incrementais[ticker] = registro['ultimo_dado']

    resumo = {'completos': 0, 'incrementais': 0, 'reajustados': 0, 'linhas_gravadas': 0}

    if forcar_completo and completos:
        con.executemany('DELETE FROM cotacoes WHERE ticker = ?', [(t,) for t in completos])
        con.commit()

    if completos:
        acoes = [t for t in completos if not t.startswith('^')]
        indices = [t for t in completos if t.startswith('^')]
        for grupo, inicio in ((acoes, INICIO_ACOES), (indices, INICIO_INDICE)):
            if not grupo:
                continue
            resumo['linhas_gravadas'] += gravar_cotacoes(con, baixar_do_yahoo(grupo, inicio, amanha))
        atualizar_controle(con, completos)
        resumo['completos'] = len(completos)

    if incrementais:
        inicio_lote = (pd.to_datetime(min(incrementais.values())) -
                       timedelta(days=MARGEM_REVISAO)).strftime('%Y-%m-%d')
        baixado = baixar_do_yahoo(list(incrementais.keys()), inicio_lote, amanha)

        refazer = []
        if not baixado.empty:
            marcadores = ','.join('?' * len(incrementais))
            antigo = pd.read_sql_query(
                f'SELECT data, ticker, fechamento FROM cotacoes '
                f'WHERE ticker IN ({marcadores}) AND data >= ?',
                con, params=list(incrementais.keys()) + [inicio_lote])
            if not antigo.empty:
                comparacao = baixado[['data', 'ticker', 'fechamento']].merge(
                    antigo, on=['data', 'ticker'], suffixes=('_novo', '_antigo')).dropna()
                comparacao = comparacao[comparacao['fechamento_antigo'].abs() > 1e-9]
                if not comparacao.empty:
                    comparacao['desvio'] = (comparacao['fechamento_novo'] /
                                            comparacao['fechamento_antigo'] - 1).abs()
                    piores = comparacao.groupby('ticker')['desvio'].max()
                    refazer = piores[piores > TOLERANCIA_AJUSTE].index.tolist()

        if not baixado.empty:
            resumo['linhas_gravadas'] += gravar_cotacoes(
                con, baixado[~baixado['ticker'].isin(refazer)])

        if refazer:
            con.executemany('DELETE FROM cotacoes WHERE ticker = ?', [(t,) for t in refazer])
            con.commit()
            acoes = [t for t in refazer if not t.startswith('^')]
            indices = [t for t in refazer if t.startswith('^')]
            for grupo, inicio in ((acoes, INICIO_ACOES), (indices, INICIO_INDICE)):
                if not grupo:
                    continue
                resumo['linhas_gravadas'] += gravar_cotacoes(con, baixar_do_yahoo(grupo, inicio, amanha))
            resumo['reajustados'] = len(refazer)

        atualizar_controle(con, list(incrementais.keys()))
        resumo['incrementais'] = len(incrementais)

    ultima_data = con.execute('SELECT MAX(data) FROM cotacoes').fetchone()[0]
    total_linhas = con.execute('SELECT COUNT(*) FROM cotacoes').fetchone()[0]
    con.close()

    resumo['ultima_data'] = ultima_data
    resumo['total_linhas'] = total_linhas
    resumo['marca'] = f"{ultima_data}|{total_linhas}"
    return resumo


@st.cache_data(ttl=900, show_spinner=False)
def ler_painel(tickers_tupla, inicio, fim, marca):
    con = sqlite3.connect(DB_COTACOES)
    marcadores = ','.join('?' * len(tickers_tupla))
    tabela = pd.read_sql_query(
        f'SELECT data, ticker, fechamento, volume FROM cotacoes '
        f'WHERE ticker IN ({marcadores}) AND data >= ? AND data <= ? ORDER BY data',
        con, params=list(tickers_tupla) + [inicio, fim])
    con.close()
    if tabela.empty:
        return pd.DataFrame(), pd.DataFrame()
    precos = tabela.pivot(index='data', columns='ticker', values='fechamento')
    volumes = tabela.pivot(index='data', columns='ticker', values='volume')
    precos.index = pd.to_datetime(precos.index)
    volumes.index = pd.to_datetime(volumes.index)
    return precos, volumes


@st.cache_data(ttl=900, show_spinner=False)
def ler_ativo(ticker, inicio, fim, marca):
    con = sqlite3.connect(DB_COTACOES)
    tabela = pd.read_sql_query(
        'SELECT data, abertura, maxima, minima, fechamento, volume FROM cotacoes '
        'WHERE ticker = ? AND data >= ? AND data <= ? ORDER BY data',
        con, params=(ticker, inicio, fim))
    con.close()
    if tabela.empty:
        return pd.DataFrame()
    tabela['data'] = pd.to_datetime(tabela['data'])
    tabela = tabela.set_index('data')
    tabela.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return tabela


@st.cache_data(ttl=300, show_spinner=False)
def ler_controle(marca):
    con = sqlite3.connect(DB_COTACOES)
    tabela = pd.read_sql_query('SELECT * FROM controle ORDER BY ticker', con)
    con.close()
    return tabela


# ============================================================================
# FUNCOES DE APOIO
# ============================================================================

def ornstein_uhlenbeck(params, serie):
    theta, mu, sigma = params
    dt = 1.0 / 252
    diferenca = np.diff(serie)
    residuo = diferenca - theta * (mu - serie[:-1]) * dt
    return np.sum(residuo ** 2)


@st.cache_data(ttl=21600, show_spinner=False)
def rodar_pairs(quotes, corte_nivel, corte_residuo, corte_pbeta):
    pvalor_nivel = {}
    for coluna in quotes.columns:
        pvalor_nivel[coluna] = adfuller(quotes[coluna].values, autolag='AIC')[1]

    elegiveis = [c for c in quotes.columns if pvalor_nivel[c] > corte_nivel]
    total_pares = int(math.comb(len(quotes.columns), 2))
    cointegrados = 0
    registros = []

    for i in range(len(elegiveis)):
        acao1 = elegiveis[i]
        y = quotes[acao1]
        for j in range(i + 1, len(elegiveis)):
            acao2 = elegiveis[j]
            x = quotes[acao2]

            modelo = OLS(y, sm.add_constant(x)).fit()
            pbeta = modelo.pvalues.iloc[1]
            if pbeta >= corte_pbeta:
                continue

            residuo = modelo.resid
            pvalor_residuo = adfuller(residuo.values, autolag='AIC')[1]
            if pvalor_residuo > corte_residuo:
                continue
            cointegrados += 1

            alfa = modelo.params.iloc[0]
            beta = modelo.params.iloc[1]
            preco1 = y.iloc[-1]
            preco2 = x.iloc[-1]
            desvio_padrao = np.std(residuo)
            desvio_absoluto = preco1 - alfa - beta * preco2

            if abs(desvio_absoluto) <= desvio_padrao:
                continue
            maximo = residuo.max()
            minimo = residuo.min()
            if desvio_absoluto >= maximo or desvio_absoluto <= minimo:
                continue

            chute = [0.1, np.mean(residuo.values), np.std(residuo.values)]
            solucao = minimize(ornstein_uhlenbeck, chute, args=(residuo.values,))
            meia_vida = np.log(2) / abs(solucao.x[0])

            n = len(y)
            u = np.argsort(np.argsort(y.values)) / (n - 1)
            v = np.argsort(np.argsort(x.values)) / (n - 1)
            denominador_baixo = np.sum(u <= 0.10)
            denominador_alto = np.sum(u >= 0.90)
            tau_l = np.sum((u <= 0.10) & (v <= 0.10)) / denominador_baixo if denominador_baixo > 0 else 0
            tau_u = np.sum((u >= 0.90) & (v >= 0.90)) / denominador_alto if denominador_alto > 0 else 0

            registros.append({
                'Acao1': acao1, 'Acao2': acao2,
                'PP1': pvalor_nivel[acao1], 'PP2': pvalor_nivel[acao2],
                'PPR': pvalor_residuo, 'Alfa': alfa, 'Beta': beta, 'pBeta': pbeta,
                'Preco1': preco1, 'Preco2': preco2,
                'Max': maximo, 'Min': minimo,
                'DesvioP': desvio_padrao, 'DesvioAb': desvio_absoluto,
                'OU': meia_vida,
                'TailDep_L': tau_l, 'TailDep_U': tau_u, 'TailDep_Avg': (tau_l + tau_u) / 2})

    resultado = pd.DataFrame(registros)
    diagnostico = {'total_pares': total_pares, 'elegiveis': len(elegiveis),
                   'cointegrados': cointegrados, 'validos': len(resultado)}
    return resultado, diagnostico


@st.cache_data(ttl=86400, show_spinner="Estimando os regimes de volatilidade...")
def rodar_markov(precos_indice, janela, n_estados):
    from hmmlearn import hmm
    retornos = np.log(precos_indice / precos_indice.shift(1))
    desvio = retornos.rolling(window=janela).std().dropna()
    modelo = hmm.GaussianHMM(n_components=n_estados, covariance_type="full",
                             n_iter=1000, random_state=42)
    modelo.fit(desvio.values.reshape(-1, 1))
    estados = modelo.predict(desvio.values.reshape(-1, 1))
    medias = modelo.means_.flatten()
    ordem = np.argsort(medias)
    mapa = {original: posicao for posicao, original in enumerate(ordem)}
    painel = pd.DataFrame({'volatilidade': desvio.values,
                           'preco': precos_indice.loc[desvio.index].values,
                           'regime': np.array([mapa[e] for e in estados])}, index=desvio.index)
    return painel, modelo.transmat_[np.ix_(ordem, ordem)], np.sort(medias)


@st.cache_data(ttl=900, show_spinner="Consultando a cadeia de opcoes...")
def baixar_chain(subjacente, vencimento):
    url = (f'https://opcoes.net.br/listaopcoes/completa'
           f'?idAcao={subjacente}&listarVencimentos=false'
           f'&cotacoes=true&vencimentos={vencimento}')
    r = requests.get(url, timeout=30).json()
    linhas = [[subjacente, vencimento, i[0].split('_')[0], i[2], i[3], i[5], i[8], i[9], i[10]]
              for i in r['data']['cotacoesOpcoes']]
    return pd.DataFrame(linhas, columns=['subjacente', 'vencimento', 'ativo', 'tipo',
                                         'modelo', 'strike', 'preco', 'negocios', 'volume'])


@st.cache_data(ttl=3600, show_spinner=False)
def baixar_vencimentos(subjacente):
    url = (f'https://opcoes.net.br/listaopcoes/completa'
           f'?idAcao={subjacente}&listarVencimentos=true&cotacoes=true')
    r = requests.get(url, timeout=30).json()
    return [v['value'] for v in r['data']['vencimentos']]


@st.cache_data(ttl=3600, show_spinner=False)
def buscar_info_ticker(ticker):
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}


def formatar_tabela_pdf(pdf, x, y, titulo, tabela, coluna):
    pdf.set_xy(x, y)
    pdf.set_font("Helvetica", 'B', 9)
    pdf.cell(80, 6, titulo, ln=False)
    pdf.ln()
    pdf.set_xy(x, pdf.get_y())
    pdf.set_font("Helvetica", 'B', 8)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(40, 6, "Ativo", border=1, fill=True)
    pdf.cell(40, 6, coluna.replace('_', ' '), border=1, fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", '', 8)
    for _, linha in tabela.iterrows():
        pdf.set_x(x)
        pdf.cell(40, 5, str(linha['Ativo']), border=1)
        pdf.cell(40, 5, f"{linha[coluna] * 100:.2f}%", border=1)
        pdf.ln()


# ============================================================================
# BARRA LATERAL E ATUALIZACAO DA BASE
# ============================================================================

if LOGO_BARRA:
    st.sidebar.image(str(LOGO_BARRA), width=120)

st.sidebar.header("Menu")

funcao = st.sidebar.radio("Funcao", [
    "Painel do Dia",
    "Market Movers",
    "Alerta de Volume",
    "Pairs Trading",
    "Put-Call Ratio",
    "Regime de Volatilidade",
    "Informacoes do Ativo",
    "Estatisticas Descritivas",
    "Analise de Retornos",
    "Metricas de Risco",
    "Correlacao entre Ativos",
    "Download de Dados",
    "Base de Dados"
])

st.sidebar.divider()

funcoes_ativo_unico = ["Informacoes do Ativo", "Estatisticas Descritivas",
                       "Analise de Retornos", "Metricas de Risco",
                       "Correlacao entre Ativos", "Download de Dados"]

chave_do_dia = date.today().isoformat()

universo = list(universo_padrao)
ticker = None
ticker_display = None
data_inicio = None
data_fim = None
lista_tickers = []

if funcao in funcoes_ativo_unico:
    st.sidebar.subheader("Configuracoes de Dados")
    ticker_input = st.sidebar.text_input("Ticker Principal", value="PETR4")
    ticker = ticker_input.upper() if ticker_input.upper().endswith(".SA") else f"{ticker_input.upper()}.SA"
    ticker_display = ticker.replace(".SA", "")
    data_inicio = st.sidebar.date_input("Data Inicio", value=datetime.now() - timedelta(days=365),
                                        format="DD/MM/YYYY")
    data_fim = st.sidebar.date_input("Data Fim", value=datetime.now(), format="DD/MM/YYYY")
    universo.append(ticker)

if funcao == "Correlacao entre Ativos":
    st.sidebar.subheader("Ativos para Comparacao")
    tickers_comparacao = st.sidebar.text_area("Tickers (um por linha)", value="VALE3\nITUB4\nBBDC4\nABEV3")
    lista_tickers = [t.strip() if t.strip().endswith(".SA") else f"{t.strip()}.SA"
                     for t in tickers_comparacao.split('\n') if t.strip()]
    lista_tickers = [ticker] + lista_tickers
    universo += lista_tickers

universo = tuple(dict.fromkeys(universo))

if not DB_COTACOES.exists():
    st.info("Primeira execucao: a base local de cotacoes sera montada agora. "
            "Isso leva alguns minutos e acontece uma unica vez. "
            "Nas proximas aberturas so os pregoes novos sao baixados.")

resumo_base = atualizar_base(universo, False, chave_do_dia)
marca = resumo_base['marca']

st.sidebar.divider()
st.sidebar.caption(f"Base ate {resumo_base['ultima_data']} com "
                   + f"{resumo_base['total_linhas']:,}".replace(',', '.') + " observacoes")
if st.sidebar.button("Buscar pregoes novos"):
    st.cache_data.clear()
    st.rerun()


# ============================================================================
# PAINEL DO DIA
# ============================================================================

if funcao == "Painel do Dia":

    st.title("Painel do Dia")
    st.caption(f"Rodada de {datetime.now().strftime('%d/%m/%Y %H:%M')}")

    inicio_leitura = (date.today() - timedelta(days=400)).isoformat()
    precos, volumes = ler_painel(universo_padrao, inicio_leitura, chave_do_dia, marca)

    if precos.empty:
        st.error("A base local esta vazia. Use a pagina Base de Dados para reconstruir.")
        st.stop()

    ultimo_dia = precos.index[-1]
    disponiveis = [a for a in ativos_yf if a in precos.columns]

    resumo = []
    for ativo in disponiveis:
        serie = precos[ativo].dropna()
        if len(serie) < 22:
            continue
        resumo.append({'Ativo': ativo.replace('.SA', ''),
                       'Retorno 1d (%)': (serie.iloc[-1] / serie.iloc[-2] - 1) * 100,
                       'Retorno 5d (%)': (serie.iloc[-1] / serie.iloc[-6] - 1) * 100,
                       'Retorno 21d (%)': (serie.iloc[-1] / serie.iloc[-22] - 1) * 100})
    resumo = pd.DataFrame(resumo)

    serie_ibov = precos[ibov].dropna()
    variacao_ibov = (serie_ibov.iloc[-1] / serie_ibov.iloc[-2] - 1) * 100
    variacao_ibov_5d = (serie_ibov.iloc[-1] / serie_ibov.iloc[-6] - 1) * 100
    retorno_ibov_21 = (serie_ibov.iloc[-1] / serie_ibov.iloc[-22] - 1) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ibovespa", f"{serie_ibov.iloc[-1]:,.0f}".replace(',', '.'), f"{variacao_ibov:.2f}%")
    col2.metric("Ibovespa 5 dias", f"{variacao_ibov_5d:.2f}%")
    col3.metric("Altas no dia", f"{int((resumo['Retorno 1d (%)'] > 0).sum())} de {len(resumo)}")
    col4.metric("Ultimo pregao", ultimo_dia.strftime('%d/%m/%Y'))

    st.divider()

    st.subheader("Extremos do dia")
    formato_painel = {'Retorno 1d (%)': '{:.2f}', 'Retorno 5d (%)': '{:.2f}', 'Retorno 21d (%)': '{:.2f}'}
    col_alta, col_baixa = st.columns(2)
    with col_alta:
        st.write("Maiores altas")
        st.dataframe(resumo.sort_values('Retorno 1d (%)', ascending=False).head(5).style.format(formato_painel),
                     use_container_width=True, hide_index=True)
    with col_baixa:
        st.write("Maiores baixas")
        st.dataframe(resumo.sort_values('Retorno 1d (%)', ascending=True).head(5).style.format(formato_painel),
                     use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Alertas de volume")
    volume_medio = volumes.shift(1).rolling(window=3).mean()
    volume_desvio = volumes.shift(1).rolling(window=3).std()

    alertas = []
    for ativo in disponiveis:
        try:
            volume_hoje = volumes.loc[ultimo_dia, ativo]
            media = volume_medio.loc[ultimo_dia, ativo]
            desvio = volume_desvio.loc[ultimo_dia, ativo]
            if pd.isna(volume_hoje) or pd.isna(media) or pd.isna(desvio) or media == 0:
                continue
            if volume_hoje <= media + 2 * desvio:
                continue
            serie = precos[ativo]
            retorno = (serie.loc[ultimo_dia] / serie.iloc[-22] - 1) * 100
            alertas.append({'Ativo': ativo.replace('.SA', ''),
                            'Retorno 21d (%)': round(retorno, 2),
                            'Volume medio': int(media),
                            'Volume do dia': int(volume_hoje),
                            'Excesso (%)': round((volume_hoje / media - 1) * 100, 2),
                            'Movimento': 'alta' if retorno > 0 else 'baixa',
                            'Contra o IBOV': 'melhor' if retorno > retorno_ibov_21 else 'pior'})
        except Exception:
            continue

    if alertas:
        st.dataframe(pd.DataFrame(alertas).sort_values('Excesso (%)', ascending=False),
                     use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum ativo rompeu a media mais dois desvios de volume no ultimo pregao.")

    st.divider()

    col_pcr, col_regime = st.columns(2)

    with col_pcr:
        st.subheader("Put-Call Ratio")
        try:
            con = sqlite3.connect(DB_PCR)
            historico_rapido = pd.read_sql_query(
                "SELECT data, subjacente, pcr_negocios, pcr_volume FROM pcr ORDER BY data DESC LIMIT 2", con)
            con.close()
            if len(historico_rapido) >= 1:
                atual = historico_rapido.iloc[0]
                variacao_n = (atual['pcr_negocios'] - historico_rapido.iloc[1]['pcr_negocios']) \
                    if len(historico_rapido) > 1 else None
                variacao_v = (atual['pcr_volume'] - historico_rapido.iloc[1]['pcr_volume']) \
                    if len(historico_rapido) > 1 else None
                st.metric(f"PCR negocios ({atual['subjacente']})", f"{atual['pcr_negocios']:.4f}",
                          f"{variacao_n:+.4f}" if variacao_n is not None else None)
                st.metric("PCR volume", f"{atual['pcr_volume']:.4f}",
                          f"{variacao_v:+.4f}" if variacao_v is not None else None)
                st.caption(f"Ultimo registro gravado em {atual['data']}. "
                           "Abra a pagina Put-Call Ratio para gravar o dia de hoje.")
            else:
                st.info("Sem historico gravado. Abra a pagina Put-Call Ratio.")
        except Exception as erro:
            st.warning(f"Historico de PCR indisponivel: {erro}")

    with col_regime:
        st.subheader("Regime de volatilidade")
        try:
            indice = ler_ativo(ibov, INICIO_INDICE, chave_do_dia, marca)
            painel_regime, transicao, medias_regime = rodar_markov(indice['Close'], 30, 2)
            regime_atual = int(painel_regime['regime'].iloc[-1])
            st.metric("Estado atual", "Alta volatilidade" if regime_atual == 1 else "Baixa volatilidade")
            st.metric("Volatilidade movel 30 dias", f"{painel_regime['volatilidade'].iloc[-1] * 100:.2f}%")
            st.caption("Probabilidade de permanecer no mesmo estado amanha: "
                       f"{transicao[regime_atual, regime_atual] * 100:.1f}%")
        except ImportError:
            st.warning("Instale hmmlearn para habilitar a deteccao de regimes.")
        except Exception as erro:
            st.warning(f"Nao foi possivel estimar o regime: {erro}")

    st.divider()
    st.caption("O rastreamento de pares cointegrados fica na pagina Pairs Trading, "
               "que exige alguns segundos de processamento na primeira execucao do dia.")


# ============================================================================
# MARKET MOVERS
# ============================================================================

elif funcao == "Market Movers":

    st.title("Market Movers")

    st.sidebar.subheader("Parametros")
    janela_movers = st.sidebar.slider("Janela em pregoes", 3, 21, 5)
    quantos = st.sidebar.slider("Quantos ativos por lista", 3, 15, 5)

    inicio_leitura = (date.today() - timedelta(days=120)).isoformat()
    precos, volumes = ler_painel(universo_padrao, inicio_leitura, chave_do_dia, marca)
    ultimo_dia = precos.index[-1]

    informacoes = []
    for ativo in ativos_yf:
        if ativo not in precos.columns:
            continue
        serie = precos[ativo].dropna()
        volume = volumes[ativo].dropna()
        if len(serie) < janela_movers + 1 or len(volume) < janela_movers + 1:
            continue
        serie = serie.tail(janela_movers + 1)
        volume = volume.tail(janela_movers + 1)
        retorno = (serie.iloc[-1] / serie.iloc[0]) - 1
        variacao_volume = (volume.iloc[-1] - volume.iloc[0]) / volume.iloc[0] if volume.iloc[0] != 0 else np.nan
        volatilidade = np.log(serie / serie.shift(1)).dropna().std() * np.sqrt(252)
        informacoes.append({'Ativo': ativo.replace('.SA', ''), 'Retorno': retorno,
                            'VolumeVar': variacao_volume, 'Volatilidade': volatilidade})

    df_info = pd.DataFrame(informacoes).dropna()

    st.caption(f"Fechamento de {ultimo_dia.strftime('%d/%m/%Y')} contra {janela_movers} pregoes atras. "
               f"{len(df_info)} ativos com dados completos.")

    ret_maiores = df_info.sort_values('Retorno', ascending=False).head(quantos)
    ret_piores = df_info.sort_values('Retorno', ascending=True).head(quantos)
    vol_maiores = df_info.sort_values('VolumeVar', ascending=False).head(quantos)
    vol_piores = df_info.sort_values('VolumeVar', ascending=True).head(quantos)
    vola_maiores = df_info.sort_values('Volatilidade', ascending=False).head(quantos)
    vola_menores = df_info.sort_values('Volatilidade', ascending=True).head(quantos)

    formato = {'Retorno': '{:.2%}', 'VolumeVar': '{:.2%}', 'Volatilidade': '{:.2%}'}

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Maiores retornos")
        st.dataframe(ret_maiores[['Ativo', 'Retorno']].style.format(formato),
                     use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Piores retornos")
        st.dataframe(ret_piores[['Ativo', 'Retorno']].style.format(formato),
                     use_container_width=True, hide_index=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Maior variacao de volume")
        st.dataframe(vol_maiores[['Ativo', 'VolumeVar']].style.format(formato),
                     use_container_width=True, hide_index=True)
    with col4:
        st.subheader("Menor variacao de volume")
        st.dataframe(vol_piores[['Ativo', 'VolumeVar']].style.format(formato),
                     use_container_width=True, hide_index=True)

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Maior volatilidade")
        st.dataframe(vola_maiores[['Ativo', 'Volatilidade']].style.format(formato),
                     use_container_width=True, hide_index=True)
    with col6:
        st.subheader("Menor volatilidade")
        st.dataframe(vola_menores[['Ativo', 'Volatilidade']].style.format(formato),
                     use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Exportar")
    col_pdf, col_csv = st.columns(2)

    with col_csv:
        st.download_button("Baixar planilha completa",
                           data=df_info.to_csv(index=False).encode('utf-8'),
                           file_name=f"movers_{ultimo_dia.strftime('%Y%m%d')}.csv",
                           mime="text/csv")

    with col_pdf:
        try:
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", 'B', 12)
            pdf.cell(0, 10, f"Market Movers - Ultimos {janela_movers} Dias Uteis", ln=True, align='C')
            pdf.ln(3)
            pdf.set_font("Helvetica", '', 10)
            pdf.cell(0, 6, ultimo_dia.strftime('%d/%m/%Y'), ln=True, align='C')
            formatar_tabela_pdf(pdf, 10, 45, "Maiores Retornos", ret_maiores, "Retorno")
            formatar_tabela_pdf(pdf, 110, 45, "Piores Retornos", ret_piores, "Retorno")
            formatar_tabela_pdf(pdf, 10, 90, "Maior Variacao de Volume", vol_maiores, "VolumeVar")
            formatar_tabela_pdf(pdf, 110, 90, "Menor Variacao de Volume", vol_piores, "VolumeVar")
            formatar_tabela_pdf(pdf, 10, 135, "Maior Volatilidade", vola_maiores, "Volatilidade")
            formatar_tabela_pdf(pdf, 110, 135, "Menor Volatilidade", vola_menores, "Volatilidade")
            try:
                conteudo = pdf.output(dest='S')
            except TypeError:
                conteudo = pdf.output()
            if isinstance(conteudo, str):
                conteudo = conteudo.encode('latin-1')
            else:
                conteudo = bytes(conteudo)
            st.download_button("Baixar PDF", data=conteudo,
                               file_name=f"zmarketMovers_{ultimo_dia.strftime('%Y%m%d')}.pdf",
                               mime="application/pdf")
        except ImportError:
            st.info("Instale fpdf2 para habilitar a exportacao em PDF.")


# ============================================================================
# ALERTA DE VOLUME
# ============================================================================

elif funcao == "Alerta de Volume":

    st.title("Alerta de Volume")

    st.sidebar.subheader("Parametros")
    janela_volume = st.sidebar.slider("Pregoes na media de volume", 2, 20, 3)
    corte_sigma = st.sidebar.slider("Desvios padrao para disparar", 1.0, 4.0, 2.0, 0.5)
    janela_retorno = st.sidebar.slider("Pregoes no retorno comparado", 5, 60, 21)

    inicio_leitura = (date.today() - timedelta(days=200)).isoformat()
    precos, volumes = ler_painel(universo_padrao, inicio_leitura, chave_do_dia, marca)
    ultimo_dia = volumes.index[-1]

    volume_medio = volumes.shift(1).rolling(window=janela_volume).mean()
    volume_desvio = volumes.shift(1).rolling(window=janela_volume).std()

    serie_ibov = precos[ibov].dropna()
    retorno_ibov = (serie_ibov.iloc[-1] / serie_ibov.iloc[-(janela_retorno + 1)] - 1) * 100

    alertas = []
    for ativo in ativos_yf:
        if ativo not in volumes.columns:
            continue
        try:
            volume_hoje = volumes.loc[ultimo_dia, ativo]
            media = volume_medio.loc[ultimo_dia, ativo]
            desvio = volume_desvio.loc[ultimo_dia, ativo]
            if pd.isna(volume_hoje) or pd.isna(media) or pd.isna(desvio) or media == 0:
                continue
            if volume_hoje <= media + corte_sigma * desvio:
                continue
            serie = precos[ativo]
            retorno = (serie.loc[ultimo_dia] / serie.iloc[-(janela_retorno + 1)] - 1) * 100
            alertas.append({'Ativo': ativo.replace('.SA', ''),
                            'Retorno (%)': round(retorno, 2),
                            'Volume medio': int(media),
                            'Volume atual': int(volume_hoje),
                            'PercentualVol (%)': round((volume_hoje / media - 1) * 100, 2),
                            'Movimento': 'alta' if retorno > 0 else 'baixa',
                            'Comparacao com IBOV': 'melhor que o IBOV' if retorno > retorno_ibov
                            else 'pior que o IBOV'})
        except Exception:
            continue

    st.write(f"Pregao analisado: {ultimo_dia.strftime('%d/%m/%Y')}. "
             f"Retorno do Ibovespa em {janela_retorno} pregoes: {retorno_ibov:.2f}%. "
             f"O alerta dispara quando o volume do dia supera a media de {janela_volume} pregoes anteriores "
             f"mais {corte_sigma:g} desvios padrao.")

    if alertas:
        df_alertas = pd.DataFrame(alertas).sort_values('PercentualVol (%)', ascending=False)
        st.dataframe(df_alertas, use_container_width=True, hide_index=True)
        st.download_button("Baixar alertas em CSV",
                           data=df_alertas.to_csv(index=False).encode('utf-8'),
                           file_name=f"alertas_volume_{ultimo_dia.strftime('%Y%m%d')}.csv",
                           mime="text/csv")
        st.divider()
        escolhido = st.selectbox("Ver o volume de", df_alertas['Ativo'].tolist())
        st.bar_chart(volumes[escolhido + '.SA'].dropna().tail(60))
    else:
        st.info("Nenhum ativo disparou o alerta no ultimo pregao com os parametros escolhidos.")


# ============================================================================
# PAIRS TRADING
# ============================================================================

elif funcao == "Pairs Trading":

    st.title("Pairs Trading")

    st.sidebar.subheader("Parametros")
    dias_janela = st.sidebar.slider("Dias corridos na janela", 90, 365, 189)
    corte_nivel = st.sidebar.number_input("p-valor minimo do ADF em nivel", value=0.10, step=0.01, format="%.2f")
    corte_residuo = st.sidebar.number_input("p-valor maximo do ADF nos residuos", value=0.01, step=0.01, format="%.2f")
    corte_pbeta = st.sidebar.number_input("p-valor maximo do beta", value=0.01, step=0.01, format="%.2f")
    cobertura_minima = st.sidebar.slider("Cobertura minima do ativo na janela", 0.50, 1.00, 0.95, 0.05)
    filtrar_mediana = st.sidebar.checkbox("Filtrar pela mediana da dependencia de cauda", value=True)

    inicio_leitura = (date.today() - timedelta(days=dias_janela)).isoformat()
    precos, volumes = ler_painel(tuple(ativos_yf), inicio_leitura, chave_do_dia, marca)

    if precos.empty:
        st.error("A base local nao tem cotacoes nessa janela. Verifique a pagina Base de Dados.")
        st.stop()

    cobertura = precos.notna().mean().sort_values()
    aproveitados = cobertura[cobertura >= cobertura_minima].index.tolist()
    quotes = precos[aproveitados].ffill().dropna(axis=1)
    quotes.columns = [c.replace('.SA', '') for c in quotes.columns]

    st.write(f"Janela de {precos.shape[0]} pregoes. {quotes.shape[1]} ativos aproveitados dos "
             f"{precos.shape[1]} presentes na base, exigindo pelo menos "
             f"{cobertura_minima * 100:.0f} por cento de dias negociados. "
             "A primeira execucao do dia leva alguns segundos; depois o resultado fica em cache.")

    descartados = cobertura[cobertura < cobertura_minima]
    if len(descartados) > 0:
        with st.expander(f"Ver os {len(descartados)} ativos descartados por falta de dados"):
            st.dataframe(pd.DataFrame({'Ativo': [i.replace('.SA', '') for i in descartados.index],
                                       'Dias com negocio (%)': (descartados.values * 100).round(1)}),
                         use_container_width=True, hide_index=True)

    if quotes.shape[1] < 2:
        st.error("Menos de dois ativos com serie completa nesta janela. "
                 "Reduza a exigencia de cobertura na barra lateral ou verifique a base de dados.")
        st.stop()

    if st.button("Rodar a varredura de pares") or "pairs_rodou" in st.session_state:
        st.session_state["pairs_rodou"] = True

        with st.spinner("Testando cointegracao, meia-vida e dependencia de cauda..."):
            resultado, diagnostico = rodar_pairs(quotes, corte_nivel, corte_residuo, corte_pbeta)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pares possiveis", f"{diagnostico['total_pares']:,}".replace(',', '.'))
        col2.metric("Ativos com raiz unitaria", diagnostico['elegiveis'])
        col3.metric("Pares cointegrados", diagnostico['cointegrados'])
        col4.metric("Pares abertos e validos", diagnostico['validos'])

        if resultado.empty:
            st.warning("Nenhum par sobreviveu aos filtros nesta janela.")
            st.stop()

        if filtrar_mediana:
            df = resultado[resultado['TailDep_Avg'] >= resultado['TailDep_Avg'].median()].copy()
        else:
            df = resultado.copy()

        df['acaoVende'] = np.where(df['DesvioAb'] < 0, df['Acao2'], df['Acao1'])
        df['acaoCompra'] = np.where(df['DesvioAb'] < 0, df['Acao1'], df['Acao2'])
        df = df.sort_values(by=['TailDep_Avg', 'OU'], ascending=[False, True])

        st.write(f"{len(df)} pares selecionados, com dependencia de cauda media de "
                 f"{df['TailDep_Avg'].mean():.4f} e meia-vida mediana de {df['OU'].median():.1f} dias.")

        st.divider()
        st.subheader("Pares selecionados")
        colunas_visiveis = ['Acao1', 'Acao2', 'acaoCompra', 'acaoVende', 'Beta',
                            'DesvioAb', 'DesvioP', 'OU', 'TailDep_Avg', 'PPR']
        st.dataframe(df[colunas_visiveis].style.format({
            'Beta': '{:.4f}', 'DesvioAb': '{:.4f}', 'DesvioP': '{:.4f}',
            'OU': '{:.1f}', 'TailDep_Avg': '{:.4f}', 'PPR': '{:.5f}'}),
            use_container_width=True, hide_index=True)

        st.download_button("Baixar pares em CSV", data=df.to_csv(index=False).encode('utf-8'),
                           file_name=f"pares_{date.today().strftime('%Y%m%d')}.csv", mime="text/csv")

        st.divider()
        st.subheader("Concentracao por ativo")
        col_compra, col_venda = st.columns(2)
        with col_compra:
            st.write("Mais indicados para compra")
            st.dataframe(df['acaoCompra'].value_counts().head(10).rename_axis('Acao').reset_index(name='Compras'),
                         use_container_width=True, hide_index=True)
        with col_venda:
            st.write("Mais indicados para venda")
            st.dataframe(df['acaoVende'].value_counts().head(10).rename_axis('Acao').reset_index(name='Vendas'),
                         use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Graficos dos pares")
        quantos_graficos = st.slider("Quantos pares plotar na tela", 1, max(2, min(20, len(df))),
                                     min(5, len(df)))

        for i in range(min(quantos_graficos, len(df))):
            linha = df.iloc[i]
            figura, eixo = plt.subplots(figsize=(10, 4))
            serie1 = quotes[linha['Acao1']]
            serie2 = quotes[linha['Acao2']]
            eixo.plot(serie1.index, serie1.values, color='tab:red', label=linha['Acao1'])
            eixo.set_ylabel(linha['Acao1'], color='tab:red')
            eixo2 = eixo.twinx()
            eixo2.plot(serie2.index, serie2.values, color='tab:blue', label=linha['Acao2'])
            eixo2.set_ylabel(linha['Acao2'], color='tab:blue')
            linhas1, rotulos1 = eixo.get_legend_handles_labels()
            linhas2, rotulos2 = eixo2.get_legend_handles_labels()
            eixo.legend(linhas1 + linhas2, rotulos1 + rotulos2, loc='upper left')
            eixo.grid(False)
            st.pyplot(figura)
            st.caption(f"Vender {linha['acaoVende']} e comprar {linha['acaoCompra']}. "
                       f"Desvio de {linha['DesvioAb']:.4f} contra um desvio padrao de {linha['DesvioP']:.4f}. "
                       f"Meia-vida estimada de {linha['OU']:.1f} dias.")
            plt.close(figura)

        st.divider()
        if st.button("Gerar PDF com todos os pares"):
            buffer_pdf = io.BytesIO()
            with PdfPages(buffer_pdf) as pdf_pares:
                for i in range(len(df)):
                    linha = df.iloc[i]
                    figura, eixo = plt.subplots(figsize=(10, 6))
                    serie1 = quotes[linha['Acao1']]
                    serie2 = quotes[linha['Acao2']]
                    eixo.plot(serie1.index, serie1.values, color='tab:red', label=linha['Acao1'])
                    eixo.set_ylabel(linha['Acao1'], color='tab:red', fontsize=12)
                    eixo.set_xlabel('Data', fontsize=12)
                    eixo2 = eixo.twinx()
                    eixo2.plot(serie2.index, serie2.values, color='tab:blue', label=linha['Acao2'])
                    eixo2.set_ylabel(linha['Acao2'], color='tab:blue', fontsize=12)
                    linhas1, rotulos1 = eixo.get_legend_handles_labels()
                    linhas2, rotulos2 = eixo2.get_legend_handles_labels()
                    eixo.legend(linhas1 + linhas2, rotulos1 + rotulos2, loc='upper left', fontsize='medium')
                    eixo.grid(False)
                    pdf_pares.savefig(figura)
                    plt.close(figura)
            st.download_button("Baixar PDF dos pares", data=buffer_pdf.getvalue(),
                               file_name=f"zpairsTrading_{date.today().strftime('%Y%m%d')}.pdf",
                               mime="application/pdf")


# ============================================================================
# PUT-CALL RATIO
# ============================================================================

elif funcao == "Put-Call Ratio":

    st.title("Put-Call Ratio")

    st.sidebar.subheader("Parametros")
    subjacente = st.sidebar.text_input("Subjacente", value="BOVA11").upper()

    try:
        lista_vencimentos = baixar_vencimentos(subjacente)
    except Exception:
        lista_vencimentos = []

    if lista_vencimentos:
        vencimento = st.sidebar.selectbox("Vencimento", lista_vencimentos)
    else:
        vencimento = st.sidebar.text_input("Vencimento (AAAA-MM-DD)", value="2026-09-18")

    con = sqlite3.connect(DB_PCR)
    cur = con.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS pcr (
                       data TEXT, subjacente TEXT, vencimento TEXT,
                       pcr_negocios REAL, pcr_volume REAL,
                       PRIMARY KEY (data, subjacente, vencimento))''')
    colunas_pk = [c[1] for c in cur.execute("PRAGMA table_info(pcr)").fetchall() if c[5] > 0]
    if colunas_pk == ['data']:
        cur.execute('ALTER TABLE pcr RENAME TO pcr_antigo')
        cur.execute('''CREATE TABLE pcr (
                           data TEXT, subjacente TEXT, vencimento TEXT,
                           pcr_negocios REAL, pcr_volume REAL,
                           PRIMARY KEY (data, subjacente, vencimento))''')
        cur.execute('''INSERT INTO pcr SELECT data, subjacente, vencimento,
                       pcr_negocios, pcr_volume FROM pcr_antigo''')
        cur.execute('DROP TABLE pcr_antigo')
        con.commit()
        st.caption("Chave primaria do banco migrada para data, subjacente e vencimento.")

    try:
        chain = baixar_chain(subjacente, vencimento)
    except Exception as erro:
        st.error(f"Nao foi possivel consultar a cadeia de opcoes: {erro}")
        con.close()
        st.stop()

    if chain.empty:
        st.error("A cadeia retornou vazia para esse subjacente e vencimento.")
        con.close()
        st.stop()

    calls = chain[chain['tipo'] == 'CALL']
    puts = chain[chain['tipo'] == 'PUT']

    pcr_negocios = puts['negocios'].sum() / calls['negocios'].sum()
    pcr_volume = puts['volume'].sum() / calls['volume'].sum()
    hoje = date.today().isoformat()

    cur.execute('''SELECT data, pcr_negocios, pcr_volume FROM pcr
                   WHERE subjacente = ? AND data < ?
                   ORDER BY data DESC LIMIT 1''', (subjacente, hoje))
    ultimo = cur.fetchone()

    cur.execute('''INSERT INTO pcr (data, subjacente, vencimento, pcr_negocios, pcr_volume)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(data, subjacente, vencimento) DO UPDATE SET
                       pcr_negocios = excluded.pcr_negocios,
                       pcr_volume = excluded.pcr_volume''',
                (hoje, subjacente, vencimento, pcr_negocios, pcr_volume))
    con.commit()

    col1, col2, col3 = st.columns(3)
    col1.metric("PCR negocios", f"{pcr_negocios:.4f}",
                f"{pcr_negocios - ultimo[1]:+.4f}" if ultimo else None)
    col2.metric("PCR volume", f"{pcr_volume:.4f}",
                f"{pcr_volume - ultimo[2]:+.4f}" if ultimo else None)
    col3.metric("Leitura", "Bearish" if pcr_negocios > 1 else "Bullish")

    if ultimo:
        st.write(f"Comparado com o registro de {ultimo[0]}, o PCR de negocios "
                 f"{'subiu' if pcr_negocios > ultimo[1] else 'caiu'} "
                 f"{abs((pcr_negocios / ultimo[1] - 1) * 100):.1f} por cento e o PCR de volume "
                 f"{'subiu' if pcr_volume > ultimo[2] else 'caiu'} "
                 f"{abs((pcr_volume / ultimo[2] - 1) * 100):.1f} por cento. "
                 "Razoes acima de um indicam maior atividade em puts do que em calls.")
    else:
        st.write("Primeiro registro para esse subjacente, sem dia anterior para comparar.")

    st.divider()

    st.subheader("Historico")
    historico = pd.read_sql_query(
        "SELECT data, vencimento, pcr_negocios, pcr_volume FROM pcr WHERE subjacente = ? ORDER BY data",
        con, params=(subjacente,))
    con.close()

    if len(historico) > 1:
        grafico = historico.copy()
        grafico['data'] = pd.to_datetime(grafico['data'])
        st.line_chart(grafico.set_index('data')[['pcr_negocios', 'pcr_volume']])
        st.dataframe(historico.sort_values('data', ascending=False),
                     use_container_width=True, hide_index=True)
        st.download_button("Baixar historico em CSV", data=historico.to_csv(index=False).encode('utf-8'),
                           file_name=f"pcr_{subjacente}.csv", mime="text/csv")

    st.divider()

    st.subheader("Cadeia de opcoes")
    col_calls, col_puts = st.columns(2)
    with col_calls:
        st.write(f"Calls: {len(calls)} series, {int(calls['negocios'].sum())} negocios")
        st.dataframe(calls[['ativo', 'strike', 'preco', 'negocios', 'volume']],
                     use_container_width=True, hide_index=True)
    with col_puts:
        st.write(f"Puts: {len(puts)} series, {int(puts['negocios'].sum())} negocios")
        st.dataframe(puts[['ativo', 'strike', 'preco', 'negocios', 'volume']],
                     use_container_width=True, hide_index=True)


# ============================================================================
# REGIME DE VOLATILIDADE
# ============================================================================

elif funcao == "Regime de Volatilidade":

    st.title("Regime de Volatilidade")

    st.sidebar.subheader("Parametros")
    anos_historico = st.sidebar.slider("Anos de historico", 5, 25, 20)
    janela_vol = st.sidebar.slider("Janela do desvio movel", 10, 90, 30)
    n_estados = st.sidebar.slider("Numero de regimes", 2, 4, 2)

    inicio_leitura = (date.today() - timedelta(days=anos_historico * 365)).isoformat()
    indice = ler_ativo(ibov, inicio_leitura, chave_do_dia, marca)

    if indice.empty:
        st.error("Sem historico do Ibovespa na base local.")
        st.stop()

    try:
        painel_regime, transicao, medias_regime = rodar_markov(indice['Close'], janela_vol, n_estados)
    except ImportError:
        st.error("A biblioteca hmmlearn nao esta instalada. Inclua hmmlearn no requirements.txt.")
        st.stop()

    nomes = {0: "Baixa volatilidade", 1: "Alta volatilidade"} if n_estados == 2 else \
        {i: f"Regime {i + 1}" for i in range(n_estados)}
    cores = ['green', 'red', 'orange', 'purple']
    regime_atual = int(painel_regime['regime'].iloc[-1])

    col1, col2, col3 = st.columns(3)
    col1.metric("Regime atual", nomes[regime_atual])
    col2.metric("Volatilidade movel", f"{painel_regime['volatilidade'].iloc[-1] * 100:.2f}%")
    col3.metric("Persistencia do regime", f"{transicao[regime_atual, regime_atual] * 100:.1f}%")

    figura, eixo = plt.subplots(figsize=(16, 6))
    for estado in range(n_estados):
        mascara = painel_regime['regime'] == estado
        eixo.plot(painel_regime.index[mascara], painel_regime['volatilidade'][mascara],
                  color=cores[estado], marker='.', markersize=3, linestyle='None')
    eixo.set_ylabel(f'Volatilidade (desvio padrao movel de {janela_vol} dias)')
    eixo.set_xlabel('Data')

    eixo2 = eixo.twinx()
    eixo2.plot(painel_regime.index, painel_regime['preco'], color='navy', linewidth=1, alpha=0.6)
    eixo2.set_ylabel('Ibovespa (pontos)', color='navy')
    eixo2.tick_params(axis='y', labelcolor='navy')

    elementos = [Line2D([0], [0], color=cores[e], marker='.', linestyle='None', label=nomes[e])
                 for e in range(n_estados)]
    elementos.append(Line2D([0], [0], color='navy', linewidth=1.5, label='Ibovespa (pontos)'))
    eixo.legend(handles=elementos, loc='upper left')
    eixo.set_title('Mudanca de Regime na Volatilidade do Ibovespa')
    figura.tight_layout()
    st.pyplot(figura)
    plt.close(figura)

    st.divider()

    col_transicao, col_medias = st.columns(2)
    with col_transicao:
        st.subheader("Matriz de transicao")
        matriz = pd.DataFrame(transicao, index=[nomes[i] for i in range(n_estados)],
                              columns=[nomes[i] for i in range(n_estados)])
        st.dataframe(matriz.style.format("{:.4f}"), use_container_width=True)
        st.caption("Cada linha traz a probabilidade de migrar do regime da linha para o regime da coluna "
                   "no pregao seguinte.")
    with col_medias:
        st.subheader("Volatilidade media por regime")
        st.dataframe(pd.DataFrame({'Regime': [nomes[i] for i in range(n_estados)],
                                   'Volatilidade media (%)': medias_regime * 100,
                                   'Dias no regime': [int((painel_regime['regime'] == i).sum())
                                                      for i in range(n_estados)]}
                                  ).style.format({'Volatilidade media (%)': '{:.3f}'}),
                     use_container_width=True, hide_index=True)


# ============================================================================
# INFORMACOES DO ATIVO
# ============================================================================

elif funcao == "Informacoes do Ativo":

    st.title(f"Informacoes: {ticker_display}")

    dados = ler_ativo(ticker, str(data_inicio), str(data_fim), marca)
    if dados.empty:
        st.error("Nenhum dado encontrado para o ticker informado.")
        st.stop()

    dados['Retorno'] = dados['Close'].pct_change()
    dados = dados.dropna()

    col1, col2, col3, col4 = st.columns(4)
    preco_atual = dados['Close'].iloc[-1]
    preco_anterior = dados['Close'].iloc[-2]
    variacao = ((preco_atual - preco_anterior) / preco_anterior) * 100

    col1.metric("Preco Atual", f"R$ {preco_atual:.2f}", f"{variacao:.2f}%")
    col2.metric("Maximo no Periodo", f"R$ {dados['High'].max():.2f}")
    col3.metric("Minimo no Periodo", f"R$ {dados['Low'].min():.2f}")
    col4.metric("Volume Medio", f"{dados['Volume'].mean():,.0f}".replace(',', '.'))

    st.line_chart(dados['Close'])

    st.divider()

    info = buscar_info_ticker(ticker)

    if info and len(info) > 5:
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.subheader("Dados Cadastrais")
            st.write(f"Nome: {info.get('longName', 'N/A')}")
            st.write(f"Setor: {info.get('sector', 'N/A')}")
            st.write(f"Industria: {info.get('industry', 'N/A')}")
            st.write(f"Pais: {info.get('country', 'N/A')}")
            st.write(f"Cidade: {info.get('city', 'N/A')}")
            st.write(f"Site: {info.get('website', 'N/A')}")
            st.write(f"Funcionarios: {info.get('fullTimeEmployees', 'N/A'):,}"
                     if isinstance(info.get('fullTimeEmployees'), (int, float)) else "Funcionarios: N/A")
        with col_info2:
            st.subheader("Dados Fundamentalistas")
            st.write(f"Market Cap: {info.get('marketCap', 'N/A'):,}"
                     if isinstance(info.get('marketCap'), (int, float)) else "Market Cap: N/A")
            st.write(f"P/L: {info.get('trailingPE', 'N/A'):.2f}"
                     if isinstance(info.get('trailingPE'), (int, float)) else "P/L: N/A")
            st.write(f"P/VP: {info.get('priceToBook', 'N/A'):.2f}"
                     if isinstance(info.get('priceToBook'), (int, float)) else "P/VP: N/A")
            st.write(f"Dividend Yield: {info.get('dividendYield', 0) * 100:.2f}%"
                     if isinstance(info.get('dividendYield'), (int, float)) else "Dividend Yield: N/A")
            st.write(f"ROE: {info.get('returnOnEquity', 0) * 100:.2f}%"
                     if isinstance(info.get('returnOnEquity'), (int, float)) else "ROE: N/A")
            st.write(f"Beta: {info.get('beta', 'N/A'):.2f}"
                     if isinstance(info.get('beta'), (int, float)) else "Beta: N/A")

        st.divider()
        st.subheader("Descricao da Empresa")
        descricao = info.get('longBusinessSummary', 'Descricao nao disponivel.')
        if isinstance(descricao, str):
            descricao = descricao.replace("\n", " ").strip()
        st.write(descricao)
    else:
        st.warning("Nao foi possivel carregar informacoes cadastrais (limite de requisicoes do Yahoo). "
                   "Os precos vem da base local e nao dependem dessa consulta.")


# ============================================================================
# ESTATISTICAS DESCRITIVAS
# ============================================================================

elif funcao == "Estatisticas Descritivas":

    st.title(f"Estatisticas Descritivas: {ticker_display}")

    dados = ler_ativo(ticker, str(data_inicio), str(data_fim), marca)
    if dados.empty:
        st.error("Nenhum dado encontrado para o ticker informado.")
        st.stop()
    dados['Retorno'] = dados['Close'].pct_change()
    dados = dados.dropna()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Precos de Fechamento")
        estatisticas_preco = {
            'Media': dados['Close'].mean(), 'Mediana': dados['Close'].median(),
            'Desvio Padrao': dados['Close'].std(), 'Variancia': dados['Close'].var(),
            'Minimo': dados['Close'].min(), 'Maximo': dados['Close'].max(),
            'Amplitude': dados['Close'].max() - dados['Close'].min(),
            'Coef. Variacao (%)': (dados['Close'].std() / dados['Close'].mean()) * 100,
            'Assimetria': dados['Close'].skew(), 'Curtose': dados['Close'].kurtosis()}
        df_preco = pd.DataFrame(estatisticas_preco.items(), columns=['Estatistica', 'Valor'])
        df_preco['Valor'] = df_preco['Valor'].apply(lambda x: f"{x:.4f}")
        st.dataframe(df_preco, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Retornos Diarios")
        estatisticas_retorno = {
            'Media': dados['Retorno'].mean(), 'Mediana': dados['Retorno'].median(),
            'Desvio Padrao': dados['Retorno'].std(), 'Variancia': dados['Retorno'].var(),
            'Minimo': dados['Retorno'].min(), 'Maximo': dados['Retorno'].max(),
            'Amplitude': dados['Retorno'].max() - dados['Retorno'].min(),
            'Coef. Variacao': dados['Retorno'].std() / abs(dados['Retorno'].mean())
            if dados['Retorno'].mean() != 0 else np.nan,
            'Assimetria': dados['Retorno'].skew(), 'Curtose': dados['Retorno'].kurtosis()}
        df_retorno = pd.DataFrame(estatisticas_retorno.items(), columns=['Estatistica', 'Valor'])
        df_retorno['Valor'] = df_retorno['Valor'].apply(lambda x: f"{x:.6f}")
        st.dataframe(df_retorno, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Distribuicao dos Retornos")
    figura, eixo = plt.subplots(figsize=(10, 4))
    eixo.hist(dados['Retorno'], bins=50, color='tab:blue', alpha=0.7, density=True)
    grade = np.linspace(dados['Retorno'].min(), dados['Retorno'].max(), 200)
    eixo.plot(grade, stats.norm.pdf(grade, dados['Retorno'].mean(), dados['Retorno'].std()),
              color='tab:red', linewidth=1.5, label='Normal ajustada')
    eixo.legend()
    eixo.set_xlabel('Retorno diario')
    st.pyplot(figura)
    plt.close(figura)

    st.divider()

    st.subheader("Percentis dos Retornos")
    percentis = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    df_percentis = pd.DataFrame({'Percentil': [f"{p}%" for p in percentis],
                                 'Valor': [np.percentile(dados['Retorno'], p) for p in percentis]})
    df_percentis['Valor'] = df_percentis['Valor'].apply(lambda x: f"{x:.6f}")
    st.dataframe(df_percentis.T, use_container_width=True)

    st.divider()

    st.subheader("Testes de Normalidade")
    jb_stat, jb_pvalue = stats.jarque_bera(dados['Retorno'])
    amostra_shapiro = dados['Retorno'].sample(min(5000, len(dados['Retorno'])), random_state=42)
    sw_stat, sw_pvalue = stats.shapiro(amostra_shapiro)

    col_teste1, col_teste2 = st.columns(2)
    with col_teste1:
        st.write("Teste Jarque-Bera")
        st.write(f"Estatistica: {jb_stat:.4f}")
        st.write(f"P-valor: {jb_pvalue:.6f}")
        st.write(f"Conclusao (5%): {'Rejeita normalidade' if jb_pvalue < 0.05 else 'Nao rejeita normalidade'}")
    with col_teste2:
        st.write("Teste Shapiro-Wilk")
        st.write(f"Estatistica: {sw_stat:.4f}")
        st.write(f"P-valor: {sw_pvalue:.6f}")
        st.write(f"Conclusao (5%): {'Rejeita normalidade' if sw_pvalue < 0.05 else 'Nao rejeita normalidade'}")


# ============================================================================
# ANALISE DE RETORNOS
# ============================================================================

elif funcao == "Analise de Retornos":

    st.title(f"Analise de Retornos: {ticker_display}")

    dados = ler_ativo(ticker, str(data_inicio), str(data_fim), marca)
    if dados.empty:
        st.error("Nenhum dado encontrado para o ticker informado.")
        st.stop()
    dados['Retorno'] = dados['Close'].pct_change()
    dados = dados.dropna()

    dias_uteis = 252
    retorno_medio_diario = dados['Retorno'].mean()
    retorno_anualizado = retorno_medio_diario * dias_uteis
    volatilidade_diaria = dados['Retorno'].std()
    volatilidade_anualizada = volatilidade_diaria * np.sqrt(dias_uteis)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Retorno Medio Diario", f"{retorno_medio_diario * 100:.4f}%")
    col2.metric("Retorno Anualizado", f"{retorno_anualizado * 100:.2f}%")
    col3.metric("Volatilidade Diaria", f"{volatilidade_diaria * 100:.4f}%")
    col4.metric("Volatilidade Anualizada", f"{volatilidade_anualizada * 100:.2f}%")

    st.divider()

    retorno_total = (dados['Close'].iloc[-1] / dados['Close'].iloc[0]) - 1
    retornos_mensais = dados['Close'].resample('ME').last().pct_change().dropna()

    st.subheader("Retorno acumulado")
    st.line_chart((1 + dados['Retorno']).cumprod() - 1)

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Retorno Total no Periodo: {retorno_total * 100:.2f}%")
        st.write(f"Numero de Observacoes: {len(dados)}")
        st.write(f"Dias com Retorno Positivo: {(dados['Retorno'] > 0).sum()} "
                 f"({(dados['Retorno'] > 0).mean() * 100:.1f}%)")
        st.write(f"Dias com Retorno Negativo: {(dados['Retorno'] < 0).sum()} "
                 f"({(dados['Retorno'] < 0).mean() * 100:.1f}%)")
    with col2:
        st.write(f"Maior Retorno Diario: {dados['Retorno'].max() * 100:.2f}%")
        st.write(f"Menor Retorno Diario: {dados['Retorno'].min() * 100:.2f}%")
        st.write(f"Retorno Medio Mensal: {retornos_mensais.mean() * 100:.2f}%"
                 if len(retornos_mensais) > 0 else "Retorno Medio Mensal: N/A")
        st.write(f"Volatilidade Mensal: {retornos_mensais.std() * 100:.2f}%"
                 if len(retornos_mensais) > 0 else "Volatilidade Mensal: N/A")

    if len(retornos_mensais) > 0:
        st.divider()
        st.subheader("Retornos Mensais")
        st.dataframe(pd.DataFrame({'Mes': retornos_mensais.index.strftime('%m-%Y'),
                                   'Retorno (%)': retornos_mensais.values * 100}),
                     use_container_width=True, hide_index=True)


# ============================================================================
# METRICAS DE RISCO
# ============================================================================

elif funcao == "Metricas de Risco":

    st.title(f"Metricas de Risco: {ticker_display}")

    st.sidebar.subheader("Parametros de Risco")
    nivel_confianca = st.sidebar.slider("Nivel de Confianca VaR (%)", 90, 99, 95)
    taxa_livre_risco = st.sidebar.number_input("Taxa Livre de Risco (% a.a.)", value=10.0, step=0.5)

    dados = ler_ativo(ticker, str(data_inicio), str(data_fim), marca)
    if dados.empty:
        st.error("Nenhum dado encontrado para o ticker informado.")
        st.stop()
    dados['Retorno'] = dados['Close'].pct_change()
    dados = dados.dropna()

    dias_uteis = 252
    retorno_anualizado = dados['Retorno'].mean() * dias_uteis
    volatilidade_anualizada = dados['Retorno'].std() * np.sqrt(dias_uteis)
    rf_anual = taxa_livre_risco / 100

    z_score = stats.norm.ppf(1 - nivel_confianca / 100)
    var_parametrico = dados['Retorno'].mean() + z_score * dados['Retorno'].std()
    var_historico = np.percentile(dados['Retorno'], 100 - nivel_confianca)
    cvar = dados['Retorno'][dados['Retorno'] <= var_historico].mean()

    sharpe = (retorno_anualizado - rf_anual) / volatilidade_anualizada
    retornos_negativos = dados['Retorno'][dados['Retorno'] < 0]
    downside_deviation = retornos_negativos.std() * np.sqrt(dias_uteis)
    sortino = (retorno_anualizado - rf_anual) / downside_deviation if downside_deviation > 0 else np.nan

    preco_acumulado = (1 + dados['Retorno']).cumprod()
    pico = preco_acumulado.expanding(min_periods=1).max()
    drawdown = (preco_acumulado - pico) / pico
    max_drawdown = drawdown.min()
    calmar = retorno_anualizado / abs(max_drawdown) if max_drawdown != 0 else np.nan

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Value at Risk")
        st.metric(f"VaR Parametrico ({nivel_confianca}%)", f"{var_parametrico * 100:.4f}%")
        st.metric(f"VaR Historico ({nivel_confianca}%)", f"{var_historico * 100:.4f}%")
        st.metric("CVaR / Expected Shortfall", f"{cvar * 100:.4f}%")
    with col2:
        st.subheader("Indices de Performance")
        st.metric("Sharpe Ratio", f"{sharpe:.4f}")
        st.metric("Sortino Ratio", f"{sortino:.4f}")
        st.metric("Calmar Ratio", f"{calmar:.4f}")
    with col3:
        st.subheader("Drawdown")
        st.metric("Maximum Drawdown", f"{max_drawdown * 100:.2f}%")
        st.metric("Volatilidade Anualizada", f"{volatilidade_anualizada * 100:.2f}%")
        st.metric("Downside Deviation", f"{downside_deviation * 100:.2f}%")

    st.divider()
    st.subheader("Drawdown ao longo do periodo")
    st.area_chart(drawdown)

    st.divider()
    st.subheader("Interpretacao")
    st.write(f"Com {nivel_confianca}% de confianca, a perda maxima esperada em um dia e de "
             f"{abs(var_historico) * 100:.4f}% (VaR Historico). Caso essa perda seja excedida, "
             f"a perda media esperada e de {abs(cvar) * 100:.4f}% (CVaR).")
    st.write(f"O Sharpe Ratio de {sharpe:.4f} indica "
             f"{'retorno ajustado ao risco positivo' if sharpe > 0 else 'retorno ajustado ao risco negativo'} "
             f"considerando taxa livre de risco de {taxa_livre_risco}% a.a.")


# ============================================================================
# CORRELACAO ENTRE ATIVOS
# ============================================================================

elif funcao == "Correlacao entre Ativos":

    st.title("Analise de Correlacao")

    dados_multi, _ = ler_painel(tuple(lista_tickers), str(data_inicio), str(data_fim), marca)

    if dados_multi.empty:
        st.error("Nao foi possivel ler os dados dos ativos na base local.")
        st.stop()

    dados_multi = dados_multi.dropna()
    dados_multi.columns = [col.replace(".SA", "") for col in dados_multi.columns]
    retornos_multi = dados_multi.pct_change().dropna()

    st.subheader("Matriz de Correlacao dos Retornos")
    st.dataframe(retornos_multi.corr().style.format("{:.4f}").background_gradient(
        cmap='RdYlGn', vmin=-1, vmax=1), use_container_width=True)

    st.divider()
    st.subheader("Matriz de Covariancia dos Retornos")
    st.dataframe(retornos_multi.cov().style.format("{:.6f}"), use_container_width=True)

    st.divider()
    st.subheader("Precos normalizados na base 100")
    st.line_chart(dados_multi / dados_multi.iloc[0] * 100)

    st.divider()
    st.subheader("Estatisticas Comparativas")
    st.dataframe(pd.DataFrame({
        'Retorno Medio Diario (%)': retornos_multi.mean() * 100,
        'Volatilidade Diaria (%)': retornos_multi.std() * 100,
        'Retorno Anualizado (%)': retornos_multi.mean() * 252 * 100,
        'Volatilidade Anualizada (%)': retornos_multi.std() * np.sqrt(252) * 100,
        'Sharpe (rf=10%)': (retornos_multi.mean() * 252 - 0.10) / (retornos_multi.std() * np.sqrt(252))
    }).style.format("{:.4f}"), use_container_width=True)


# ============================================================================
# DOWNLOAD DE DADOS
# ============================================================================

elif funcao == "Download de Dados":

    st.title("Download de Dados")

    st.sidebar.subheader("Opcoes de Download")
    tipo_dado = st.sidebar.selectbox("Tipo de Dado", ["Precos OHLCV", "Retornos", "Ambos"])
    formato = st.sidebar.selectbox("Formato", ["CSV", "Excel"])

    dados = ler_ativo(ticker, str(data_inicio), str(data_fim), marca)
    if dados.empty:
        st.error("Nenhum dado encontrado para o ticker informado.")
        st.stop()
    dados['Retorno'] = dados['Close'].pct_change()
    dados['Retorno_Log'] = np.log(dados['Close'] / dados['Close'].shift(1))
    dados = dados.dropna()

    if tipo_dado == "Precos OHLCV":
        df_download = dados[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    elif tipo_dado == "Retornos":
        df_download = dados[['Retorno', 'Retorno_Log']].copy()
    else:
        df_download = dados[['Open', 'High', 'Low', 'Close', 'Volume', 'Retorno', 'Retorno_Log']].copy()

    df_download.index.name = 'Data'

    st.subheader("Preview dos Dados")
    st.write(f"Total de observacoes: {len(df_download)}")
    st.write(f"Periodo: {df_download.index[0].strftime('%d-%m-%Y')} a {df_download.index[-1].strftime('%d-%m-%Y')}")
    st.dataframe(df_download.head(20), use_container_width=True)

    st.divider()

    if formato == "CSV":
        st.download_button("Baixar CSV", data=df_download.to_csv(),
                           file_name=f"{ticker_display}_{tipo_dado.lower().replace(' ', '_')}.csv",
                           mime="text/csv")
    else:
        buffer = io.BytesIO()
        df_download.to_excel(buffer, engine='openpyxl')
        st.download_button("Baixar Excel", data=buffer.getvalue(),
                           file_name=f"{ticker_display}_{tipo_dado.lower().replace(' ', '_')}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.divider()
    st.subheader("Estatisticas Rapidas")
    st.dataframe(df_download.describe(), use_container_width=True)


# ============================================================================
# BASE DE DADOS
# ============================================================================

elif funcao == "Base de Dados":

    st.title("Base de Dados")

    controle = ler_controle(marca)
    tamanho = DB_COTACOES.stat().st_size / (1024 * 1024) if DB_COTACOES.exists() else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tickers na base", len(controle))
    col2.metric("Observacoes", f"{resumo_base['total_linhas']:,}".replace(',', '.'))
    col3.metric("Ultimo pregao", resumo_base['ultima_data'] or "vazio")
    col4.metric("Tamanho do arquivo", f"{tamanho:.1f} MB")

    st.write("A base guarda abertura, maxima, minima, fechamento e volume ajustados por proventos, "
             "com chave primaria de data e ticker. A cada abertura do site o aplicativo consulta a "
             f"ultima data gravada de cada ticker e baixa apenas dali para frente, com {MARGEM_REVISAO} "
             "dias de margem para capturar revisoes do Yahoo. Se o fechamento de alguma data ja gravada "
             f"divergir mais de {TOLERANCIA_AJUSTE * 100:.1f} por cento do valor recem baixado, o que "
             "acontece quando ha dividendo ou desdobramento, a serie inteira daquele ticker e refeita "
             "automaticamente.")

    st.caption(f"Nesta sessao: {resumo_base['completos']} cargas completas, "
               f"{resumo_base['incrementais']} atualizacoes incrementais, "
               f"{resumo_base['reajustados']} series refeitas por ajuste de proventos, "
               f"{resumo_base['linhas_gravadas']} linhas gravadas.")

    st.divider()

    st.subheader("Situacao por ticker")
    if not controle.empty:
        exibicao = controle.copy()
        exibicao['ultima_coleta'] = pd.to_datetime(exibicao['ultima_coleta']).dt.strftime('%d/%m/%Y %H:%M')
        st.dataframe(exibicao.rename(columns={'ticker': 'Ticker', 'primeiro_dado': 'Primeiro dado',
                                              'ultimo_dado': 'Ultimo dado', 'ultima_coleta': 'Ultima coleta',
                                              'linhas': 'Observacoes'}),
                     use_container_width=True, hide_index=True)

        atrasados = controle[controle['ultimo_dado'].fillna('') < (resumo_base['ultima_data'] or '')]
        if not atrasados.empty:
            st.warning(f"{len(atrasados)} tickers estao com a ultima data anterior ao pregao mais recente "
                       "da base. Em geral sao papeis sem negociacao ou ja descontinuados: "
                       + ", ".join(atrasados['ticker'].str.replace('.SA', '', regex=False).tolist()))

    st.divider()

    st.subheader("Manutencao")
    st.write("A reconstrucao apaga e rebaixa todo o historico dos tickers do universo padrao. "
             "Use apenas se desconfiar da integridade dos dados, porque a operacao leva alguns minutos "
             "e faz uma coleta pesada no Yahoo.")
    confirmar = st.checkbox("Confirmo que quero reconstruir a base do zero")
    if st.button("Reconstruir base completa", disabled=not confirmar):
        atualizar_base(universo_padrao, True, datetime.now().isoformat())
        st.cache_data.clear()
        st.success("Base reconstruida.")
        st.rerun()

    st.divider()

    st.subheader("Ultimos registros gravados")
    con = sqlite3.connect(DB_COTACOES)
    amostra = pd.read_sql_query('SELECT * FROM cotacoes ORDER BY data DESC LIMIT 500', con)
    con.close()
    st.dataframe(amostra, use_container_width=True, hide_index=True)
    st.download_button("Baixar as ultimas 500 linhas", data=amostra.to_csv(index=False).encode('utf-8'),
                       file_name="cotacoes_amostra.csv", mime="text/csv")


st.sidebar.divider()
st.sidebar.caption("Harpa Quant")
st.sidebar.caption("Dados: Yahoo Finance e opcoes.net.br")
st.sidebar.caption(f"Atualizado: {datetime.now().strftime('%d-%m-%Y %H:%M')}")