import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from scipy import stats
import io

st.set_page_config(page_title="Harpa Quant", layout="wide", page_icon="harpa.jpg")

st.title("Harpa Quant")

# Sidebar - Menu principal
st.sidebar.image("harpa.jpg", width=120)
st.sidebar.header("Menu")

funcao = st.sidebar.selectbox("Funcao", [
    "Informacoes do Ativo",
    "Estatisticas Descritivas",
    "Analise de Retornos",
    "Metricas de Risco",
    "Correlacao entre Ativos",
    "Download de Dados"
])

st.sidebar.divider()
st.sidebar.subheader("Configuracoes de Dados")

ticker_input = st.sidebar.text_input("Ticker Principal", value="PETR4")
ticker = ticker_input.upper() if ticker_input.upper().endswith(".SA") else f"{ticker_input.upper()}.SA"
ticker_display = ticker.replace(".SA", "")
data_inicio = st.sidebar.date_input("Data Inicio", value=datetime.now() - timedelta(days=365), format="DD/MM/YYYY")
data_fim = st.sidebar.date_input("Data Fim", value=datetime.now(), format="DD/MM/YYYY")

# Download dos dados principais com cache
@st.cache_data(ttl=300)
def baixar_dados(ticker, data_inicio_str, data_fim_str):
    return yf.download(ticker, start=data_inicio_str, end=data_fim_str, progress=False)

dados = baixar_dados(ticker, str(data_inicio), str(data_fim))

if dados.empty:
    st.error("Nenhum dado encontrado para o ticker informado.")
    st.stop()

if isinstance(dados.columns, pd.MultiIndex):
    dados.columns = dados.columns.get_level_values(0)

# Calculo de retornos
dados['Retorno'] = dados['Close'].pct_change()
dados['Retorno_Log'] = np.log(dados['Close'] / dados['Close'].shift(1))
dados = dados.dropna()


# ============ INFORMACOES DO ATIVO ============
if funcao == "Informacoes do Ativo":
    
    st.header(f"Informacoes: {ticker_display}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    preco_atual = dados['Close'].iloc[-1]
    preco_anterior = dados['Close'].iloc[-2]
    variacao = ((preco_atual - preco_anterior) / preco_anterior) * 100
    
    col1.metric("Preco Atual", f"R$ {preco_atual:.2f}", f"{variacao:.2f}%")
    col2.metric("Maximo no Periodo", f"R$ {dados['High'].max():.2f}")
    col3.metric("Minimo no Periodo", f"R$ {dados['Low'].min():.2f}")
    col4.metric("Volume Medio", f"{dados['Volume'].mean():,.0f}")
    
    st.divider()
    
    # Buscar informacoes da empresa com cache
    @st.cache_data(ttl=3600, show_spinner="Carregando informacoes...")
    def buscar_info_ticker(ticker):
        import time
        info = {}
        fast = {}
        
        for tentativa in range(3):
            try:
                session = requests.Session()
                session.headers.update({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                obj = yf.Ticker(ticker, session=session)
                
                # Tentar fast_info primeiro (menos restritivo)
                try:
                    fi = obj.fast_info
                    fast = {
                        'marketCap': getattr(fi, 'market_cap', None),
                        'shares': getattr(fi, 'shares', None),
                        'currency': getattr(fi, 'currency', None),
                        'exchange': getattr(fi, 'exchange', None),
                        'fiftyDayAverage': getattr(fi, 'fifty_day_average', None),
                        'twoHundredDayAverage': getattr(fi, 'two_hundred_day_average', None),
                        'yearHigh': getattr(fi, 'year_high', None),
                        'yearLow': getattr(fi, 'year_low', None),
                    }
                except:
                    pass
                
                # Tentar info completo
                try:
                    info = obj.info
                    if info and len(info) > 5:
                        return {'info': info, 'fast': fast}
                except:
                    pass
                
                if fast:
                    return {'info': {}, 'fast': fast}
                    
                time.sleep(2)
                
            except:
                time.sleep(2)
                continue
        
        return {'info': {}, 'fast': fast}
    
    resultado = buscar_info_ticker(ticker)
    info = resultado.get('info', {})
    fast = resultado.get('fast', {})
    
    if info and len(info) > 5:
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.subheader("Dados Cadastrais")
            st.write(f"Nome: {info.get('longName', 'N/A')}")
            st.write(f"Setor: {info.get('sector', 'N/A')}")
            st.write(f"Industria: {info.get('industry', 'N/A')}")
            st.write(f"Pais: {info.get('country', 'N/A')}")
            st.write(f"Moeda: {info.get('currency', 'N/A')}")
            st.write(f"Bolsa: {info.get('exchange', 'N/A')}")
        
        with col_info2:
            st.subheader("Dados Fundamentalistas")
            st.write(f"Market Cap: {info.get('marketCap', 'N/A'):,}" if isinstance(info.get('marketCap'), (int, float)) else "Market Cap: N/A")
            st.write(f"P/L: {info.get('trailingPE', 'N/A'):.2f}" if isinstance(info.get('trailingPE'), (int, float)) else "P/L: N/A")
            st.write(f"P/VP: {info.get('priceToBook', 'N/A'):.2f}" if isinstance(info.get('priceToBook'), (int, float)) else "P/VP: N/A")
            st.write(f"Dividend Yield: {info.get('dividendYield', 0)*100:.2f}%" if isinstance(info.get('dividendYield'), (int, float)) else "Dividend Yield: N/A")
            st.write(f"ROE: {info.get('returnOnEquity', 0)*100:.2f}%" if isinstance(info.get('returnOnEquity'), (int, float)) else "ROE: N/A")
            st.write(f"Beta: {info.get('beta', 'N/A'):.2f}" if isinstance(info.get('beta'), (int, float)) else "Beta: N/A")
        
        st.divider()
        st.subheader("Descricao da Empresa")
        st.write(info.get('longBusinessSummary', 'Descricao nao disponivel.'))
    
    elif fast:
        st.subheader("Dados Disponiveis (modo reduzido)")
        
        col_fast1, col_fast2 = st.columns(2)
        
        with col_fast1:
            st.write(f"Moeda: {fast.get('currency', 'N/A')}")
            st.write(f"Bolsa: {fast.get('exchange', 'N/A')}")
            st.write(f"Market Cap: {fast.get('marketCap', 'N/A'):,.0f}" if fast.get('marketCap') else "Market Cap: N/A")
            st.write(f"Acoes em Circulacao: {fast.get('shares', 'N/A'):,.0f}" if fast.get('shares') else "Acoes: N/A")
        
        with col_fast2:
            st.write(f"Media 50 dias: R$ {fast.get('fiftyDayAverage', 'N/A'):.2f}" if fast.get('fiftyDayAverage') else "Media 50d: N/A")
            st.write(f"Media 200 dias: R$ {fast.get('twoHundredDayAverage', 'N/A'):.2f}" if fast.get('twoHundredDayAverage') else "Media 200d: N/A")
            st.write(f"Maxima 52 semanas: R$ {fast.get('yearHigh', 'N/A'):.2f}" if fast.get('yearHigh') else "Max 52s: N/A")
            st.write(f"Minima 52 semanas: R$ {fast.get('yearLow', 'N/A'):.2f}" if fast.get('yearLow') else "Min 52s: N/A")
        
        st.info("Informacoes completas indisponiveis no momento. Exibindo dados basicos.")
    
    else:
        st.warning("Nao foi possivel carregar informacoes detalhadas do ativo (limite de requisicoes).")
        if st.button("Tentar novamente"):
            st.cache_data.clear()
            st.rerun()


# ============ ESTATISTICAS DESCRITIVAS ============
elif funcao == "Estatisticas Descritivas":
    
    st.header(f"Estatisticas Descritivas: {ticker_display}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Precos de Fechamento")
        
        estatisticas_preco = {
            'Media': dados['Close'].mean(),
            'Mediana': dados['Close'].median(),
            'Desvio Padrao': dados['Close'].std(),
            'Variancia': dados['Close'].var(),
            'Minimo': dados['Close'].min(),
            'Maximo': dados['Close'].max(),
            'Amplitude': dados['Close'].max() - dados['Close'].min(),
            'Coef. Variacao (%)': (dados['Close'].std() / dados['Close'].mean()) * 100,
            'Assimetria': dados['Close'].skew(),
            'Curtose': dados['Close'].kurtosis()
        }
        
        df_preco = pd.DataFrame(estatisticas_preco.items(), columns=['Estatistica', 'Valor'])
        df_preco['Valor'] = df_preco['Valor'].apply(lambda x: f"{x:.4f}")
        st.dataframe(df_preco, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Retornos Diarios")
        
        estatisticas_retorno = {
            'Media': dados['Retorno'].mean(),
            'Mediana': dados['Retorno'].median(),
            'Desvio Padrao': dados['Retorno'].std(),
            'Variancia': dados['Retorno'].var(),
            'Minimo': dados['Retorno'].min(),
            'Maximo': dados['Retorno'].max(),
            'Amplitude': dados['Retorno'].max() - dados['Retorno'].min(),
            'Coef. Variacao': dados['Retorno'].std() / abs(dados['Retorno'].mean()) if dados['Retorno'].mean() != 0 else np.nan,
            'Assimetria': dados['Retorno'].skew(),
            'Curtose': dados['Retorno'].kurtosis()
        }
        
        df_retorno = pd.DataFrame(estatisticas_retorno.items(), columns=['Estatistica', 'Valor'])
        df_retorno['Valor'] = df_retorno['Valor'].apply(lambda x: f"{x:.6f}")
        st.dataframe(df_retorno, use_container_width=True, hide_index=True)
    
    st.divider()
    
    st.subheader("Percentis dos Retornos")
    percentis = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    valores_percentis = [np.percentile(dados['Retorno'], p) for p in percentis]
    df_percentis = pd.DataFrame({'Percentil': [f"{p}%" for p in percentis], 'Valor': valores_percentis})
    df_percentis['Valor'] = df_percentis['Valor'].apply(lambda x: f"{x:.6f}")
    st.dataframe(df_percentis.T, use_container_width=True)
    
    st.divider()
    
    st.subheader("Testes de Normalidade")
    
    # Teste Jarque-Bera
    jb_stat, jb_pvalue = stats.jarque_bera(dados['Retorno'])
    
    # Teste Shapiro-Wilk (limitado a 5000 obs)
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


# ============ ANALISE DE RETORNOS ============
elif funcao == "Analise de Retornos":
    
    st.header(f"Analise de Retornos: {ticker_display}")
    
    st.subheader("Retornos Anualizados")
    
    dias_uteis = 252
    retorno_medio_diario = dados['Retorno'].mean()
    retorno_anualizado = retorno_medio_diario * dias_uteis
    volatilidade_diaria = dados['Retorno'].std()
    volatilidade_anualizada = volatilidade_diaria * np.sqrt(dias_uteis)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Retorno Medio Diario", f"{retorno_medio_diario*100:.4f}%")
    col2.metric("Retorno Anualizado", f"{retorno_anualizado*100:.2f}%")
    col3.metric("Volatilidade Diaria", f"{volatilidade_diaria*100:.4f}%")
    col4.metric("Volatilidade Anualizada", f"{volatilidade_anualizada*100:.2f}%")
    
    st.divider()
    
    st.subheader("Retornos Acumulados por Periodo")
    
    # Retorno acumulado total
    retorno_total = (dados['Close'].iloc[-1] / dados['Close'].iloc[0]) - 1
    
    # Retornos por subperiodos
    retornos_mensais = dados['Close'].resample('M').last().pct_change().dropna()
    retornos_anuais = dados['Close'].resample('Y').last().pct_change().dropna()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"Retorno Total no Periodo: {retorno_total*100:.2f}%")
        st.write(f"Numero de Observacoes: {len(dados)}")
        st.write(f"Dias com Retorno Positivo: {(dados['Retorno'] > 0).sum()} ({(dados['Retorno'] > 0).mean()*100:.1f}%)")
        st.write(f"Dias com Retorno Negativo: {(dados['Retorno'] < 0).sum()} ({(dados['Retorno'] < 0).mean()*100:.1f}%)")
    
    with col2:
        st.write(f"Maior Retorno Diario: {dados['Retorno'].max()*100:.2f}%")
        st.write(f"Menor Retorno Diario: {dados['Retorno'].min()*100:.2f}%")
        st.write(f"Retorno Medio Mensal: {retornos_mensais.mean()*100:.2f}%" if len(retornos_mensais) > 0 else "Retorno Medio Mensal: N/A")
        st.write(f"Volatilidade Mensal: {retornos_mensais.std()*100:.2f}%" if len(retornos_mensais) > 0 else "Volatilidade Mensal: N/A")
    
    st.divider()
    
    if len(retornos_mensais) > 0:
        st.subheader("Retornos Mensais")
        df_mensais = pd.DataFrame({'Mes': retornos_mensais.index.strftime('%m-%Y'), 'Retorno (%)': retornos_mensais.values * 100})
        st.dataframe(df_mensais, use_container_width=True, hide_index=True)


# ============ METRICAS DE RISCO ============
elif funcao == "Metricas de Risco":
    
    st.header(f"Metricas de Risco: {ticker_display}")
    
    st.sidebar.divider()
    st.sidebar.subheader("Parametros de Risco")
    nivel_confianca = st.sidebar.slider("Nivel de Confianca VaR (%)", 90, 99, 95)
    taxa_livre_risco = st.sidebar.number_input("Taxa Livre de Risco (% a.a.)", value=10.0, step=0.5)
    
    dias_uteis = 252
    retorno_anualizado = dados['Retorno'].mean() * dias_uteis
    volatilidade_anualizada = dados['Retorno'].std() * np.sqrt(dias_uteis)
    rf_diario = (1 + taxa_livre_risco/100) ** (1/dias_uteis) - 1
    rf_anual = taxa_livre_risco / 100
    
    # VaR Parametrico
    z_score = stats.norm.ppf(1 - nivel_confianca/100)
    var_parametrico = dados['Retorno'].mean() + z_score * dados['Retorno'].std()
    
    # VaR Historico
    var_historico = np.percentile(dados['Retorno'], 100 - nivel_confianca)
    
    # CVaR (Expected Shortfall)
    cvar = dados['Retorno'][dados['Retorno'] <= var_historico].mean()
    
    # Sharpe Ratio
    sharpe = (retorno_anualizado - rf_anual) / volatilidade_anualizada
    
    # Sortino Ratio
    retornos_negativos = dados['Retorno'][dados['Retorno'] < 0]
    downside_deviation = retornos_negativos.std() * np.sqrt(dias_uteis)
    sortino = (retorno_anualizado - rf_anual) / downside_deviation if downside_deviation > 0 else np.nan
    
    # Maximum Drawdown
    preco_acumulado = (1 + dados['Retorno']).cumprod()
    pico = preco_acumulado.expanding(min_periods=1).max()
    drawdown = (preco_acumulado - pico) / pico
    max_drawdown = drawdown.min()
    
    # Calmar Ratio
    calmar = retorno_anualizado / abs(max_drawdown) if max_drawdown != 0 else np.nan
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Value at Risk")
        st.metric(f"VaR Parametrico ({nivel_confianca}%)", f"{var_parametrico*100:.4f}%")
        st.metric(f"VaR Historico ({nivel_confianca}%)", f"{var_historico*100:.4f}%")
        st.metric(f"CVaR / Expected Shortfall", f"{cvar*100:.4f}%")
    
    with col2:
        st.subheader("Indices de Performance")
        st.metric("Sharpe Ratio", f"{sharpe:.4f}")
        st.metric("Sortino Ratio", f"{sortino:.4f}")
        st.metric("Calmar Ratio", f"{calmar:.4f}")
    
    with col3:
        st.subheader("Drawdown")
        st.metric("Maximum Drawdown", f"{max_drawdown*100:.2f}%")
        st.metric("Volatilidade Anualizada", f"{volatilidade_anualizada*100:.2f}%")
        st.metric("Downside Deviation", f"{downside_deviation*100:.2f}%")
    
    st.divider()
    
    st.subheader("Interpretacao")
    st.write(f"Com {nivel_confianca}% de confianca, a perda maxima esperada em um dia e de {abs(var_historico)*100:.4f}% (VaR Historico). Caso essa perda seja excedida, a perda media esperada e de {abs(cvar)*100:.4f}% (CVaR).")
    st.write(f"O Sharpe Ratio de {sharpe:.4f} indica {'retorno ajustado ao risco positivo' if sharpe > 0 else 'retorno ajustado ao risco negativo'} considerando taxa livre de risco de {taxa_livre_risco}% a.a.")


# ============ CORRELACAO ENTRE ATIVOS ============
elif funcao == "Correlacao entre Ativos":
    
    st.header("Analise de Correlacao")
    
    st.sidebar.divider()
    st.sidebar.subheader("Ativos para Comparacao")
    
    tickers_comparacao = st.sidebar.text_area("Tickers (um por linha)", value="VALE3\nITUB4\nBBDC4\nABEV3")
    lista_tickers = [t.strip() if t.strip().endswith(".SA") else f"{t.strip()}.SA" for t in tickers_comparacao.split('\n') if t.strip()]
    lista_tickers = [ticker] + lista_tickers
    
    # Download de todos os ativos com cache
    @st.cache_data(ttl=300)
    def baixar_multiplos(tickers_tuple, data_inicio_str, data_fim_str):
        return yf.download(list(tickers_tuple), start=data_inicio_str, end=data_fim_str, progress=False)['Close']
    
    dados_multi = baixar_multiplos(tuple(lista_tickers), str(data_inicio), str(data_fim))
    
    if dados_multi.empty:
        st.error("Nao foi possivel baixar os dados dos ativos.")
        st.stop()
    
    # Renomear colunas removendo .SA
    dados_multi.columns = [col.replace(".SA", "") for col in dados_multi.columns]
    
    # Retornos
    retornos_multi = dados_multi.pct_change().dropna()
    
    # Matriz de correlacao
    matriz_corr = retornos_multi.corr()
    
    st.subheader("Matriz de Correlacao dos Retornos")
    st.dataframe(matriz_corr.style.format("{:.4f}").background_gradient(cmap='RdYlGn', vmin=-1, vmax=1), use_container_width=True)
    
    st.divider()
    
    # Matriz de covariancia
    matriz_cov = retornos_multi.cov()
    
    st.subheader("Matriz de Covariancia dos Retornos")
    st.dataframe(matriz_cov.style.format("{:.6f}"), use_container_width=True)
    
    st.divider()
    
    st.subheader("Estatisticas Comparativas")
    
    estatisticas_comparativas = pd.DataFrame({
        'Retorno Medio Diario (%)': retornos_multi.mean() * 100,
        'Volatilidade Diaria (%)': retornos_multi.std() * 100,
        'Retorno Anualizado (%)': retornos_multi.mean() * 252 * 100,
        'Volatilidade Anualizada (%)': retornos_multi.std() * np.sqrt(252) * 100,
        'Sharpe (rf=10%)': (retornos_multi.mean() * 252 - 0.10) / (retornos_multi.std() * np.sqrt(252))
    })
    
    st.dataframe(estatisticas_comparativas.style.format("{:.4f}"), use_container_width=True)


# ============ DOWNLOAD DE DADOS ============
elif funcao == "Download de Dados":
    
    st.header("Download de Dados")
    
    st.sidebar.divider()
    st.sidebar.subheader("Opcoes de Download")
    
    tipo_dado = st.sidebar.selectbox("Tipo de Dado", ["Precos OHLCV", "Retornos", "Ambos"])
    formato = st.sidebar.selectbox("Formato", ["CSV", "Excel"])
    
    # Preparar dados para download
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
    
    # Botao de download
    if formato == "CSV":
        csv = df_download.to_csv()
        st.download_button(
            label="Baixar CSV",
            data=csv,
            file_name=f"{ticker_display}_{tipo_dado.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )
    else:
        buffer = io.BytesIO()
        df_download.to_excel(buffer, engine='openpyxl')
        st.download_button(
            label="Baixar Excel",
            data=buffer.getvalue(),
            file_name=f"{ticker_display}_{tipo_dado.lower().replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    st.divider()
    
    st.subheader("Estatisticas Rapidas")
    st.dataframe(df_download.describe(), use_container_width=True)


# Rodape
st.sidebar.divider()
st.sidebar.caption("Harpa Quant")
st.sidebar.caption(f"Dados: Yahoo Finance")
st.sidebar.caption(f"Atualizado: {datetime.now().strftime('%d-%m-%Y %H:%M')}")