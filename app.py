import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import datetime
from datetime import datetime, timedelta, time
import math
from scipy import stats
from scipy.stats import norm
from scipy.optimize import newton

st.title("Harpa Quant")
st.markdown("""##### Ferramentas quantitativas para o investidor prospectivo.""")
st.markdown("""Escolha à esquerda a ferramenta.""")

st.markdown("[![Twitter](https://img.shields.io/badge/Twitter-%231DA1F2.svg?style=for-the-badge&logo=Twitter&logoColor=white)](https://twitter.com/harpaquant)")
st.markdown("[![Instagram](https://img.shields.io/badge/Instagram-%23E4405F.svg?style=for-the-badge&logo=Instagram&logoColor=white)](https://www.instagram.com/harpaquant)")

st.markdown('---')

st.sidebar.markdown("""
    Atualmente, disponibilizando ferramentas para o mercado de derivativos. 
    Acesse nossa [Comunidade no Discord](https://discord.gg/MaF7wZDQvZ) para participar de discussões no tema. Email: harpaquant@gmail.com
    """)

st.sidebar.markdown('---')

selected_calculator = st.sidebar.selectbox(
    "Selecione a ferramenta:",
    ("Calculadoras Black-Scholes-Merton", "Calculadora de Gregas de Opções", "Cones de Volatilidade")
)

st.sidebar.markdown('---')
st.sidebar.subheader('Ferramentas disponíveis')
st.sidebar.write('Calculadoras Black-Scholes-Merton \n\n- Preço da opção\n\n- Volatilidade implícita')
st.sidebar.write('Calculadora de Gregas de Opções')
st.sidebar.write('Cones de Volatilidade')

###########################
### BLACK-SCHOLES

if selected_calculator == "Calculadoras Black-Scholes-Merton":
    
    ### Calculadora Black-Scholes-Merton - Preço
    def black_scholes_call_put(S, K, T, r, sigma, option_type):
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        if option_type == 'call':
            option_price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        elif option_type == 'put':
            option_price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        
        return option_price

    # Função de distribuição cumulativa normal padrão (CDF)
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    # Título do aplicativo
    st.subheader('Calculadora Black-Scholes-Merton - Preço da opção')
    st.markdown("""
    Black, Scholes e Merton revolucionaram a análise de opções, fornecendo um arcabouço robusto 
                para entender e precificar riscos financeiros. Sua influência perdura, moldando 
                a maneira como investidores e instituições lidam com a complexidade dos mercados. 
    """)
    # Organizando os campos de entrada em duas colunas
    col1, col2 = st.columns(2)

    # Entrada dos valores dos parâmetros
    with col1:
        S = st.number_input('Preço do ativo subjacente (S)', min_value=0.0)
        K = st.number_input('Preço de exercício da opção (K)', min_value=0.0)
        T = st.number_input('Vencimento da opção (T), como fração do ano', min_value=0.0)
    with col2:
        r = st.number_input('Taxa de juros anual (r), como fração', min_value=0.0)
        sigma = st.number_input('Volatilidade (Sigma), como fração', min_value=0.0)
        option_type = st.selectbox('Tipo de opção', ['call', 'put'])

    # Botão para calcular o preço da opção
    if st.button('Calcular preço da opção'):
        option_price = black_scholes_call_put(S, K, T, r, sigma, option_type)
        st.write(f'O preço da {option_type} é: ${round(option_price,2)}')

    st.markdown('---')

    # Função para calcular o preço da opção Black-Scholes-Merton
    def black_scholes_call_put(S, K, T, r, sigma, option_type):
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        if option_type == 'call':
            option_price = S * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(d2)
        elif option_type == 'put':
            option_price = K * math.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        
        return option_price

    # Função para calcular a volatilidade implícita
    def implied_volatility(option_price, S, K, T, r, option_type):
        # Função para encontrar a volatilidade implícita usando o método de Newton-Raphson
        def func(sigma, option_price, S, K, T, r, option_type):
            return black_scholes_call_put(S, K, T, r, sigma, option_type) - option_price

        # Chamada para o método de Newton-Raphson para encontrar a volatilidade implícita
        sigma = newton(func, x0=0.2, args=(option_price, S, K, T, r, option_type))
        return sigma

    # Título do aplicativo
    st.subheader('Calculadora Black-Scholes-Merton - Volatilidade implícita')
    st.markdown("""
    A volatilidade implícita, elemento-chave no modelo Black-Scholes, reflete as expectativas 
                do mercado sobre a flutuação futura dos preços de ativos. Sua análise é crucial 
                para precificar opções e compreender as percepções dos investidores em relação 
                ao risco. 
    """)
    # Organizando os campos de entrada em duas colunas
    col1, col2 = st.columns(2)

    # Entrada dos valores dos parâmetros
    with col1:
        Sv = st.number_input('Preço do ativo subjacente (S)', min_value=0.0, key='Sv')
        Kv = st.number_input('Preço de exercício da opção (K)', min_value=0.0, key='Kv')
        Tv = st.number_input('Vencimento da opção (T), como fração do ano', min_value=0.0, key='Tv')
    with col2:
        rv = st.number_input('Taxa de juros anual (r), como fração', min_value=0.0, key='rv')
        option_price = st.number_input('Preço da opção', min_value=0.0, key='option_price')
        option_type = st.selectbox('Tipo de opção', ['call', 'put'], key='option_type')

    # Botão para calcular a volatilidade implícita
    if st.button('Calcular volatilidade implícita'):
        sigma_impl = implied_volatility(option_price, Sv, Kv, Tv, rv, option_type)
        st.write(f'A volatilidade implícita é {round(sigma_impl,2)}')

###########################
### GREGAS

elif selected_calculator == "Calculadora de Gregas de Opções":
    # Calculadora de gregas
    # Função para calcular a grega Delta
    def delta(S, K, T, r, sigma, option_type):
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

        if option_type == 'call':
            delta = norm.cdf(d1)
        elif option_type == 'put':
            delta = norm.cdf(d1) - 1
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        
        return delta

    # Função para calcular a grega Gamma
    def gamma(S, K, T, r, sigma):
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        return gamma

    # Função para calcular a grega Vega
    def vega(S, K, T, r, sigma):
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        vega = S * norm.pdf(d1) * math.sqrt(T)
        return vega

    # Função para calcular a grega Theta
    def theta(S, K, T, r, sigma, option_type):
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_type == 'call':
            theta = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'put':
            theta = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        
        return theta

    # Função para calcular a grega Rho
    def rho(S, K, T, r, sigma, option_type):
        d2 = (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

        if option_type == 'call':
            rho = K * T * math.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'put':
            rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        
        return rho

    # Título do aplicativo
    st.subheader('Calculadora de Gregas de Opções')
    st.markdown("""
        As 'Gregas', na análise de derivativos, referem-se a medidas de sensibilidade de preço 
                de opções a diferentes variáveis, como volatilidade, tempo e  movimento do preço 
                do ativo subjacente. Delta, Gamma, Vega, Theta e Rho são alguns exemplos essenciais. 
        """)
    # Organizando os campos de entrada em duas colunas
    col1, col2 = st.columns(2)

    # Entrada dos valores dos parâmetros
    with col1:
        S = st.number_input('Preço do ativo subjacente (S)', min_value=0.0, step=0.01, key='S_gregas')
        K = st.number_input('Preço de exercício da opção (K)', min_value=0.0, step=0.01, key='K_gregas')
        T = st.number_input('Tempo até o vencimento (T) em anos', min_value=0.0, step=0.01, key='T_gregas')
    with col2:
        r = st.number_input('Taxa de juros anual (r), como fração', min_value=0.0, step=0.0001, key='r_gregas')
        sigma = st.number_input('Volatilidade (Sigma), como fração', min_value=0.0, step=0.0001, key='sigma_gregas')
        option_type = st.selectbox('Tipo de opção', ['call', 'put'], key='option_type_gregas')

    # Botão para calcular as "gregas"
    if st.button('Calcular Gregas'):
        st.write("Delta:", round(delta(S, K, T, r, sigma, option_type),6))
        st.write("Gamma:", round(gamma(S, K, T, r, sigma),6))
        st.write("Vega:", round(vega(S, K, T, r, sigma),6))
        st.write("Theta:", round(theta(S, K, T, r, sigma, option_type),6))
        st.write("Rho:", round(rho(S, K, T, r, sigma, option_type),6))

###########################
### GREGAS

elif selected_calculator == "Cones de Volatilidade":
    # Cones de volatilidade para diferentes ativos
    st.subheader('Cones de Volatilidade')
    st.markdown("""
        A parte mais difícil da negociação de opções é determinar se elas estão baratas ou caras. 
                Ao comprar ou vender uma opção, você está exposto à volatilidade do ativo subjacente. 
                Por isso, é importante comparar a volatilidade aos seus níveis recentes. 
                Os cones de volatilidade podem ajudar nessa análise. Veja abaixo gráficos do cone de
                volatilidade para diferentes ativos subjacentes.
        """)

    acaocone = st.radio('Escolha o ativo subjacente', ['ABEV3','BBDC4','BOVA11','PETR4','VALE3'])    
    windows = [15, 30, 45, 60, 75, 90, 105, 120]
    quantiles = [0.25, 0.75]
    min_ = []
    max_ = []
    median = []
    top_q = []
    bottom_q = []
    realized = []
    start = "2006-01-02"
    def realized_vol(price_data, window=30):
        log_return = (price_data["Close"] / price_data["Close"].shift(1)).apply(np.log)
        return log_return.rolling(window=window, center=False).std() * math.sqrt(252)

    if acaocone == 'ABEV3':
        data = yf.download('ABEV3.SA', start=start)
    if acaocone == 'BBDC4':
        data = yf.download('BBDC4.SA', start=start)
    if acaocone == 'BOVA11':
        data = yf.download('BOVA11.SA', start=start)
    if acaocone == 'PETR4':
        data = yf.download('PETR4.SA', start=start)
    if acaocone == 'VALE3':
        data = yf.download('VALE3.SA', start=start)

    for window in windows:
        # get a dataframe with realized volatility
        estimator = realized_vol(window=window, price_data=data)
        # append the summary stats to a list
        min_.append(estimator.min())
        max_.append(estimator.max())
        median.append(estimator.median())
        top_q.append(estimator.quantile(quantiles[1]))
        bottom_q.append(estimator.quantile(quantiles[0]))
        realized.append(estimator.iloc[-1])
    
    data = [
    go.Scatter(x=windows, y=min_, mode='markers+lines', name='Min'),
    go.Scatter(x=windows, y=max_, mode='markers+lines', name='Max'),
    go.Scatter(x=windows, y=median, mode='markers+lines', name='Mediana'),
    go.Scatter(x=windows, y=top_q, mode='markers+lines', name=f'{quantiles[1] * 100:.0f} Percentil'),
    go.Scatter(x=windows, y=bottom_q, mode='markers+lines', name=f'{quantiles[0] * 100:.0f} Percentil'),
    go.Scatter(x=windows, y=realized, mode='markers+lines', name='Realizado', marker=dict(color='yellow'))
    ]

    # Criar o layout do gráfico
    layout = go.Layout(
        title=f'Cone de Volatilidade - {acaocone}',
        xaxis=dict(title='Janelas'),
        yaxis=dict(title='Valores'),
        legend=dict(x=0.5, y=1.0, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)')
    )

    # Criar o gráfico
    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig)

#    # create the plots on the chart
#    plt.plot(windows, min_, "-o", linewidth=1, label="Min")
#    plt.plot(windows, max_, "-o", linewidth=1, label="Max")
#    plt.plot(windows, median, "-o", linewidth=1, label="Mediana")
#    plt.plot(windows, top_q, "-o", linewidth=1, label=f"{quantiles[1] * 100:.0f} Prctl")
#    plt.plot(windows, bottom_q, "-o", linewidth=1, label=f"{quantiles[0] * 100:.0f} Prctl")
#    plt.plot(windows, realized, "ro-.", linewidth=1, label="Realizado")
#    # set the x-axis labels
#    plt.xticks(windows)
#    # format the legend
#    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1), ncol=3)
#    plt.title(f'Cone de Volatilidade - {ativo_sem_extensao}')
#    plt.xlabel('Janelas')

#    ativo_sem_extensao = ativo_do_dia.rstrip('.SA')




#elif selected_calculator == "Put Call Ratio - PCR":
#    # Título do aplicativo
#    st.subheader('Put Call Ratio - PCR')
#    st.markdown("""
#        O Put Call Ratio - PCR é um indicador utilizado para avaliar o sentimento 
#                do mercado em relação às opções. Ele compara o volume de negociação 
#                de opções de venda com o volume de negociação de opções de compra, 
#                oferecendo insights sobre as expectativas dos investidores. 
#        """)
    


