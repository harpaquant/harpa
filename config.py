# Parametros do Clube Fortem. Backtest e producao leem daqui.
import os
from pathlib import Path

PASTA = Path(__file__).parent
PASTA_BASES = PASTA / "bases"
PASTA_QUANT = PASTA / "quant"

DB_COTACOES = PASTA_BASES / "zcotacoes.db"
DB_PCR = PASTA_BASES / "zpcr_historico.db"
DB_OPCOES = PASTA_BASES / "zopcoes.db"
DB_DIARIO = PASTA_BASES / "zdiario.db"
DB_PARES = PASTA_BASES / "zpares.db"

# --- credenciais, lidas do ambiente ---------------------------------
MT5_CONTA = os.getenv("MT5_CONTA")
MT5_SENHA = os.getenv("MT5_SENHA")

# --- enquadramento do clube -----------------------------------------
MINIMO_RENDA_VARIAVEL = 0.67   # fracao do PL
PERMITE_ALAVANCAGEM = False
SUBJACENTE_OPCOES = "BOVA11"
RV_LIQUIDA = True   # True: ponta vendida reduz a exposicao. False: soma em modulo.

# --- camada de protecao ----------------------------------------------
PUT_MONEYNESS = 0.15           # strike 15% abaixo do preco a vista
PUT_ROLAGEM_MESES = 2
PUT_ORCAMENTO_ANUAL = 0.02     # fracao do PL gasta em premio por ano

# --- camada long-short ------------------------------------------------
PARES_CAPITAL_MAXIMO = 0.25    # fracao do capital do clube
PARES_MAXIMO_SIMULTANEO = 10
PARES_JANELA_DIAS = 189
PARES_CORTE_ADF_NIVEL = 0.10
PARES_CORTE_ADF_RESIDUO = 0.01
PARES_CORTE_PBETA = 0.01

# --- gatilhos de risco -------------------------------------------------
DRAWDOWN_REVISAO = 0.15
DRAWDOWN_REDUCAO = 0.25
VAR_NIVEL = 0.95

# --- custos de transacao ------------------------------------------------
CORRETAGEM_ORDEM = 0.0         # em reais, por ordem
EMOLUMENTOS = 0.00005          # fracao do financeiro
ALUGUEL_ANUAL_PADRAO = 0.02    # taxa usada quando falta a taxa do papel
DIAS_UTEIS_ANO = 252