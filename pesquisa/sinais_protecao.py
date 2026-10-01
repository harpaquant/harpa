# Estudo: os sinais de regime e de PCR antecipam quedas que acionariam a put?
# Evento: queda de 10% ou mais do Ibovespa em ate 42 pregoes.
# Roda da raiz: python pesquisa\sinais_protecao.py
import sys
sys.path.append(".")

import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hmmlearn import hmm

import config

QUEDA = 0.10          # tamanho da queda que aciona a put 15% OTM
HORIZONTE = 42        # pregoes, prazo da put bimestral
JANELA_VOL = 30       # janela do desvio movel usado no HMM
REAJUSTE = 21         # a cada quantos pregoes o HMM e reestimado
INICIO = "2008-01-01"
PERCENTIL_PCR = 80    # sinal de PCR acima deste percentil da historia recente
JANELA_PCR = 60       # registros usados na referencia do percentil

# ---------------------------------------------------------------- dados
con = sqlite3.connect(config.DB_COTACOES)
indice = pd.read_sql_query(
    "SELECT data, fechamento FROM cotacoes WHERE ticker = '^BVSP' "
    "AND data >= ? ORDER BY data", con, params=(INICIO,))
con.close()

indice["data"] = pd.to_datetime(indice["data"])
indice = indice.set_index("data")["fechamento"].dropna()
print("Ibovespa: %d pregoes, de %s a %s"
      % (len(indice), indice.index[0].date(), indice.index[-1].date()))

# ---------------------------------------------------------------- evento
minimo_futuro = indice[::-1].rolling(HORIZONTE, min_periods=1).min()[::-1].shift(-1)
queda_futura = minimo_futuro / indice - 1
evento = queda_futura <= -QUEDA
print("Evento (queda de %.0f%% em ate %d pregoes): %.1f%% das datas"
      % (QUEDA * 100, HORIZONTE, evento.mean() * 100))

# ------------------------------------------------- sinal de regime, expansivo
retornos = np.log(indice / indice.shift(1))
vol = retornos.rolling(JANELA_VOL).std().dropna()

sinal_markov = pd.Series(False, index=vol.index)
inicio_estimacao = 500   # minimo de observacoes antes da primeira estimacao
posicao = inicio_estimacao
modelo = None
estado_alto = None

print("Estimando o HMM de forma expansiva, sem usar informacao futura...")
while posicao < len(vol):
    if (posicao - inicio_estimacao) % REAJUSTE == 0 or modelo is None:
        amostra = vol.iloc[:posicao].values.reshape(-1, 1)
        modelo = hmm.GaussianHMM(n_components=2, covariance_type="full",
                                 n_iter=200, random_state=42)
        try:
            modelo.fit(amostra)
            estado_alto = int(np.argmax(modelo.means_.flatten()))
        except Exception:
            posicao += 1
            continue
    bloco = vol.iloc[:posicao + 1].values.reshape(-1, 1)
    estado = modelo.predict(bloco)[-1]
    sinal_markov.iloc[posicao] = (estado == estado_alto)
    posicao += 1

sinal_markov = sinal_markov.iloc[inicio_estimacao:]
print("Sinal de regime ativo em %.1f%% das datas" % (sinal_markov.mean() * 100))

# ---------------------------------------------------------------- sinal de PCR
con = sqlite3.connect(config.DB_PCR)
try:
    pcr = pd.read_sql_query(
        "SELECT data, pcr_negocios FROM pcr WHERE subjacente = 'BOVA11' ORDER BY data", con)
except Exception:
    pcr = pd.DataFrame()
con.close()

tem_pcr = len(pcr) >= JANELA_PCR + 20
if tem_pcr:
    pcr["data"] = pd.to_datetime(pcr["data"])
    pcr = pcr.set_index("data")["pcr_negocios"].dropna()
    referencia = pcr.rolling(JANELA_PCR, min_periods=20).quantile(PERCENTIL_PCR / 100)
    sinal_pcr = (pcr > referencia).reindex(indice.index).fillna(False)
    print("Sinal de PCR ativo em %.1f%% das datas, com %d registros"
          % (sinal_pcr.mean() * 100, len(pcr)))
else:
    sinal_pcr = pd.Series(False, index=indice.index)
    print("Historico de PCR insuficiente (%d registros). Essa parte do estudo fica "
          "em aberto ate acumular pelo menos %d." % (len(pcr), JANELA_PCR + 20))

# ---------------------------------------------------------------- avaliacao
print()
print("%-22s %10s %10s %10s %10s" % ("sinal", "frequencia", "P(evento)",
                                     "P(ev|sinal)", "ganho"))
print("-" * 66)

base = evento.reindex(sinal_markov.index).dropna()
combinados = {"regime alto": sinal_markov.reindex(base.index).fillna(False)}
if tem_pcr:
    combinados["pcr alto"] = sinal_pcr.reindex(base.index).fillna(False)
    combinados["regime e pcr"] = (sinal_markov.reindex(base.index).fillna(False)
                                  & sinal_pcr.reindex(base.index).fillna(False))
    combinados["regime ou pcr"] = (sinal_markov.reindex(base.index).fillna(False)
                                   | sinal_pcr.reindex(base.index).fillna(False))

p_evento = base.mean()
for nome in combinados:
    s = combinados[nome]
    if s.sum() == 0:
        print("%-22s %9.1f%% %9.1f%% %10s %10s" % (nome, 0, p_evento * 100, "-", "-"))
        continue
    p_condicional = base[s].mean()
    print("%-22s %9.1f%% %9.1f%% %10.1f%% %9.2fx"
          % (nome, s.mean() * 100, p_evento * 100, p_condicional * 100,
             p_condicional / p_evento if p_evento > 0 else np.nan))

# -------------------------------------------- custo dos falsos positivos
s = combinados["regime alto"]
entradas = s & (~s.shift(1).fillna(False))
acertos = 0
total = 0
antecedencias = []
for data in entradas[entradas].index:
    total += 1
    futuro = indice.loc[data:].iloc[:HORIZONTE + 1]
    if len(futuro) < 5:
        continue
    pior = futuro.min() / futuro.iloc[0] - 1
    if pior <= -QUEDA:
        acertos += 1
        antecedencias.append(int(np.argmin(futuro.values)))

print()
print("Entradas no regime de alta volatilidade: %d" % total)
if total > 0:
    print("  seguidas de queda de %.0f%% em %d pregoes: %d (%.0f%%)"
          % (QUEDA * 100, HORIZONTE, acertos, 100 * acertos / total))
    print("  sem queda, prêmio gasto a toa: %d (%.0f%%)"
          % (total - acertos, 100 * (total - acertos) / total))
if antecedencias:
    print("  pregoes entre o sinal e o fundo: mediana de %.0f, minimo de %d"
          % (np.median(antecedencias), min(antecedencias)))

# ---------------------------------------------------------------- grafico
figura, eixo = plt.subplots(figsize=(11, 4.5), dpi=150)
eixo.plot(indice.index, indice.values, color=(0, 0.13, 0.31), lw=1.0, label="Ibovespa")
for data in sinal_markov[sinal_markov].index:
    eixo.axvspan(data, data, color=(0.29, 0.5, 0.71), alpha=0.08)
eixo.set_yscale("log")
eixo.set_ylabel("Ibovespa (escala log)", fontsize=9)
eixo.legend(frameon=False, fontsize=9)
eixo.set_title("Regime de alta volatilidade estimado sem informacao futura",
               fontsize=10, loc="left")
for lado in ["top", "right"]:
    eixo.spines[lado].set_visible(False)
figura.tight_layout()
saida = config.PASTA_QUANT / "sinais_protecao.png"
figura.savefig(saida)
print()
print("Grafico salvo em %s" % saida)