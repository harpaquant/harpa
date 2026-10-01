# Varredura de pares cointegrados: persistencia e consulta.
# A execucao e manual. Este modulo grava o que foi visto e o que esta aberto.
import sqlite3
from datetime import date

import pandas as pd

import config

COLUNAS = ("data", "acao1", "acao2", "alfa", "beta", "pbeta", "ppr",
           "preco1", "preco2", "desvio_abs", "desvio_pad", "z",
           "meia_vida", "taildep_l", "taildep_u",
           "acao_compra", "acao_vende")


def abrir():
    con = sqlite3.connect(config.DB_PARES)
    con.execute("""CREATE TABLE IF NOT EXISTS varredura (
                       data TEXT NOT NULL,
                       acao1 TEXT NOT NULL,
                       acao2 TEXT NOT NULL,
                       alfa REAL, beta REAL, pbeta REAL, ppr REAL,
                       preco1 REAL, preco2 REAL,
                       desvio_abs REAL, desvio_pad REAL, z REAL,
                       meia_vida REAL,
                       taildep_l REAL, taildep_u REAL,
                       acao_compra TEXT, acao_vende TEXT,
                       PRIMARY KEY (data, acao1, acao2))""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_varredura_par ON varredura (acao1, acao2, data)")
    con.execute("""CREATE TABLE IF NOT EXISTS diagnostico (
                       data TEXT PRIMARY KEY,
                       total_pares INTEGER,
                       elegiveis INTEGER,
                       cointegrados INTEGER,
                       validos INTEGER,
                       esperado_acaso REAL,
                       janela_dias INTEGER)""")
    con.commit()
    return con


def gravar(tabela, diagnostico, dia=None):
    if tabela.empty:
        return 0
    dia = dia or date.today().isoformat()

    registros = tabela.copy()
    registros["data"] = dia
    registros["z"] = registros["DesvioAb"] / registros["DesvioP"]
    registros = registros.rename(columns={
        "Acao1": "acao1", "Acao2": "acao2", "Alfa": "alfa", "Beta": "beta",
        "pBeta": "pbeta", "PPR": "ppr", "Preco1": "preco1", "Preco2": "preco2",
        "DesvioAb": "desvio_abs", "DesvioP": "desvio_pad", "OU": "meia_vida",
        "TailDep_L": "taildep_l", "TailDep_U": "taildep_u",
        "acaoCompra": "acao_compra", "acaoVende": "acao_vende"})

    for coluna in COLUNAS:
        if coluna not in registros.columns:
            registros[coluna] = None
    registros = registros[list(COLUNAS)]

    con = abrir()
    linhas = list(registros.itertuples(index=False, name=None))
    marcadores = ", ".join("?" * len(COLUNAS))
    con.executemany("INSERT OR REPLACE INTO varredura VALUES (%s)" % marcadores, linhas)

    esperado = diagnostico["total_pares"] * config.PARES_CORTE_ADF_RESIDUO
    con.execute("""INSERT OR REPLACE INTO diagnostico
                   (data, total_pares, elegiveis, cointegrados, validos,
                    esperado_acaso, janela_dias)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (dia, diagnostico["total_pares"], diagnostico["elegiveis"],
                 diagnostico["cointegrados"], diagnostico["validos"],
                 esperado, config.PARES_JANELA_DIAS))
    con.commit()
    con.close()
    return len(linhas)


def ler(dia=None, acao1=None, acao2=None, limite=2000):
    con = abrir()
    consulta = "SELECT * FROM varredura WHERE 1 = 1"
    parametros = []
    if dia is not None:
        consulta += " AND data = ?"
        parametros.append(dia)
    if acao1 is not None and acao2 is not None:
        consulta += " AND acao1 = ? AND acao2 = ?"
        parametros.extend([acao1, acao2])
    consulta += " ORDER BY data DESC, abs(z) DESC LIMIT ?"
    parametros.append(limite)
    tabela = pd.read_sql_query(consulta, con, params=parametros)
    con.close()
    return tabela


def persistencia(dia=None, dias=3):
    # quantos dos ultimos pregoes gravados cada par apareceu na varredura
    con = abrir()
    datas = pd.read_sql_query(
        "SELECT DISTINCT data FROM varredura ORDER BY data DESC LIMIT ?",
        con, params=(dias,))["data"].tolist()
    if not datas:
        con.close()
        return pd.DataFrame()
    marcadores = ", ".join("?" * len(datas))
    tabela = pd.read_sql_query(
        "SELECT acao1, acao2, COUNT(*) AS dias_na_lista "
        "FROM varredura WHERE data IN (%s) "
        "GROUP BY acao1, acao2 ORDER BY dias_na_lista DESC" % marcadores,
        con, params=datas)
    con.close()
    return tabela