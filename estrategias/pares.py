# Varredura de pares cointegrados: persistencia e consulta.
# A execucao e manual. Este modulo grava o que foi visto e o que esta aberto.
import sqlite3
from datetime import date
import numpy as np
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

def abrir_posicoes():
    con = sqlite3.connect(config.DB_PARES)
    con.execute("""CREATE TABLE IF NOT EXISTS posicoes (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       data_entrada TEXT NOT NULL,
                       acao_compra TEXT NOT NULL,
                       acao_vende TEXT NOT NULL,
                       acao1 TEXT NOT NULL,
                       acao2 TEXT NOT NULL,
                       alfa REAL, beta REAL,
                       preco1_entrada REAL, preco2_entrada REAL,
                       desvio_entrada REAL, desvio_pad REAL, z_entrada REAL,
                       meia_vida REAL,
                       qtde_compra REAL, qtde_vende REAL,
                       preco_compra_exec REAL, preco_vende_exec REAL,
                       situacao TEXT NOT NULL,
                       data_saida TEXT,
                       preco_compra_saida REAL, preco_vende_saida REAL,
                       motivo_saida TEXT,
                       observacao TEXT)""")
    con.commit()
    return con


def registrar_entrada(linha, qtde_compra=None, qtde_vende=None,
                      preco_compra_exec=None, preco_vende_exec=None,
                      dia=None, observacao=None):
    # linha: uma linha da varredura, com as colunas originais do painel.
    # Os precos da varredura ficam em preco1_entrada e preco2_entrada, coerentes
    # com o alfa, o beta e o desvio. Os precos executados ficam separados.
    dia = dia or date.today().isoformat()
    con = abrir_posicoes()
    cursor = con.execute(
        """INSERT INTO posicoes (data_entrada, acao_compra, acao_vende, acao1, acao2,
                                 alfa, beta, preco1_entrada, preco2_entrada,
                                 desvio_entrada, desvio_pad, z_entrada, meia_vida,
                                 qtde_compra, qtde_vende,
                                 preco_compra_exec, preco_vende_exec,
                                 situacao, observacao)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'aberta', ?)""",
        (dia, linha["acaoCompra"], linha["acaoVende"], linha["Acao1"], linha["Acao2"],
         float(linha["Alfa"]), float(linha["Beta"]),
         float(linha["Preco1"]), float(linha["Preco2"]),
         float(linha["DesvioAb"]), float(linha["DesvioP"]),
         float(linha["DesvioAb"]) / float(linha["DesvioP"]),
         float(linha["OU"]), qtde_compra, qtde_vende,
         preco_compra_exec, preco_vende_exec, observacao))
    con.commit()
    identificador = cursor.lastrowid
    con.close()
    return identificador


def registrar_saida(identificador, preco_compra_saida, preco_vende_saida,
                    motivo, dia=None):
    dia = dia or date.today().isoformat()
    con = abrir_posicoes()
    con.execute("""UPDATE posicoes SET situacao = 'fechada', data_saida = ?,
                                       preco_compra_saida = ?, preco_vende_saida = ?,
                                       motivo_saida = ?
                   WHERE id = ?""",
                (dia, preco_compra_saida, preco_vende_saida, motivo, identificador))
    con.commit()
    con.close()


def ler_posicoes(situacao="aberta"):
    con = abrir_posicoes()
    if situacao is None:
        tabela = pd.read_sql_query("SELECT * FROM posicoes ORDER BY data_entrada DESC", con)
    else:
        tabela = pd.read_sql_query(
            "SELECT * FROM posicoes WHERE situacao = ? ORDER BY data_entrada DESC",
            con, params=(situacao,))
    con.close()
    return tabela


def resultado_posicao(posicao, preco_compra_atual, preco_vende_atual):
    # Resultado financeiro estimado da posicao, em reais, com os precos informados.
    if posicao["qtde_compra"] is None or posicao["qtde_vende"] is None:
        return None
    if posicao["preco_compra_exec"] is None or posicao["preco_vende_exec"] is None:
        return None
    ganho_compra = (preco_compra_atual - posicao["preco_compra_exec"]) * posicao["qtde_compra"]
    ganho_venda = (posicao["preco_vende_exec"] - preco_vende_atual) * posicao["qtde_vende"]
    return ganho_compra + ganho_venda

def avaliar_abertas(precos_atuais, dia=None):
    # precos_atuais: dicionario {ticker: preco}, tickers sem o sufixo .SA
    abertas = ler_posicoes("aberta")
    if abertas.empty:
        return abertas

    dia = dia or date.today().isoformat()
    hoje = pd.to_datetime(dia)
    linhas = []

    for _, p in abertas.iterrows():
        preco1 = precos_atuais.get(p["acao1"])
        preco2 = precos_atuais.get(p["acao2"])
        if preco1 is None or preco2 is None:
            continue

        desvio = preco1 - p["alfa"] - p["beta"] * preco2
        z = desvio / p["desvio_pad"]
        dias_aberto = int(np.busday_count(pd.to_datetime(p["data_entrada"]).date(), hoje.date()))
        vidas = dias_aberto / p["meia_vida"] if p["meia_vida"] else None

        avisos = []
        if abs(z) <= 0.5:
            avisos.append("desvio fechou")
        if abs(z) > abs(p["z_entrada"]) * 1.5:
            avisos.append("desvio ampliou")
        if vidas is not None and vidas > 2:
            avisos.append("tempo acima de duas meias-vidas")

        resultado = None
        if (p["qtde_compra"] is not None and p["qtde_vende"] is not None
                and p["preco_compra_exec"] is not None and p["preco_vende_exec"] is not None):
            preco_compra_atual = precos_atuais.get(p["acao_compra"])
            preco_vende_atual = precos_atuais.get(p["acao_vende"])
            if preco_compra_atual is not None and preco_vende_atual is not None:
                resultado = ((preco_compra_atual - p["preco_compra_exec"]) * p["qtde_compra"]
                             + (p["preco_vende_exec"] - preco_vende_atual) * p["qtde_vende"])

        linhas.append({"id": p["id"],
                       "entrada": p["data_entrada"],
                       "compra": p["acao_compra"],
                       "vende": p["acao_vende"],
                       "z_entrada": p["z_entrada"],
                       "z_atual": z,
                       "meia_vida": p["meia_vida"],
                       "dias_aberto": dias_aberto,
                       "meias_vidas": vidas,
                       "resultado": resultado,
                       "avisos": ", ".join(avisos) if avisos else ""})

    return pd.DataFrame(linhas)