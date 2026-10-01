# Seguro de carteira com puts de BOVA11.
# Piso mensal mais reforco condicionado ao regime de volatilidade.
# O modulo calcula e registra. A compra e manual.
import sqlite3
from datetime import date

import pandas as pd

import config


def abrir():
    con = sqlite3.connect(config.DB_PARES)
    con.execute("""CREATE TABLE IF NOT EXISTS protecao (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       data TEXT NOT NULL,
                       tipo TEXT NOT NULL,
                       serie TEXT,
                       vencimento TEXT,
                       strike REAL,
                       spot REAL,
                       moneyness REAL,
                       quantidade REAL,
                       premio_unitario REAL,
                       premio_total REAL,
                       pl_referencia REAL,
                       motivo TEXT,
                       observacao TEXT)""")
    con.commit()
    return con


def registrar(tipo, serie, vencimento, strike, spot, quantidade,
              premio_unitario, pl_referencia, motivo=None,
              observacao=None, dia=None):
    # tipo: 'mensal' ou 'reforco'
    dia = dia or date.today().isoformat()
    premio_total = (quantidade or 0) * (premio_unitario or 0)
    moneyness = (strike / spot - 1) if spot else None
    con = abrir()
    cursor = con.execute(
        """INSERT INTO protecao (data, tipo, serie, vencimento, strike, spot,
                                 moneyness, quantidade, premio_unitario,
                                 premio_total, pl_referencia, motivo, observacao)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (dia, tipo, serie, vencimento, strike, spot, moneyness, quantidade,
         premio_unitario, premio_total, pl_referencia, motivo, observacao))
    con.commit()
    identificador = cursor.lastrowid
    con.close()
    return identificador


def ler(ano=None):
    con = abrir()
    if ano is None:
        tabela = pd.read_sql_query("SELECT * FROM protecao ORDER BY data DESC", con)
    else:
        tabela = pd.read_sql_query(
            "SELECT * FROM protecao WHERE data LIKE ? ORDER BY data DESC",
            con, params=("%d%%" % ano,))
    con.close()
    return tabela


def orcamento(pl, ano=None):
    # Situacao do orcamento anual de premio, em reais.
    ano = ano or date.today().year
    gasto = ler(ano)

    teto_total = pl * config.PUT_ORCAMENTO_ANUAL
    teto_mensal = pl * config.PUT_ORCAMENTO_MENSAL
    teto_reserva = pl * config.PUT_RESERVA_ANUAL

    if gasto.empty:
        gasto_mensal = 0.0
        gasto_reforco = 0.0
        reforcos = 0
    else:
        gasto_mensal = gasto[gasto["tipo"] == "mensal"]["premio_total"].sum()
        gasto_reforco = gasto[gasto["tipo"] == "reforco"]["premio_total"].sum()
        reforcos = int((gasto["tipo"] == "reforco").sum())

    return {"ano": ano,
            "pl": pl,
            "teto_anual": teto_total,
            "teto_mensal": teto_mensal,
            "teto_reserva": teto_reserva,
            "gasto_mensal": gasto_mensal,
            "gasto_reforco": gasto_reforco,
            "gasto_total": gasto_mensal + gasto_reforco,
            "reserva_disponivel": max(teto_reserva - gasto_reforco, 0.0),
            "reforcos_usados": reforcos,
            "reforcos_restantes": max(config.PUT_REFORCOS_ANO - reforcos, 0)}


def sugerir(pl, spot, tipo="mensal", regime_alto=False, dia=None):
    # Devolve o que comprar: valor de premio disponivel e strike alvo.
    dia = dia or date.today().isoformat()
    situacao = orcamento(pl, int(dia[:4]))

    if tipo == "mensal":
        valor = situacao["teto_mensal"]
        moneyness = config.PUT_MONEYNESS
        liberado = True
        motivo = "piso mensal de protecao"
    else:
        valor = min(pl * config.PUT_REFORCO, situacao["reserva_disponivel"])
        moneyness = config.PUT_MONEYNESS_REFORCO
        liberado = (regime_alto
                    and situacao["reforcos_restantes"] > 0
                    and valor > 0)
        if not regime_alto:
            motivo = "sem sinal de regime alto"
        elif situacao["reforcos_restantes"] <= 0:
            motivo = "limite de reforcos do ano atingido"
        elif valor <= 0:
            motivo = "reserva do ano esgotada"
        else:
            motivo = "regime de alta volatilidade"

    return {"tipo": tipo,
            "liberado": liberado,
            "motivo": motivo,
            "premio_disponivel": valor if liberado else 0.0,
            "strike_alvo": spot * (1 - moneyness),
            "moneyness": moneyness,
            "spot": spot,
            "situacao": situacao}