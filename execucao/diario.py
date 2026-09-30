# Diario de sinais e ordens do clube. Grava tudo, executado ou nao.
import sqlite3
from datetime import datetime

import config

CAMPOS = ("momento", "estrategia", "ticker", "lado", "quantidade",
          "preco_sinal", "preco_executado", "situacao", "motivo",
          "pl", "fracao_rv", "observacao")


def abrir():
    con = sqlite3.connect(config.DB_DIARIO)
    con.execute("""CREATE TABLE IF NOT EXISTS ordens (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       momento TEXT NOT NULL,
                       estrategia TEXT NOT NULL,
                       ticker TEXT NOT NULL,
                       lado TEXT,
                       quantidade REAL,
                       preco_sinal REAL,
                       preco_executado REAL,
                       situacao TEXT NOT NULL,
                       motivo TEXT,
                       pl REAL,
                       fracao_rv REAL,
                       observacao TEXT)""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_ordens_momento ON ordens (momento)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_ordens_estrategia ON ordens (estrategia, momento)")
    con.commit()
    return con


def registrar(estrategia, ticker, lado, quantidade, preco_sinal,
              situacao, preco_executado=None, motivo=None,
              pl=None, fracao_rv=None, observacao=None):
    # situacao: 'gerado', 'enviado', 'executado', 'barrado', 'rejeitado', 'cancelado'
    con = abrir()
    con.execute("""INSERT INTO ordens (momento, estrategia, ticker, lado, quantidade,
                                       preco_sinal, preco_executado, situacao, motivo,
                                       pl, fracao_rv, observacao)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (datetime.now().isoformat(timespec="seconds"), estrategia, ticker,
                 lado, quantidade, preco_sinal, preco_executado, situacao, motivo,
                 pl, fracao_rv, observacao))
    con.commit()
    identificador = con.execute("SELECT last_insert_rowid()").fetchone()[0]
    con.close()
    return identificador


def atualizar(identificador, situacao, preco_executado=None, motivo=None):
    con = abrir()
    con.execute("""UPDATE ordens SET situacao = ?,
                                     preco_executado = COALESCE(?, preco_executado),
                                     motivo = COALESCE(?, motivo)
                   WHERE id = ?""",
                (situacao, preco_executado, motivo, identificador))
    con.commit()
    con.close()


def ler(estrategia=None, desde=None, limite=500):
    con = abrir()
    consulta = "SELECT * FROM ordens WHERE 1 = 1"
    parametros = []
    if estrategia is not None:
        consulta += " AND estrategia = ?"
        parametros.append(estrategia)
    if desde is not None:
        consulta += " AND momento >= ?"
        parametros.append(desde)
    consulta += " ORDER BY momento DESC LIMIT ?"
    parametros.append(limite)
    linhas = con.execute(consulta, parametros).fetchall()
    con.close()
    return linhas