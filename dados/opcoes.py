# Coleta e gravacao da cadeia de opcoes.
# Duas fontes: opcoes.net.br (sem spread, roda sempre) e MT5 (com spread,
# exige terminal aberto durante o pregao). Falhas de um dia nao invalidam a serie.
import sqlite3
from datetime import date, datetime

import pandas as pd
import requests

import config

URL_BASE = "https://opcoes.net.br/listaopcoes/completa"


def abrir():
    con = sqlite3.connect(config.DB_OPCOES)
    con.execute("""CREATE TABLE IF NOT EXISTS cadeia (
                       data TEXT NOT NULL,
                       fonte TEXT NOT NULL,
                       subjacente TEXT NOT NULL,
                       serie TEXT NOT NULL,
                       vencimento TEXT,
                       tipo TEXT,
                       estilo TEXT,
                       strike REAL,
                       spot REAL,
                       ultimo REAL,
                       bid REAL,
                       ask REAL,
                       ultimo_negocio TEXT,
                       negocios REAL,
                       volume REAL,
                       formador INTEGER,
                       momento TEXT,
                       PRIMARY KEY (data, fonte, serie))""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_cadeia_serie ON cadeia (serie, data)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_cadeia_venc ON cadeia (subjacente, vencimento, data)")
    con.commit()
    return con


def gravar(linhas):
    if not linhas:
        return 0
    con = abrir()
    con.executemany(
        """INSERT OR REPLACE INTO cadeia
            (data, fonte, subjacente, serie, vencimento, tipo, estilo, strike, spot,
            ultimo, bid, ask, ultimo_negocio, negocios, volume, formador, momento)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", linhas)
    con.commit()
    con.close()
    return len(linhas)


def listar_vencimentos(subjacente):
    url = f"{URL_BASE}?idAcao={subjacente}&listarVencimentos=true&cotacoes=true"
    resposta = requests.get(url, timeout=30).json()
    return [v["value"] for v in resposta["data"]["vencimentos"]]


def coletar_web(subjacente, vencimento, dia=None):
    # Le os campos pelo nome declarado em data.columns, nao pela posicao.
    # Volatilidade implicita e gregas vem borradas para quem nao assina o site.
    url = (f"{URL_BASE}?idAcao={subjacente}&listarVencimentos=false"
           f"&cotacoes=true&vencimentos={vencimento}")
    resposta = requests.get(url, timeout=30).json()
    dados = resposta["data"]

    posicao = {}
    for coluna in dados["columns"]:
        posicao[coluna["name"]] = coluna["index"]

    def campo(linha, nome):
        indice = posicao.get(nome)
        if indice is None or indice >= len(linha):
            return None
        valor = linha[indice]
        if valor is None or valor == "":
            return None
        if isinstance(valor, str) and "volblur" in valor:
            return None
        return valor

    def numero(linha, nome):
        valor = campo(linha, nome)
        if valor is None:
            return None
        if isinstance(valor, (int, float)):
            return float(valor)
        texto = str(valor).replace(".", "").replace(",", ".").strip()
        try:
            return float(texto)
        except ValueError:
            return None

    momento = datetime.now().isoformat(timespec="seconds")
    dia_registro = dia or date.today().isoformat()
    linhas = []
    for linha in dados["cotacoesOpcoes"]:
        bruto = campo(linha, "ticker")
        if bruto is None:
            continue
        serie = str(bruto).split("_")[0]

        marcador = campo(linha, "fm")
        tem_formador = 1 if (marcador and "10004" in str(marcador)) else 0

        linhas.append((
            dia_registro, "web", subjacente, serie, vencimento,
            campo(linha, "tipo"),
            campo(linha, "mod."),
            numero(linha, "strike"),
            None,
            numero(linha, "ultimo"),
            None, None,
            campo(linha, "data/hora"),
            numero(linha, "numerodenegocios"),
            numero(linha, "volumenegociado"),
            tem_formador, momento))
    return linhas


def coletar_web_tudo(subjacente=None, quantos_vencimentos=4, dia=None):
    subjacente = subjacente or config.SUBJACENTE_OPCOES
    vencimentos = listar_vencimentos(subjacente)[:quantos_vencimentos]
    total = 0
    for vencimento in vencimentos:
        total += gravar(coletar_web(subjacente, vencimento, dia))
    return total, vencimentos


def coletar_mt5(subjacente=None, quantos_vencimentos=4, dte_minimo=5, dia=None):
    # Exige o terminal MT5 aberto e o pregao em andamento, porque bid e ask
    # so existem com o livro ativo.
    import MetaTrader5 as mt5

    subjacente = subjacente or config.SUBJACENTE_OPCOES
    raiz = subjacente[:4]

    if not mt5.initialize():
        return 0, "falha ao inicializar o MT5: %s" % str(mt5.last_error())

    mt5.symbol_select(subjacente, True)
    tick_spot = mt5.symbol_info_tick(subjacente)
    if tick_spot is None:
        mt5.shutdown()
        return 0, "sem cotacao do subjacente"
    spot = (0.5 * (tick_spot.bid + tick_spot.ask)
            if tick_spot.bid > 0 and tick_spot.ask > 0 else tick_spot.last)

    hoje = date.today()
    candidatas = []
    for simbolo in mt5.symbols_get():
        if not simbolo.name.upper().startswith(raiz.upper()):
            continue
        informacao = mt5.symbol_info(simbolo.name)
        if informacao is None:
            continue
        unix = getattr(informacao, "expiration_time", 0)
        strike = getattr(informacao, "option_strike", 0.0)
        if not unix or unix <= 86400 or not strike:
            continue
        try:
            vencimento = datetime.fromtimestamp(unix).date()
        except (OSError, OverflowError, ValueError):
            continue
        if (vencimento - hoje).days < dte_minimo:
            continue
        tipo = "CALL" if getattr(informacao, "option_right", 0) == 0 else "PUT"
        candidatas.append((simbolo.name, strike, vencimento, tipo))

    vencimentos = sorted(set(c[2] for c in candidatas))[:quantos_vencimentos]
    candidatas = [c for c in candidatas if c[2] in vencimentos]

    for nome, _, _, _ in candidatas:
        mt5.symbol_select(nome, True)

    import time
    time.sleep(3)

    momento = datetime.now().isoformat(timespec="seconds")
    dia_registro = dia or hoje.isoformat()
    linhas = []
    for nome, strike, vencimento, tipo in candidatas:
        tick = mt5.symbol_info_tick(nome)
        if tick is None:
            continue
        bid = tick.bid if tick.bid > 0 else None
        ask = tick.ask if tick.ask > 0 else None
        ultimo = tick.last if tick.last > 0 else None
        if bid is None and ask is None and ultimo is None:
            continue
        linhas.append((dia_registro, "mt5", subjacente, nome, vencimento.isoformat(),
                       tipo, None, float(strike), float(spot),
                       ultimo, bid, ask, None, None, None, None, momento))

    mt5.shutdown()
    return gravar(linhas), "%d series com cotacao" % len(linhas)


def resumo():
    con = abrir()
    tabela = pd.read_sql_query(
        """SELECT data, fonte, COUNT(*) AS series,
                  SUM(CASE WHEN bid IS NOT NULL THEN 1 ELSE 0 END) AS com_spread
           FROM cadeia GROUP BY data, fonte ORDER BY data DESC LIMIT 30""", con)
    con.close()
    return tabela