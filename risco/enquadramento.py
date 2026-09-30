# Verifica enquadramento do clube antes de cada ordem.
# Posicao esperada: dicionario {ticker: quantidade}, quantidade negativa = vendido.
# Precos esperados: dicionario {ticker: preco}.

import config

RENDA_VARIAVEL = ("acao", "etf", "opcao")


def valor_posicao(posicao, precos):
    total = 0.0
    for ticker, qtde in posicao.items():
        total += qtde * precos[ticker]
    return total


def exposicao_renda_variavel(posicao, precos, classes, liquida=True):
    total = 0.0
    for ticker, qtde in posicao.items():
        if classes.get(ticker) in RENDA_VARIAVEL:
            valor = qtde * precos[ticker]
            total += valor if liquida else abs(valor)
    return total


def verificar(posicao, precos, classes, caixa, ordem=None):
    # ordem: dicionario {ticker, quantidade, preco} ou None para checar a carteira atual
    nova = dict(posicao)
    caixa_novo = caixa
    if ordem is not None:
        ticker = ordem["ticker"]
        nova[ticker] = nova.get(ticker, 0) + ordem["quantidade"]
        caixa_novo = caixa - ordem["quantidade"] * ordem["preco"]

    pl = valor_posicao(nova, precos) + caixa_novo
    rv = exposicao_renda_variavel(nova, precos, classes, config.RV_LIQUIDA)
    fracao_rv = rv / pl if pl > 0 else 0.0

    vendidos = []
    for ticker, qtde in nova.items():
        if qtde < 0:
            vendidos.append(ticker)

    alavancagem = valor_posicao(nova, precos) / pl if pl > 0 else 0.0

    problemas = []
    if fracao_rv < config.MINIMO_RENDA_VARIAVEL:
        problemas.append("renda variavel em %.1f%%, abaixo do minimo de %.0f%%"
                         % (fracao_rv * 100, config.MINIMO_RENDA_VARIAVEL * 100))
    if caixa_novo < 0:
        problemas.append("caixa negativo de R$ %.2f" % caixa_novo)
    if alavancagem > 1.0 and not config.PERMITE_ALAVANCAGEM:
        problemas.append("exposicao de %.2f vez o PL" % alavancagem)

    return {"aprovado": len(problemas) == 0,
            "pl": pl,
            "fracao_rv": fracao_rv,
            "alavancagem": alavancagem,
            "caixa": caixa_novo,
            "vendidos": vendidos,
            "problemas": problemas}