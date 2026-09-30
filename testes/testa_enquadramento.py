import sys
sys.path.append(".")
from risco import enquadramento

precos = {"BOVA11": 180.0, "CMIG4": 11.30, "ENGI11": 51.80}
classes = {"BOVA11": "etf", "CMIG4": "acao", "ENGI11": "acao"}

# carteira tipica: quase tudo em BOVA11, um par montado, algum caixa
posicao = {"BOVA11": 4000, "CMIG4": 3000, "ENGI11": -650}
caixa = 60000.0

r = enquadramento.verificar(posicao, precos, classes, caixa)
print("carteira atual")
print("  PL R$ %.2f" % r["pl"])
print("  renda variavel %.1f%%" % (r["fracao_rv"] * 100))
print("  exposicao %.2f vez o PL" % r["alavancagem"])
print("  vendidos:", r["vendidos"])
print("  aprovado:", r["aprovado"], r["problemas"])

# ordem que consome todo o caixa e mais um pouco
ordem = {"ticker": "BOVA11", "quantidade": 500, "preco": 180.0}
r2 = enquadramento.verificar(posicao, precos, classes, caixa, ordem)
print("\ncompra de 500 BOVA11")
print("  caixa resultante R$ %.2f" % r2["caixa"])
print("  aprovado:", r2["aprovado"], r2["problemas"])

# venda que derruba a renda variavel abaixo do minimo
ordem = {"ticker": "BOVA11", "quantidade": -3500, "preco": 180.0}
r3 = enquadramento.verificar(posicao, precos, classes, caixa, ordem)
print("\nvenda de 3500 BOVA11")
print("  renda variavel %.1f%%" % (r3["fracao_rv"] * 100))
print("  aprovado:", r3["aprovado"], r3["problemas"])