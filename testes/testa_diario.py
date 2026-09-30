import sys
sys.path.append(".")
from execucao import diario

# sinal gerado e executado
ident = diario.registrar("pares", "CMIG4", "compra", 3000, 11.30,
                         "gerado", pl=780230.0, fracao_rv=0.923,
                         observacao="par CMIG4/ENGI11, z = -2.1")
diario.atualizar(ident, "executado", preco_executado=11.32)

# sinal barrado pelo enquadramento
ident = diario.registrar("pares", "ENGI11", "venda", -650, 51.80,
                         "barrado", motivo="renda variavel em 11.6%, abaixo do minimo")

# sinal gerado e nao executado por falta de liquidez
diario.registrar("paridade", "BOVA11", "compra", 500, 180.00,
                 "cancelado", motivo="spread acima do limite")

for linha in diario.ler(limite=10):
    print(linha)