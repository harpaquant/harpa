import sys
sys.path.append(".")
from dados import opcoes

total, vencimentos = opcoes.coletar_web_tudo(quantos_vencimentos=3)
print("Gravadas %d series nos vencimentos %s" % (total, vencimentos))
print()
print(opcoes.resumo().to_string())