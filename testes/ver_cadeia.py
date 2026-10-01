import sys
sys.path.append(".")
import requests
import json

SUBJACENTE = "BOVA11"

url = (f'https://opcoes.net.br/listaopcoes/completa'
       f'?idAcao={SUBJACENTE}&listarVencimentos=true&cotacoes=true')
r = requests.get(url, timeout=30).json()
vencimentos = [v['value'] for v in r['data']['vencimentos']]
print("Vencimentos:", vencimentos[:8])

venc = vencimentos[2] if len(vencimentos) > 2 else vencimentos[0]
print("Analisando o vencimento", venc)
print()

url = (f'https://opcoes.net.br/listaopcoes/completa'
       f'?idAcao={SUBJACENTE}&listarVencimentos=false'
       f'&cotacoes=true&vencimentos={venc}')
r = requests.get(url, timeout=30).json()

print("Chaves no topo da resposta:", list(r.keys()))
print("Chaves dentro de data:", list(r['data'].keys()))
print()

for chave in r['data']:
    if chave == 'cotacoesOpcoes':
        continue
    valor = r['data'][chave]
    texto = json.dumps(valor, ensure_ascii=False)
    print("  %s: %s" % (chave, texto[:400]))

print()
linhas = r['data']['cotacoesOpcoes']
print("Series: %d" % len(linhas))

cheia = None
for linha in linhas:
    preenchidos = sum(1 for v in linha if v is not None and v != "")
    if cheia is None or preenchidos > cheia[0]:
        cheia = (preenchidos, linha)

print()
print("Serie com mais campos preenchidos (%d de %d):" % (cheia[0], len(cheia[1])))
for i, valor in enumerate(cheia[1]):
    print("  [%2d] %s" % (i, json.dumps(valor, ensure_ascii=False)))