import sys
sys.path.append(".")
import requests

SUBJACENTE = "BOVA11"

url = (f'https://opcoes.net.br/listaopcoes/completa'
       f'?idAcao={SUBJACENTE}&listarVencimentos=false'
       f'&cotacoes=true&vencimentos=2026-10-16')
r = requests.get(url, timeout=30).json()

for coluna in r['data']['columns']:
    print("  [%2d] %-18s %s" % (coluna['index'], coluna['name'], coluna['title']))