import sys
sys.path.append(".")
import sqlite3
import pandas as pd
import config

con = sqlite3.connect(config.DB_OPCOES)
t = pd.read_sql_query("SELECT * FROM cadeia WHERE fonte = 'web'", con)
con.close()

print("Series: %d" % len(t))
print()
print("Preenchimento por coluna:")
for coluna in ["tipo", "estilo", "strike", "ultimo", "ultimo_negocio", "negocios", "volume", "formador"]:
    preenchidos = t[coluna].notna().sum()
    print("  %-10s %4d de %d" % (coluna, preenchidos, len(t)))

print()
print("Estilos encontrados:", t["estilo"].value_counts().to_dict())
print("Com negocio no dia: %d" % (t["negocios"].fillna(0) > 0).sum())
print()
print("Dez series mais negociadas:")
print(t.nlargest(10, "volume")[["serie", "tipo", "estilo", "strike", "ultimo",
                                "ultimo_negocio", "negocios", "volume"]].to_string(index=False))