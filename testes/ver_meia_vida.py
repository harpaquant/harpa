import sys
sys.path.append(".")
import sqlite3
import pandas as pd
import config

con = sqlite3.connect(config.DB_PARES)
t = pd.read_sql_query("SELECT acao1, acao2, meia_vida, z FROM varredura", con)
con.close()

print(t["meia_vida"].describe())
print()
print(t.sort_values("meia_vida").head(10).to_string())