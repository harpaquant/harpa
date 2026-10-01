import sys
sys.path.append(".")
import sqlite3
import config

con = sqlite3.connect(config.DB_OPCOES)
con.execute("DROP TABLE IF EXISTS cadeia")
con.commit()
con.close()
print("Coleta web apagada.")