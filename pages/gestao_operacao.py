# -*- coding: utf-8 -*-
# Harpa Quant - modulo de acompanhamento da operacao de gestao
# Pagina Streamlit. Base SQLite propria, separada dos dados de mercado.
# Rodar: streamlit run gestao_operacao.py
# ou salvar em C:\repo\harpa\pages\ para virar pagina do app principal.

import sqlite3
import datetime
import pandas as pd
import streamlit as st

CAMINHO_BANCO = r"C:\repo\harpa\operacao.db"

# ---------------------------------------------------------------- meta
META_AUM_FASE2 = 12_000_000     # gatilho de transicao para a Fase 2
PERDA_RT_ANUAL = 91_000         # perda liquida anual ao sair da DE

st.set_page_config(page_title="Operacao - Gestao", layout="wide")

conexao = sqlite3.connect(CAMINHO_BANCO, check_same_thread=False)
cursor = conexao.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS carteiras (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cliente TEXT NOT NULL,
    veiculo TEXT,
    origem TEXT,
    data_inicio TEXT,
    aum REAL,
    taxa REAL,
    situacao TEXT,
    observacao TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS aum_historico (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    data TEXT NOT NULL,
    carteira_id INTEGER,
    aum REAL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS pipeline (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nome TEXT NOT NULL,
    origem TEXT,
    estagio TEXT,
    potencial REAL,
    proxima_acao TEXT,
    data_proxima TEXT,
    ultimo_contato TEXT,
    observacao TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS rotina (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tarefa TEXT NOT NULL,
    categoria TEXT,
    cadencia TEXT,
    responsavel TEXT,
    proxima_data TEXT,
    ultima_execucao TEXT,
    ativa INTEGER DEFAULT 1
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS canais (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canal TEXT NOT NULL,
    tipo TEXT,
    responsavel TEXT,
    situacao TEXT,
    meta_ano REAL,
    proxima_acao TEXT,
    data_proxima TEXT,
    observacao TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS iniciativas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iniciativa TEXT NOT NULL,
    canal TEXT,
    objetivo TEXT,
    situacao TEXT,
    prazo TEXT,
    proximo_passo TEXT,
    concluida INTEGER DEFAULT 0
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS diario (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    data TEXT NOT NULL,
    tipo TEXT,
    descricao TEXT,
    referencia TEXT
)
""")

conexao.commit()

hoje = datetime.date.today()
hoje_txt = hoje.isoformat()

st.title("Operacao de gestao de patrimonio")
st.caption("Acompanhamento comercial, administrativo e de carteiras. Base local, sem envio externo.")

aba_painel, aba_carteiras, aba_distrib, aba_pipeline, aba_rotina, aba_diario = st.tabs(
    ["Painel", "Carteiras", "Distribuicao", "Pipeline comercial", "Rotina", "Diario"]
)

# ================================================================ PAINEL
with aba_painel:
    df_cart = pd.read_sql_query(
        "SELECT * FROM carteiras WHERE situacao = 'ativa'", conexao
    )

    aum_total = float(df_cart["aum"].sum()) if len(df_cart) else 0.0
    receita_ano = float((df_cart["aum"] * df_cart["taxa"]).sum()) if len(df_cart) else 0.0
    n_contas = len(df_cart)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUM sob gestao", f"R$ {aum_total:,.0f}".replace(",", "."))
    c2.metric("Receita anualizada", f"R$ {receita_ano:,.0f}".replace(",", "."))
    c3.metric("Contas ativas", n_contas)
    cobertura = receita_ano / PERDA_RT_ANUAL if PERDA_RT_ANUAL else 0
    c4.metric("Cobertura da perda de RT", f"{cobertura:.0%}")

    st.divider()

    st.subheader("Caminho ate a Fase 2")
    falta = max(META_AUM_FASE2 - aum_total, 0)
    progresso = min(aum_total / META_AUM_FASE2, 1.0) if META_AUM_FASE2 else 0
    st.progress(progresso)
    st.write(
        f"AUM atual de R$ {aum_total:,.0f}".replace(",", ".")
        + f" contra meta de R$ {META_AUM_FASE2:,.0f}".replace(",", ".")
        + f". Faltam R$ {falta:,.0f}".replace(",", ".")
        + f" ({progresso:.0%} do caminho)."
    )

    st.subheader("Gatilhos que encerram a Fase 1")
    gatilhos = [
        "Clube atinge o teto de cotistas",
        "AUM total ultrapassa a faixa de R$ 12 a 15 milhoes",
        "Cliente relevante exige contraparte pessoa juridica",
        "Socio originador entrega volume consistente",
        "Caixa acumulado cobre a constituicao com folga",
        "Qualquer sinalizacao de fiscalizacao",
    ]
    for g in gatilhos:
        st.checkbox(g, key="gat_" + g[:18])

    st.divider()

    df_pipe = pd.read_sql_query("SELECT * FROM pipeline", conexao)
    if len(df_pipe):
        st.subheader("Pipeline por estagio")
        resumo = df_pipe.groupby("estagio").agg(
            quantidade=("nome", "count"), potencial=("potencial", "sum")
        ).reset_index()
        st.dataframe(resumo, width='stretch', hide_index=True)

    df_org = pd.read_sql_query(
        "SELECT origem, COUNT(*) AS contas, SUM(aum) AS aum FROM carteiras "
        "WHERE situacao = 'ativa' GROUP BY origem ORDER BY aum DESC", conexao
    )
    if len(df_org):
        st.subheader("Originacao por canal")
        st.dataframe(df_org, width='stretch', hide_index=True)

    df_ini_ab = pd.read_sql_query(
        "SELECT iniciativa, canal, situacao, prazo, proximo_passo FROM iniciativas "
        "WHERE concluida = 0 ORDER BY prazo LIMIT 10", conexao
    )
    if len(df_ini_ab):
        st.subheader("Iniciativas de distribuicao em aberto")
        st.dataframe(df_ini_ab, width='stretch', hide_index=True)

    df_rot = pd.read_sql_query(
        "SELECT tarefa, categoria, cadencia, proxima_data FROM rotina "
        "WHERE ativa = 1 AND proxima_data <= ? ORDER BY proxima_data",
        conexao, params=(hoje_txt,)
    )
    if len(df_rot):
        st.subheader("Rotina vencida ou para hoje")
        st.dataframe(df_rot, width='stretch', hide_index=True)

# ============================================================= CARTEIRAS
with aba_carteiras:
    st.subheader("Carteiras sob gestao")

    df_cart_full = pd.read_sql_query("SELECT * FROM carteiras ORDER BY aum DESC", conexao)
    if len(df_cart_full):
        exibe = df_cart_full.copy()
        exibe["receita_ano"] = exibe["aum"] * exibe["taxa"]
        st.dataframe(
            exibe[["cliente", "veiculo", "origem", "aum", "taxa", "receita_ano", "situacao"]],
            width='stretch', hide_index=True
        )
    else:
        st.info("Nenhuma carteira cadastrada.")

    with st.expander("Cadastrar carteira"):
        nome_cli = st.text_input("Cliente", key="cart_nome")
        col_a, col_b = st.columns(2)
        veic = col_a.selectbox(
            "Veiculo", ["Carteira administrada", "Clube Fortem", "Outro"], key="cart_veic"
        )
        orig = col_b.selectbox(
            "Origem", ["Rede pessoal", "Daniel", "David", "Escritorio MB",
                       "Conteudo publico", "Indicacao de cliente", "Outra"],
            key="cart_orig"
        )
        col_c, col_d = st.columns(2)
        valor = col_c.number_input("AUM (R$)", min_value=0.0, step=10000.0, key="cart_aum")
        tx = col_d.number_input(
            "Taxa anual (ex.: 0.015)", min_value=0.0, max_value=0.05,
            value=0.015, step=0.0005, format="%.4f", key="cart_taxa"
        )
        inicio = st.date_input("Data de inicio", value=hoje, key="cart_ini")
        obs_c = st.text_area("Observacao", key="cart_obs")

        if st.button("Salvar carteira", key="btn_cart"):
            if nome_cli.strip():
                cursor.execute(
                    "INSERT INTO carteiras (cliente, veiculo, origem, data_inicio, aum, taxa, situacao, observacao) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'ativa', ?)",
                    (nome_cli.strip(), veic, orig, inicio.isoformat(), valor, tx, obs_c)
                )
                novo_id = cursor.lastrowid
                cursor.execute(
                    "INSERT INTO aum_historico (data, carteira_id, aum) VALUES (?, ?, ?)",
                    (hoje_txt, novo_id, valor)
                )
                conexao.commit()
                st.success("Carteira cadastrada.")
                st.rerun()
            else:
                st.warning("Informe o nome do cliente.")

    with st.expander("Atualizar AUM"):
        if len(df_cart_full):
            opcoes = {f"{r.cliente} ({r.veiculo})": r.id for r in df_cart_full.itertuples()}
            escolha = st.selectbox("Carteira", list(opcoes.keys()), key="upd_sel")
            novo_aum = st.number_input("Novo AUM (R$)", min_value=0.0, step=10000.0, key="upd_aum")
            if st.button("Registrar atualizacao", key="btn_upd"):
                cid = opcoes[escolha]
                cursor.execute("UPDATE carteiras SET aum = ? WHERE id = ?", (novo_aum, cid))
                cursor.execute(
                    "INSERT INTO aum_historico (data, carteira_id, aum) VALUES (?, ?, ?)",
                    (hoje_txt, cid, novo_aum)
                )
                conexao.commit()
                st.success("AUM atualizado.")
                st.rerun()

    df_hist = pd.read_sql_query(
        "SELECT data, SUM(aum) AS aum_total FROM aum_historico GROUP BY data ORDER BY data",
        conexao
    )
    if len(df_hist) > 1:
        st.subheader("Evolucao do AUM")
        st.line_chart(df_hist.set_index("data")["aum_total"])

# =========================================================== DISTRIBUICAO
with aba_distrib:
    st.subheader("Estrategia de distribuicao")
    st.caption("Canais pelos quais o patrimonio chega, metas por canal e iniciativas em andamento.")

    df_can = pd.read_sql_query("SELECT * FROM canais ORDER BY canal", conexao)
    df_cart_org = pd.read_sql_query(
        "SELECT origem, COUNT(*) AS contas, SUM(aum) AS aum FROM carteiras "
        "WHERE situacao = 'ativa' GROUP BY origem", conexao
    )

    if len(df_can):
        base = df_can[["canal", "tipo", "responsavel", "situacao", "meta_ano",
                       "proxima_acao", "data_proxima"]].copy()
        if len(df_cart_org):
            base = base.merge(
                df_cart_org.rename(columns={"origem": "canal"}), on="canal", how="left"
            )
            base["aum"] = base["aum"].fillna(0.0)
            base["contas"] = base["contas"].fillna(0).astype(int)
            base["atingimento"] = base.apply(
                lambda r: (r["aum"] / r["meta_ano"]) if r["meta_ano"] else 0.0, axis=1
            )
        st.dataframe(base, width='stretch', hide_index=True)
    else:
        st.info("Nenhum canal cadastrado. Use o bloco abaixo para carregar o conjunto inicial.")

    if st.button("Carregar canais iniciais", key="btn_seed_can"):
        canais_iniciais = [
            ("Rede pessoal", "Direto", "Vinicio", "Ativo", 3_000_000.0,
             "Abordar os contatos que ainda nao sabem da atividade"),
            ("Daniel", "Originacao por terceiro", "Daniel Dantas", "Ativo", 4_000_000.0,
             "Enviar apresentacao para ele compartilhar"),
            ("David", "Originacao por terceiro", "David Macedo", "Ativo", 3_000_000.0,
             "Conversa presencial em Fortaleza"),
            ("Eduardo Holder", "Audiencia qualificada", "Eduardo Holder", "Ativo", 1_000_000.0,
             "Redirecionar indicacoes para publico empresarial"),
            ("Escritorio MB", "Parceria institucional", "Vinicio", "Em negociacao", 2_000_000.0,
             "Aguardar resposta do segundo socio"),
            ("Clube Fortem", "Veiculo proprio", "Vinicio", "Ativo", 2_000_000.0,
             "Atualizar apresentacao e captar cotistas"),
            ("Conteudo publico", "Inbound", "Vinicio", "A estruturar", 1_000_000.0,
             "Definir cadencia de publicacao"),
            ("Indicacao de cliente", "Referencia", "Vinicio", "Ativo", 1_500_000.0,
             "Pedir indicacao apos entrega de relatorio"),
        ]
        for nome_can, tipo_can, resp_can, sit_can, meta_can, acao_can in canais_iniciais:
            cursor.execute(
                "INSERT INTO canais (canal, tipo, responsavel, situacao, meta_ano, "
                "proxima_acao, data_proxima) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (nome_can, tipo_can, resp_can, sit_can, meta_can, acao_can, hoje_txt)
            )
        conexao.commit()
        st.success("Canais carregados.")
        st.rerun()

    with st.expander("Cadastrar ou atualizar canal"):
        can_nome = st.text_input("Canal", key="can_nome")
        col_k, col_l = st.columns(2)
        can_tipo = col_k.selectbox(
            "Tipo", ["Direto", "Originacao por terceiro", "Audiencia qualificada",
                     "Parceria institucional", "Veiculo proprio", "Inbound", "Referencia"],
            key="can_tipo"
        )
        can_sit = col_l.selectbox(
            "Situacao", ["Ativo", "Em negociacao", "A estruturar", "Pausado", "Encerrado"],
            key="can_sit"
        )
        col_m, col_n = st.columns(2)
        can_resp = col_m.text_input("Responsavel", value="Vinicio", key="can_resp")
        can_meta = col_n.number_input("Meta de originacao no ano (R$)", min_value=0.0,
                                      step=250000.0, key="can_meta")
        can_acao = st.text_input("Proxima acao", key="can_acao")
        can_data = st.date_input("Data da proxima acao", value=hoje, key="can_data")

        if st.button("Salvar canal", key="btn_can"):
            if can_nome.strip():
                cursor.execute("SELECT id FROM canais WHERE canal = ?", (can_nome.strip(),))
                existente = cursor.fetchone()
                if existente:
                    cursor.execute(
                        "UPDATE canais SET tipo = ?, responsavel = ?, situacao = ?, meta_ano = ?, "
                        "proxima_acao = ?, data_proxima = ? WHERE id = ?",
                        (can_tipo, can_resp, can_sit, can_meta, can_acao,
                         can_data.isoformat(), existente[0])
                    )
                else:
                    cursor.execute(
                        "INSERT INTO canais (canal, tipo, responsavel, situacao, meta_ano, "
                        "proxima_acao, data_proxima) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (can_nome.strip(), can_tipo, can_resp, can_sit, can_meta,
                         can_acao, can_data.isoformat())
                    )
                conexao.commit()
                st.success("Canal salvo.")
                st.rerun()

    st.divider()
    st.subheader("Iniciativas de distribuicao")

    df_ini = pd.read_sql_query(
        "SELECT * FROM iniciativas WHERE concluida = 0 ORDER BY prazo", conexao
    )
    if len(df_ini):
        st.dataframe(
            df_ini[["iniciativa", "canal", "situacao", "prazo", "proximo_passo"]],
            width='stretch', hide_index=True
        )
    else:
        st.info("Nenhuma iniciativa aberta.")

    with st.expander("Nova iniciativa"):
        ini_nome = st.text_input("Iniciativa", key="ini_nome")
        col_o, col_p = st.columns(2)
        lista_canais = list(df_can["canal"]) if len(df_can) else ["Geral"]
        ini_canal = col_o.selectbox("Canal", lista_canais + ["Geral"], key="ini_canal")
        ini_sit = col_p.selectbox(
            "Situacao", ["Nao iniciada", "Em andamento", "Aguardando terceiro", "Bloqueada"],
            key="ini_sit"
        )
        ini_obj = st.text_input("Objetivo", key="ini_obj")
        ini_passo = st.text_input("Proximo passo", key="ini_passo")
        ini_prazo = st.date_input("Prazo", value=hoje, key="ini_prazo")

        if st.button("Salvar iniciativa", key="btn_ini"):
            if ini_nome.strip():
                cursor.execute(
                    "INSERT INTO iniciativas (iniciativa, canal, objetivo, situacao, prazo, "
                    "proximo_passo, concluida) VALUES (?, ?, ?, ?, ?, ?, 0)",
                    (ini_nome.strip(), ini_canal, ini_obj, ini_sit,
                     ini_prazo.isoformat(), ini_passo)
                )
                conexao.commit()
                st.success("Iniciativa criada.")
                st.rerun()

    with st.expander("Atualizar iniciativa"):
        if len(df_ini):
            op_i = {r.iniciativa: r.id for r in df_ini.itertuples()}
            sel_i = st.selectbox("Iniciativa", list(op_i.keys()), key="upi_sel")
            nova_sit = st.selectbox(
                "Situacao", ["Nao iniciada", "Em andamento", "Aguardando terceiro",
                             "Bloqueada", "Concluida"], key="upi_sit"
            )
            novo_passo = st.text_input("Proximo passo", key="upi_passo")
            novo_prazo = st.date_input("Prazo", value=hoje, key="upi_prazo")
            if st.button("Atualizar", key="btn_upi"):
                concl = 1 if nova_sit == "Concluida" else 0
                cursor.execute(
                    "UPDATE iniciativas SET situacao = ?, proximo_passo = ?, prazo = ?, "
                    "concluida = ? WHERE id = ?",
                    (nova_sit, novo_passo, novo_prazo.isoformat(), concl, op_i[sel_i])
                )
                cursor.execute(
                    "INSERT INTO diario (data, tipo, descricao, referencia) "
                    "VALUES (?, 'Comercial', ?, ?)",
                    (hoje_txt, "Iniciativa atualizada para " + nova_sit, sel_i)
                )
                conexao.commit()
                st.success("Iniciativa atualizada.")
                st.rerun()

# ============================================================== PIPELINE
with aba_pipeline:
    st.subheader("Pipeline comercial")

    ESTAGIOS = [
        "Mapeado", "Sabe que faco gestao", "Conversa iniciada",
        "Analise de carteira", "Proposta", "Fechado", "Perdido"
    ]

    df_p = pd.read_sql_query("SELECT * FROM pipeline ORDER BY data_proxima", conexao)
    if len(df_p):
        st.dataframe(
            df_p[["nome", "origem", "estagio", "potencial", "proxima_acao", "data_proxima"]],
            width='stretch', hide_index=True
        )
    else:
        st.info("Pipeline vazio.")

    with st.expander("Adicionar ao pipeline"):
        p_nome = st.text_input("Nome", key="pip_nome")
        col_e, col_f = st.columns(2)
        p_orig = col_e.selectbox(
            "Origem", ["Rede pessoal", "Daniel", "David", "Eduardo Holder",
                       "Escritorio MB", "Conteudo publico", "Indicacao de cliente", "Outra"],
            key="pip_orig"
        )
        p_est = col_f.selectbox("Estagio", ESTAGIOS, key="pip_est")
        col_g, col_h = st.columns(2)
        p_pot = col_g.number_input("Potencial estimado (R$)", min_value=0.0, step=50000.0, key="pip_pot")
        p_data = col_h.date_input("Data da proxima acao", value=hoje, key="pip_data")
        p_acao = st.text_input("Proxima acao", key="pip_acao")
        p_obs = st.text_area("Observacao", key="pip_obs")

        if st.button("Salvar no pipeline", key="btn_pip"):
            if p_nome.strip():
                cursor.execute(
                    "INSERT INTO pipeline (nome, origem, estagio, potencial, proxima_acao, "
                    "data_proxima, ultimo_contato, observacao) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (p_nome.strip(), p_orig, p_est, p_pot, p_acao,
                     p_data.isoformat(), hoje_txt, p_obs)
                )
                conexao.commit()
                st.success("Registro criado.")
                st.rerun()
            else:
                st.warning("Informe o nome.")

    with st.expander("Mover estagio"):
        if len(df_p):
            op_p = {f"{r.nome} - {r.estagio}": r.id for r in df_p.itertuples()}
            sel_p = st.selectbox("Registro", list(op_p.keys()), key="mv_sel")
            novo_est = st.selectbox("Novo estagio", ESTAGIOS, key="mv_est")
            nova_acao = st.text_input("Proxima acao", key="mv_acao")
            nova_data = st.date_input("Data da proxima acao", value=hoje, key="mv_data")
            if st.button("Atualizar registro", key="btn_mv"):
                cursor.execute(
                    "UPDATE pipeline SET estagio = ?, proxima_acao = ?, data_proxima = ?, "
                    "ultimo_contato = ? WHERE id = ?",
                    (novo_est, nova_acao, nova_data.isoformat(), hoje_txt, op_p[sel_p])
                )
                conexao.commit()
                st.success("Registro atualizado.")
                st.rerun()

# ================================================================ ROTINA
with aba_rotina:
    st.subheader("Rotina administrativa e regulatoria")

    df_r = pd.read_sql_query(
        "SELECT * FROM rotina WHERE ativa = 1 ORDER BY proxima_data", conexao
    )
    if len(df_r):
        st.dataframe(
            df_r[["tarefa", "categoria", "cadencia", "proxima_data", "ultima_execucao"]],
            width='stretch', hide_index=True
        )
    else:
        st.info("Nenhuma tarefa cadastrada. Use o bloco abaixo para carregar o conjunto inicial.")

    if st.button("Carregar rotina inicial sugerida", key="btn_seed"):
        sugestoes = [
            ("Relatorio periodico aos clientes", "Cliente", "Mensal"),
            ("Revisao de enquadramento das carteiras", "Gestao", "Mensal"),
            ("Conferencia de custos e taxas cobradas", "Gestao", "Mensal"),
            ("Revisao de perfil e suitability", "Compliance", "Anual"),
            ("Atualizacao cadastral junto a CVM", "Regulatorio", "Anual"),
            ("Conferencia de documentos e procuracoes vigentes", "Compliance", "Semestral"),
            ("Registro de origem de cada cliente novo", "Comercial", "Continuo"),
            ("Revisao do pipeline e das proximas acoes", "Comercial", "Semanal"),
            ("Conferencia dos gatilhos de transicao para a Fase 2", "Estrategico", "Trimestral"),
            ("Acompanhamento da trilha juridica de regularizacao", "Estrategico", "Mensal"),
        ]
        for t, cat, cad in sugestoes:
            cursor.execute(
                "INSERT INTO rotina (tarefa, categoria, cadencia, responsavel, proxima_data, ativa) "
                "VALUES (?, ?, ?, 'Vinicio', ?, 1)",
                (t, cat, cad, hoje_txt)
            )
        conexao.commit()
        st.success("Rotina inicial carregada.")
        st.rerun()

    with st.expander("Nova tarefa"):
        r_tarefa = st.text_input("Tarefa", key="rot_tarefa")
        col_i, col_j = st.columns(2)
        r_cat = col_i.selectbox(
            "Categoria", ["Cliente", "Gestao", "Compliance", "Regulatorio",
                          "Comercial", "Estrategico"], key="rot_cat"
        )
        r_cad = col_j.selectbox(
            "Cadencia", ["Diaria", "Semanal", "Mensal", "Trimestral",
                         "Semestral", "Anual", "Continuo"], key="rot_cad"
        )
        r_data = st.date_input("Proxima data", value=hoje, key="rot_data")
        if st.button("Salvar tarefa", key="btn_rot"):
            if r_tarefa.strip():
                cursor.execute(
                    "INSERT INTO rotina (tarefa, categoria, cadencia, responsavel, proxima_data, ativa) "
                    "VALUES (?, ?, ?, 'Vinicio', ?, 1)",
                    (r_tarefa.strip(), r_cat, r_cad, r_data.isoformat())
                )
                conexao.commit()
                st.success("Tarefa criada.")
                st.rerun()

    with st.expander("Marcar execucao"):
        if len(df_r):
            op_r = {r.tarefa: r.id for r in df_r.itertuples()}
            sel_r = st.selectbox("Tarefa", list(op_r.keys()), key="exec_sel")
            prox = st.date_input("Proxima ocorrencia", value=hoje, key="exec_data")
            if st.button("Registrar execucao", key="btn_exec"):
                cursor.execute(
                    "UPDATE rotina SET ultima_execucao = ?, proxima_data = ? WHERE id = ?",
                    (hoje_txt, prox.isoformat(), op_r[sel_r])
                )
                cursor.execute(
                    "INSERT INTO diario (data, tipo, descricao, referencia) VALUES (?, 'Rotina', ?, '')",
                    (hoje_txt, sel_r)
                )
                conexao.commit()
                st.success("Execucao registrada.")
                st.rerun()

# ================================================================ DIARIO
with aba_diario:
    st.subheader("Diario da operacao")
    st.caption("Registro corrido do que foi feito. Util para reconstituir historico e justificar decisoes.")

    d_tipo = st.selectbox(
        "Tipo", ["Comercial", "Gestao", "Cliente", "Regulatorio",
                 "Parceria", "Rotina", "Outro"], key="di_tipo"
    )
    d_desc = st.text_area("Descricao", key="di_desc")
    d_ref = st.text_input("Referencia (cliente, parceiro, processo)", key="di_ref")

    if st.button("Registrar", key="btn_di"):
        if d_desc.strip():
            cursor.execute(
                "INSERT INTO diario (data, tipo, descricao, referencia) VALUES (?, ?, ?, ?)",
                (hoje_txt, d_tipo, d_desc.strip(), d_ref)
            )
            conexao.commit()
            st.success("Registro salvo.")
            st.rerun()

    df_d = pd.read_sql_query(
        "SELECT data, tipo, descricao, referencia FROM diario ORDER BY id DESC LIMIT 100", conexao
    )
    if len(df_d):
        st.dataframe(df_d, width='stretch', hide_index=True)
