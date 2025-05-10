import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Aviator Previsão", layout="centered")
st.title("Aviator Previsão Inteligente")

st.markdown("### Inserir Dados")
novos_valores = st.text_input("Digite os valores (ex: 1.25, 2.60, 3.44):")
botao_add = st.button("Adicionar")

if 'dados' not in st.session_state:
    st.session_state.dados = []

if botao_add and novos_valores:
    try:
        lista = [float(x.strip().replace("x", "")) for x in novos_valores.split(",")]
        st.session_state.dados.extend(lista)
        st.success("Valores adicionados com sucesso!")
    except:
        st.error("Erro: certifique-se de inserir números separados por vírgula.")

if st.session_state.dados:
    st.markdown("### Histórico de Dados")
    df = pd.DataFrame(st.session_state.dados, columns=["Valor"])
    st.dataframe(df.tail(30), height=200)

    # Previsão com regressão linear
    X = np.array(range(len(st.session_state.dados))).reshape(-1, 1)
    y = np.array(st.session_state.dados)
    modelo = LinearRegression()
    modelo.fit(X, y)
    proximo_indice = len(st.session_state.dados)
    previsao = modelo.predict([[proximo_indice]])[0]

    media_passada = np.mean(st.session_state.dados)
    minimo_proximo = min(y[-5:]) if len(y) >= 5 else min(y)
    previsao_media = (previsao + media_passada) / 2

    st.markdown("### Previsões e Recomendações")
    col1, col2, col3 = st.columns(3)
    col1.metric("Próxima Estimativa", f"{previsao:.2f}x")
    col2.metric("Mínimo Esperado", f"{minimo_proximo:.2f}x")
    col3.metric("Média Estimada", f"{previsao_media:.2f}x")

    # Alertas visuais e recomendações
    def gerar_alerta(valor):
        if valor < 1.50:
            return "Alta probabilidade de queda. Evite apostas.", "⚠️"
        elif valor < 2.00:
            return "Momento instável. Cautela!", "🔶"
        else:
            return "Momento estável. Boa oportunidade.", "✅"

    recomendacao, icone = gerar_alerta(previsao)
    st.markdown(f"### {icone} Recomendação: **{recomendacao}**")