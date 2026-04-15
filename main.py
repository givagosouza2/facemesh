import re
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import detrend

st.set_page_config(page_title="Análise Facial - Distância Euclidiana", layout="wide")

st.title("Análise de Marcadores Faciais")
st.write(
    """
    Esta aplicação:
    1. carrega um arquivo CSV contendo séries temporais de marcadores faciais (`X`, `Y`, `Z`);
    2. aplica **detrend linear** em cada coordenada de todos os marcadores;
    3. calcula a **distância euclidiana** da série detrendida de cada marcador;
    4. gera **estatística descritiva** por marcador;
    5. permite visualizar a série temporal da resultante de cada marcador.
    """
)

def find_time_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "Tempo_Decorrido(s)",
        "tempo_decorrido(s)",
        "tempo",
        "time",
        "Time"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def get_marker_bases(columns):
    bases = {}
    pattern = re.compile(r"^(.*)_(X|Y|Z)$")
    for col in columns:
        m = pattern.match(col)
        if m:
            base, axis = m.groups()
            bases.setdefault(base, set()).add(axis)
    valid = sorted([base for base, axes in bases.items() if {"X", "Y", "Z"}.issubset(axes)])
    return valid

@st.cache_data
def process_file(uploaded_bytes: bytes, file_name: str):
    if file_name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(uploaded_bytes))
    else:
        df = pd.read_excel(io.BytesIO(uploaded_bytes))

    time_col = find_time_column(df)
    marker_bases = get_marker_bases(df.columns)

    if len(marker_bases) == 0:
        raise ValueError("Nenhum conjunto de colunas no padrão marcador_X, marcador_Y, marcador_Z foi encontrado.")

    if time_col is None:
        time = pd.Series(np.arange(len(df)), name="Amostra")
        time_label = "Amostra"
    else:
        time = pd.to_numeric(df[time_col], errors="coerce")
        time_label = time_col

    resultantes = pd.DataFrame(index=df.index)
    detrended_xyz = pd.DataFrame(index=df.index)

    if time_col is not None:
        resultantes[time_label] = time
        detrended_xyz[time_label] = time

    stats_rows = []

    for base in marker_bases:
        x = pd.to_numeric(df[f"{base}_X"], errors="coerce")
        y = pd.to_numeric(df[f"{base}_Y"], errors="coerce")
        z = pd.to_numeric(df[f"{base}_Z"], errors="coerce")

        xyz = pd.concat([x, y, z], axis=1)
        xyz.columns = ["X", "Y", "Z"]

        # interpolação simples caso existam lacunas
        xyz = xyz.interpolate(limit_direction="both")

        dx = detrend(xyz["X"].to_numpy(), type="linear")
        dy = detrend(xyz["Y"].to_numpy(), type="linear")
        dz = detrend(xyz["Z"].to_numpy(), type="linear")

        detrended_xyz[f"{base}_X"] = dx
        detrended_xyz[f"{base}_Y"] = dy
        detrended_xyz[f"{base}_Z"] = dz

        dist = np.sqrt(dx**2 + dy**2)
        resultantes[base] = dist

        stats_rows.append({
            "Marcador": base,
            "N": int(np.isfinite(dist).sum()),
            "Média": float(np.mean(dist)),
            "Desvio-padrão": float(np.std(dist, ddof=1)) if len(dist) > 1 else np.nan,
            "Mínimo": float(np.min(dist)),
            "Q1": float(np.percentile(dist, 25)),
            "Mediana": float(np.median(dist)),
            "Q3": float(np.percentile(dist, 75)),
            "Máximo": float(np.max(dist)),
            "Amplitude": float(np.max(dist) - np.min(dist)),
            "RMS": float(np.sqrt(np.mean(dist**2))),
        })

    stats_df = pd.DataFrame(stats_rows).sort_values("Marcador").reset_index(drop=True)
    return df, resultantes, detrended_xyz, stats_df, marker_bases, time_label

uploaded_file = st.file_uploader("Carregue o arquivo CSV ou Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        raw_df, resultantes_df, detrended_xyz_df, stats_df, marker_bases, time_label = process_file(
            uploaded_file.getvalue(),
            uploaded_file.name
        )

        st.success(f"Arquivo processado com sucesso. Foram identificados {len(marker_bases)} marcadores válidos.")

        with st.expander("Visualizar estrutura dos dados de entrada"):
            st.dataframe(raw_df.head(), use_container_width=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Estatística descritiva por marcador")
            filtro = st.text_input("Filtrar marcador", "")
            stats_view = stats_df.copy()
            if filtro.strip():
                stats_view = stats_view[stats_view["Marcador"].str.contains(filtro.strip(), case=False, na=False)]
            st.dataframe(stats_view, use_container_width=True, height=500)

            csv_stats = stats_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Baixar estatística descritiva (CSV)",
                data=csv_stats,
                file_name="estatistica_descritiva_marcadores.csv",
                mime="text/csv"
            )

        with col2:
            st.subheader("Visualizador da série temporal da distância euclidiana")
            marcador = st.selectbox("Selecione o marcador", marker_bases, index=0)

            fig, ax = plt.subplots(figsize=(10, 4))
            x_axis = resultantes_df[time_label] if time_label in resultantes_df.columns else np.arange(len(resultantes_df))
            ax.plot(x_axis, resultantes_df[marcador], linewidth=1.5)
            ax.set_xlabel(time_label)
            ax.set_ylabel("Distância euclidiana")
            ax.set_title(f"Resultante temporal - {marcador}")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            st.write("Resumo do marcador selecionado:")
            st.dataframe(
                stats_df[stats_df["Marcador"] == marcador].reset_index(drop=True),
                use_container_width=True
            )

        st.subheader("Visualização múltipla")
        marcadores_multiplos = st.multiselect(
            "Selecione até 12 marcadores para comparar",
            marker_bases,
            default=marker_bases[:3]
        )

        if marcadores_multiplos:
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            x_axis = resultantes_df[time_label] if time_label in resultantes_df.columns else np.arange(len(resultantes_df))
            for m in marcadores_multiplos[:12]:
                ax2.plot(x_axis, resultantes_df[m], label=m, linewidth=1)
            ax2.set_xlabel(time_label)
            ax2.set_ylabel("Distância euclidiana")
            ax2.set_title("Comparação entre marcadores")
            ax2.grid(True, alpha=0.3)
            ax2.legend(ncol=3, fontsize=8)
            st.pyplot(fig2)

        with st.expander("Baixar séries temporais processadas"):
            csv_resultantes = resultantes_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Baixar resultantes de todos os marcadores (CSV)",
                data=csv_resultantes,
                file_name="resultantes_marcadores.csv",
                mime="text/csv"
            )

            csv_detrended = detrended_xyz_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Baixar coordenadas detrendadas (CSV)",
                data=csv_detrended,
                file_name="coordenadas_detrendadas.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
else:
    st.info("Envie um arquivo para iniciar a análise.")
