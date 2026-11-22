"""Dashboard del scorecard de riesgo de crédito.

Arranque:
    streamlit run app/app.py

Lee solo los artefactos que guarda `python -m scorecard.pipeline`: no necesita los datos
ni las librerías de entrenamiento.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from scorecard.puntaje import puntuar_solicitud  # noqa: E402

ARTEFACTOS = RAIZ / "artefactos"
AZUL, NARANJA, AQUA, AMARILLO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRIS = "#898781"

PROPOSITOS = {
    "Consolidación de deudas": "debt_consolidation",
    "Tarjeta de crédito": "credit_card",
    "Mejoras del hogar": "home_improvement",
    "Compra importante": "major_purchase",
    "Pequeño negocio": "small_business",
    "Auto": "car",
    "Boda": "wedding",
    "Gastos médicos": "medical",
    "Mudanza": "moving",
    "Vacaciones": "vacation",
    "Vivienda": "house",
    "Educación": "educational",
    "Energía renovable": "renewable_energy",
    "Otro": "other",
}
ESTADOS = {
    "California (CA)": "CA",
    "Nueva York (NY)": "NY",
    "Florida (FL)": "FL",
    "Texas (TX)": "TX",
    "Nueva Jersey (NJ)": "NJ",
    "Illinois (IL)": "IL",
}
ETIQUETAS = {
    "calificacion": "Calificación de LendingClub",
    "plazo": "Plazo (meses)",
    "proposito": "Propósito",
    "ingreso_anual": "Ingreso anual (USD)",
    "estado": "Estado",
}
ANCHO = "stretch"

st.set_page_config(page_title="Scorecard de riesgo de crédito", layout="wide")


@st.cache_data
def leer(nombre: str) -> pd.DataFrame:
    return pd.read_csv(ARTEFACTOS / f"{nombre}.csv")


@st.cache_data
def leer_json(nombre: str) -> dict:
    with open(ARTEFACTOS / f"{nombre}.json", encoding="utf-8") as f:
        return json.load(f)


if not (ARTEFACTOS / "scorecard.json").exists():
    st.error("Faltan los artefactos. Ejecuta `python -m scorecard.pipeline` y recarga.")
    st.stop()

definicion = leer_json("scorecard")
resumen = leer_json("resumen")
oot = resumen["oot"]
puntaje_oot = leer("puntaje_oot")
cortes = leer("cortes_oot")

st.title("Scorecard de riesgo de crédito")
st.caption(
    "Probabilidad de incumplimiento de préstamos de consumo de LendingClub, en puntos. "
    "Desarrollado con préstamos de 2007-2010 y validado fuera de tiempo con los de 2011. "
    "Proyecto de portafolio: no sirve para decidir créditos reales."
)

simulador, desempeno, calibracion, corte = st.tabs(
    ["Simulador", "Discriminación", "Calibración y estabilidad", "Punto de corte"]
)

# --- Simulador ----------------------------------------------------------------------

with simulador:
    izquierda, derecha = st.columns([1, 1.3], gap="large")
    with izquierda:
        st.subheader("Solicitud")
        calificacion = st.selectbox("Calificación de LendingClub", list("ABCDEFG"), index=1)
        plazo = st.radio("Plazo", [36, 60], format_func=lambda m: f"{m} meses", horizontal=True)
        proposito = st.selectbox("Propósito del préstamo", list(PROPOSITOS))
        ingreso = st.number_input(
            "Ingreso anual (USD)", min_value=0, max_value=2_000_000, value=60_000, step=5_000
        )
        estado = st.selectbox("Estado de residencia", list(ESTADOS))
        umbral = st.slider("Punto de corte (se aprueba con este puntaje o más)", 500, 580, 530, 5)

    solicitud = {
        "calificacion": calificacion,
        "plazo": plazo,
        "proposito": PROPOSITOS[proposito],
        "ingreso_anual": ingreso,
        "estado": ESTADOS[estado],
    }
    resultado = puntuar_solicitud(definicion, solicitud)
    puntaje = resultado["puntaje"]
    percentil = (puntaje_oot["puntaje"] < puntaje).mean()

    with derecha:
        st.subheader("Resultado")
        c1, c2, c3 = st.columns(3)
        c1.metric("Puntaje", f"{puntaje}")
        c2.metric("PD implícita", f"{resultado['pd']:.1%}")
        c3.metric("Decisión", "Aprobar" if puntaje >= umbral else "Rechazar")
        st.caption(
            f"Este puntaje supera al {percentil:.0%} de los préstamos de 2011. La PD sale "
            "de la escala de puntos (600 = odds 50:1, PDO 20) y no está recalibrada."
        )
        detalle = pd.DataFrame(resultado["detalle"])
        detalle["variable"] = detalle["variable"].map(ETIQUETAS)
        st.dataframe(detalle[["variable", "tramo", "puntos"]], hide_index=True, width=ANCHO)

        resultado_prestamo = puntaje_oot["incumple"].map({0: "Pagado", 1: "Incumplido"})
        histograma = (
            puntaje_oot.assign(resultado=resultado_prestamo)
            .groupby(["puntaje", "resultado"])
            .size()
            .rename("prestamos")
            .reset_index()
        )
        barras = (
            alt.Chart(histograma)
            .mark_bar(size=3)
            .encode(
                x=alt.X("puntaje:Q", title="Puntaje", scale=alt.Scale(zero=False)),
                y=alt.Y("prestamos:Q", title="Préstamos de 2011", stack=True),
                color=alt.Color(
                    "resultado:N",
                    title=None,
                    scale=alt.Scale(domain=["Pagado", "Incumplido"], range=[AZUL, NARANJA]),
                ),
                tooltip=["puntaje", "resultado", "prestamos"],
            )
        )
        regla = (
            alt.Chart(pd.DataFrame({"puntaje": [puntaje]}))
            .mark_rule(color="#0b0b0b", strokeWidth=2)
            .encode(x="puntaje:Q")
        )
        st.altair_chart((barras + regla).properties(height=220), width=ANCHO)
        st.caption("La línea negra marca la solicitud.")

# --- Discriminación -----------------------------------------------------------------

with desempeno:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC 2011", f"{oot['auc']:.3f}")
    c2.metric("Gini 2011", f"{oot['gini']:.3f}")
    c3.metric("KS 2011", f"{oot['ks']:.3f}")
    c4.metric("PSI del puntaje", f"{resumen['psi_puntaje_oot']:.3f}")
    st.caption(
        f"Intervalo de 95% del AUC por bootstrap: {oot['auc_ic_inferior']:.3f} a "
        f"{oot['auc_ic_superior']:.3f}."
    )

    comparacion = leer("comparacion_modelos")
    tabla = comparacion.pivot_table(index="modelo", columns="muestra", values="auc")
    tabla = tabla[["entrenamiento", "validacion", "oot"]].sort_values("oot", ascending=False)
    tabla.columns = ["AUC entrenamiento", "AUC validación", "AUC 2011"]
    st.dataframe(tabla.style.format("{:.3f}"), width=ANCHO)

    izquierda, derecha = st.columns(2, gap="large")
    with izquierda:
        roc = leer("roc_oot")
        modelos = list(dict.fromkeys(roc["modelo"]))
        grafico = (
            alt.Chart(roc)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("fpr:Q", title="Tasa de falsos positivos"),
                y=alt.Y("tpr:Q", title="Tasa de verdaderos positivos"),
                color=alt.Color(
                    "modelo:N",
                    title=None,
                    sort=modelos,
                    scale=alt.Scale(domain=modelos, range=[AZUL, NARANJA, AQUA, AMARILLO]),
                    legend=alt.Legend(
                        orient="bottom-right", fillColor="#ffffff", padding=6, labelLimit=260
                    ),
                ),
                tooltip=[
                    "modelo",
                    alt.Tooltip("fpr:Q", format=".2f"),
                    alt.Tooltip("tpr:Q", format=".2f"),
                ],
            )
            .properties(title="Curva ROC, 2011", height=320)
        )
        st.altair_chart(grafico, width=ANCHO)
    with derecha:
        deciles = leer("deciles_oot")
        grafico = (
            alt.Chart(deciles)
            .mark_bar(color=AZUL, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X(
                    "decil:O", title="Decil de riesgo (1 = mayor PD)", axis=alt.Axis(labelAngle=0)
                ),
                y=alt.Y(
                    "tasa_observada:Q", title="Tasa de incumplimiento", axis=alt.Axis(format="%")
                ),
                tooltip=[
                    "decil",
                    alt.Tooltip("tasa_observada:Q", format=".1%", title="Observada"),
                    alt.Tooltip("pd_promedio:Q", format=".1%", title="PD estimada"),
                    alt.Tooltip("captura_acumulada:Q", format=".0%", title="Captura acumulada"),
                ],
            )
            .properties(title="Incumplimiento por decil, 2011", height=320)
        )
        st.altair_chart(grafico, width=ANCHO)
    st.markdown(
        "El scorecard supera a la calificación de LendingClub sola y empata con LightGBM: "
        "el modelo lineal e interpretable no pierde poder predictivo."
    )

# --- Calibración y estabilidad ------------------------------------------------------

with calibracion:
    izquierda, derecha = st.columns(2, gap="large")
    with izquierda:
        cal = pd.concat(
            [
                leer("calibracion_validacion").assign(muestra="Validación (2007-2010)"),
                leer("calibracion_oot").assign(muestra="Fuera de tiempo (2011)"),
            ]
        )
        puntos = (
            alt.Chart(cal)
            .mark_circle(size=90, opacity=1, stroke="#fcfcfb", strokeWidth=1.5)
            .encode(
                x=alt.X("pd_promedio:Q", title="PD promedio estimada", axis=alt.Axis(format="%")),
                y=alt.Y("tasa_observada:Q", title="Tasa observada", axis=alt.Axis(format="%")),
                color=alt.Color(
                    "muestra:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["Validación (2007-2010)", "Fuera de tiempo (2011)"],
                        range=[AZUL, NARANJA],
                    ),
                    legend=alt.Legend(orient="bottom"),
                ),
                tooltip=[
                    "muestra",
                    "decil",
                    alt.Tooltip("pd_promedio:Q", format=".1%"),
                    alt.Tooltip("tasa_observada:Q", format=".1%"),
                ],
            )
        )
        diagonal = (
            alt.Chart(pd.DataFrame({"a": [0, 0.36], "b": [0, 0.36]}))
            .mark_line(color=GRIS, strokeDash=[4, 4], strokeWidth=1)
            .encode(x="a:Q", y="b:Q")
        )
        st.altair_chart(
            (diagonal + puntos).properties(title="Calibración por decil", height=340), width=ANCHO
        )
    with derecha:
        mon = leer("monitoreo_trimestral")
        largo = mon.melt(
            id_vars="trimestre",
            value_vars=["tasa_observada", "pd_promedio"],
            var_name="serie",
            value_name="valor",
        )
        largo["serie"] = largo["serie"].map(
            {"tasa_observada": "Tasa observada", "pd_promedio": "PD promedio estimada"}
        )
        grafico = (
            alt.Chart(largo)
            .mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2)
            .encode(
                x=alt.X("trimestre:O", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y(
                    "valor:Q",
                    title=None,
                    axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0.10, 0.20]),
                ),
                color=alt.Color(
                    "serie:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["Tasa observada", "PD promedio estimada"], range=[NARANJA, AZUL]
                    ),
                    legend=alt.Legend(orient="bottom"),
                ),
                tooltip=["trimestre", "serie", alt.Tooltip("valor:Q", format=".1%")],
            )
            .properties(title="PD estimada contra tasa observada, 2011", height=340)
        )
        st.altair_chart(grafico, width=ANCHO)

    recal = resumen["recalibracion"]
    st.markdown(
        f"**En 2011 el modelo subestima el riesgo:** PD promedio "
        f"{oot['pd_promedio']:.1%} contra {oot['tasa_observada']:.1%} observado "
        f"(Hosmer-Lemeshow p = {resumen['hosmer_lemeshow_oot']['p_valor']:.3f}). "
        f"Un ajuste del intercepto estimado con el primer semestre reduce el error por decil "
        f"del segundo semestre de {recal['error_calibracion_antes']:.1%} a "
        f"{recal['error_calibracion_despues']:.1%}."
    )

    izquierda, derecha = st.columns(2, gap="large")
    with izquierda:
        st.markdown("**PSI del puntaje** (referencia: entrenamiento 2007-2010)")
        st.dataframe(
            leer("psi").style.format({"psi_puntaje": "{:.3f}"}), hide_index=True, width=ANCHO
        )
    with derecha:
        st.markdown("**CSI por variable, 2011**")
        csi = leer("csi")
        csi["variable"] = csi["variable"].map(ETIQUETAS)
        st.dataframe(csi.style.format({"csi_oot": "{:.3f}"}), hide_index=True, width=ANCHO)
        st.caption("Menos de 0,10: estable. De 0,10 a 0,25: moderado. Más de 0,25: importante.")

# --- Punto de corte -----------------------------------------------------------------

with corte:
    elegido = st.select_slider("Punto de corte", options=cortes["corte"].tolist(), value=530)
    f = cortes.set_index("corte").loc[elegido]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tasa de aprobación", f"{f['tasa_aprobacion']:.1%}")
    c2.metric(
        "Incumplimiento de aprobados",
        f"{f['incumplimiento_aprobados']:.1%}",
        f"{f['incumplimiento_aprobados'] - oot['tasa_observada']:+.1%} contra la cartera",
        delta_color="inverse",
    )
    c3.metric("Malos evitados", f"{f['malos_evitados']:.1%}")
    c4.metric("Buenos rechazados", f"{f['buenos_rechazados']:.1%}")

    curva = (
        alt.Chart(cortes)
        .mark_line(color=AZUL, strokeWidth=2, point=alt.OverlayMarkDef(size=40, color=AZUL))
        .encode(
            x=alt.X("tasa_aprobacion:Q", title="Tasa de aprobación", axis=alt.Axis(format="%")),
            y=alt.Y(
                "incumplimiento_aprobados:Q",
                title="Incumplimiento de aprobados",
                axis=alt.Axis(format="%"),
            ),
            tooltip=[
                "corte",
                alt.Tooltip("tasa_aprobacion:Q", format=".1%"),
                alt.Tooltip("incumplimiento_aprobados:Q", format=".1%"),
            ],
        )
    )
    marca = (
        alt.Chart(cortes[cortes["corte"] == elegido])
        .mark_point(size=160, color=NARANJA, filled=True)
        .encode(x="tasa_aprobacion:Q", y="incumplimiento_aprobados:Q")
    )
    st.altair_chart(
        (curva + marca).properties(title="Aprobación contra riesgo, 2011", height=340),
        width=ANCHO,
    )
    st.caption(
        "El corte final depende de la rentabilidad y de la pérdida dado el incumplimiento, "
        "que esta base no permite estimar."
    )
