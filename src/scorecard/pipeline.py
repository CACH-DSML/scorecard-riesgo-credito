"""Entrena y valida todo con un comando, y guarda los resultados en `artefactos/`.

Uso:
    python -m scorecard.pipeline
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from scorecard import metricas as m
from scorecard import puntaje
from scorecard.datos import (
    BENCHMARK,
    CATEGORICAS,
    EVENTO,
    NUMERICAS,
    RAIZ,
    VARIABLES,
    Particion,
    cargar,
    particionar,
)
from scorecard.modelos import Benchmark, RetadorLightGBM, RetadorRedNeuronal, Scorecard
from scorecard.woe import AgrupadorWoE

DIR_ARTEFACTOS = RAIZ / "artefactos"
CANDIDATAS = VARIABLES + [BENCHMARK]
NOMBRES = {
    "scorecard": "Scorecard (logístico sobre WoE)",
    "scorecard_sin_calificacion": "Scorecard sin calificación",
    "benchmark": "Benchmark: calificación de LendingClub",
    "lightgbm": "Retador: LightGBM",
    "red_neuronal": "Retador: red neuronal",
}
OOT = "Fuera de tiempo (2011)"


@dataclass
class Resultados:
    particion: Particion
    exclusiones: pd.DataFrame
    agrupador: AgrupadorWoE
    scorecard: Scorecard
    modelos: dict
    tabla_puntos: pd.DataFrame

    def pd(self, modelo: str, df: pd.DataFrame) -> np.ndarray:
        return self.modelos[modelo].predecir(df)

    def puntaje(self, df: pd.DataFrame) -> pd.Series:
        return puntaje.puntuar(self.scorecard, df, self.tabla_puntos)


def entrenar() -> Resultados:
    df, exclusiones = cargar()
    particion = particionar(df)
    tr, va = particion.entrenamiento, particion.validacion

    agrupador = AgrupadorWoE(NUMERICAS, CATEGORICAS + [BENCHMARK]).ajustar(tr, tr[EVENTO])
    scorecard = Scorecard(agrupador, CANDIDATAS).ajustar(tr)
    modelos = {
        "scorecard": scorecard,
        "scorecard_sin_calificacion": Scorecard(agrupador, VARIABLES).ajustar(tr),
        "benchmark": Benchmark().ajustar(tr),
        "lightgbm": RetadorLightGBM(CANDIDATAS).ajustar(tr, va),
        "red_neuronal": RetadorRedNeuronal(CANDIDATAS).ajustar(tr),
    }
    return Resultados(
        particion, exclusiones, agrupador, scorecard, modelos, puntaje.tabla_puntos(scorecard)
    )


# --- Tablas de validación -----------------------------------------------------------


def comparacion_modelos(r: Resultados) -> pd.DataFrame:
    filas = []
    partes = {
        "entrenamiento": r.particion.entrenamiento,
        "validacion": r.particion.validacion,
        "oot": r.particion.oot,
    }
    for clave, modelo in r.modelos.items():
        for parte, df in partes.items():
            pd_ = modelo.predecir(df)
            fila = {"modelo": NOMBRES[clave], "clave": clave, "muestra": parte}
            fila.update(m.resumen(df[EVENTO], pd_))
            if parte == "oot":
                fila["auc_ic_inferior"], fila["auc_ic_superior"] = m.intervalo_auc(df[EVENTO], pd_)
            filas.append(fila)
    return pd.DataFrame(filas)


def diferencias_oot(r: Resultados) -> pd.DataFrame:
    oot = r.particion.oot
    base = r.pd("scorecard", oot)
    filas = []
    for clave in ["benchmark", "scorecard_sin_calificacion", "lightgbm", "red_neuronal"]:
        fila = {"comparacion": f"Scorecard menos {NOMBRES[clave]}"}
        fila.update(m.diferencia_auc(oot[EVENTO], base, r.pd(clave, oot)))
        filas.append(fila)
    return pd.DataFrame(filas)


def recalibracion(r: Resultados) -> dict:
    """Ajuste de tendencia central: se corrige solo el intercepto.

    El desplazamiento se estima con los préstamos de 2011T1-T2 y se evalúa con los de
    2011T3-T4, para no medir el ajuste con los mismos datos con que se estimó.
    """
    oot = r.particion.oot
    primera = oot[oot["trimestre"].isin(["2011Q1", "2011Q2"])]
    segunda = oot[oot["trimestre"].isin(["2011Q3", "2011Q4"])]
    logit_primera = r.scorecard.logit(primera)

    # Desplazamiento que iguala la PD promedio a la tasa observada (bisección).
    objetivo = primera[EVENTO].mean()
    bajo, alto = -2.0, 2.0
    for _ in range(60):
        medio = (bajo + alto) / 2
        if (1 / (1 + np.exp(-(logit_primera + medio)))).mean() < objetivo:
            bajo = medio
        else:
            alto = medio
    desplazamiento = (bajo + alto) / 2

    logit_segunda = r.scorecard.logit(segunda)
    antes = 1 / (1 + np.exp(-logit_segunda))
    despues = 1 / (1 + np.exp(-(logit_segunda + desplazamiento)))
    y = segunda[EVENTO]
    return {
        "desplazamiento_intercepto": desplazamiento,
        "tasa_observada_estimacion": float(objetivo),
        "tasa_observada_evaluacion": float(y.mean()),
        "pd_promedio_antes": float(antes.mean()),
        "pd_promedio_despues": float(despues.mean()),
        "error_calibracion_antes": m.error_calibracion(y, antes),
        "error_calibracion_despues": m.error_calibracion(y, despues),
        "hosmer_lemeshow_p_antes": m.hosmer_lemeshow(y, antes)["p_valor"],
        "hosmer_lemeshow_p_despues": m.hosmer_lemeshow(y, despues)["p_valor"],
    }


def estabilidad(r: Resultados) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PSI del puntaje y CSI (PSI por variable, sobre sus tramos) contra entrenamiento."""
    tr = r.particion.entrenamiento
    referencia = r.puntaje(tr)
    comparaciones = {"Validación (2007-2010)": r.particion.validacion, OOT: r.particion.oot}
    for trimestre, grupo in r.particion.oot.groupby("trimestre"):
        comparaciones[trimestre] = grupo
    psi = pd.DataFrame(
        [
            {"muestra": nombre, "psi_puntaje": m.psi(referencia, r.puntaje(df))}
            for nombre, df in comparaciones.items()
        ]
    )
    psi["lectura"] = psi["psi_puntaje"].map(m.lectura_psi)

    filas = []
    for v in r.scorecard.variables:
        t = r.agrupador.tramos[v]
        valor = m.psi_categorico(t.etiquetas(tr[v]), t.etiquetas(r.particion.oot[v]))
        filas.append({"variable": v, "csi_oot": valor, "lectura": m.lectura_psi(valor)})
    return psi, pd.DataFrame(filas)


def monitoreo_trimestral(r: Resultados) -> pd.DataFrame:
    referencia = r.puntaje(r.particion.entrenamiento)
    filas = []
    for trimestre, g in r.particion.oot.groupby("trimestre"):
        pd_ = r.pd("scorecard", g)
        filas.append(
            {
                "trimestre": trimestre,
                "prestamos": len(g),
                "tasa_observada": g[EVENTO].mean(),
                "pd_promedio": pd_.mean(),
                "auc": m.auc(g[EVENTO], pd_),
                "ks": m.ks(g[EVENTO], pd_),
                "psi_puntaje": m.psi(referencia, r.puntaje(g)),
                "participacion_60_meses": (g["plazo"] == 60).mean(),
            }
        )
    return pd.DataFrame(filas)


def importancia(r: Resultados) -> pd.DataFrame:
    """Scorecard: |coeficiente| x desviación estándar del WoE en entrenamiento.
    LightGBM: promedio de |SHAP| en la muestra fuera de tiempo."""
    tr = r.particion.entrenamiento
    woe = r.agrupador.transformar(tr, r.scorecard.variables).rename(columns=lambda c: c[4:])
    contribucion = r.scorecard.coeficientes.abs() * woe.std()
    scorecard = (contribucion / contribucion.sum()).rename("scorecard")

    lgbm = r.modelos["lightgbm"]
    X = lgbm.matriz(r.particion.oot)
    valores = shap.TreeExplainer(lgbm.modelo).shap_values(X)
    if isinstance(valores, list):
        valores = valores[1]
    shap_medio = pd.Series(np.abs(valores).mean(axis=0), index=X.columns)
    lightgbm = (shap_medio / shap_medio.sum()).rename("lightgbm_shap")
    tabla = pd.concat([scorecard, lightgbm], axis=1).fillna(0.0)
    tabla = tabla.sort_values("lightgbm_shap", ascending=False)
    return tabla.rename_axis("variable").reset_index()


def tabla_tramos(r: Resultados) -> pd.DataFrame:
    partes = []
    for v, t in r.agrupador.tramos.items():
        tabla = t.tabla.rename_axis("tramo").reset_index()
        tabla.insert(0, "variable", v)
        partes.append(tabla)
    return pd.concat(partes, ignore_index=True)


def curvas(r: Resultados) -> tuple[pd.DataFrame, pd.DataFrame]:
    oot = r.particion.oot
    roc, ks = [], []
    for clave in ["scorecard", "benchmark", "lightgbm", "red_neuronal"]:
        pd_ = r.pd(clave, oot)
        roc.append(m.curva_roc(oot[EVENTO], pd_).assign(modelo=NOMBRES[clave]))
        ks.append(m.curva_ks(oot[EVENTO], pd_).assign(modelo=NOMBRES[clave]))
    return pd.concat(roc), pd.concat(ks)


def resumen_global(r: Resultados, comparacion: pd.DataFrame, psi: pd.DataFrame) -> dict:
    oot = r.particion.oot
    y_oot, pd_oot = oot[EVENTO], r.pd("scorecard", oot)
    puntaje_oot = r.puntaje(oot)
    fila = comparacion.query("clave == 'scorecard' and muestra == 'oot'").iloc[0]
    claves = ["auc", "gini", "ks", "brier", "auc_ic_inferior", "auc_ic_superior"]
    claves += ["tasa_observada", "pd_promedio"]
    return {
        "variables": r.scorecard.variables,
        "oot": {k: float(fila[k]) for k in claves},
        "error_calibracion_oot": m.error_calibracion(y_oot, pd_oot),
        "hosmer_lemeshow_oot": m.hosmer_lemeshow(y_oot, pd_oot),
        "psi_puntaje_oot": float(psi.loc[psi["muestra"] == OOT, "psi_puntaje"].iloc[0]),
        "recalibracion": recalibracion(r),
        "puntaje_oot": {
            "minimo": int(puntaje_oot.min()),
            "mediana": float(puntaje_oot.median()),
            "maximo": int(puntaje_oot.max()),
        },
    }


# --- Ejecución completa -------------------------------------------------------------


def main() -> int:
    DIR_ARTEFACTOS.mkdir(exist_ok=True)
    print("Entrenando...")
    r = entrenar()
    oot = r.particion.oot
    y_oot, pd_oot = oot[EVENTO], r.pd("scorecard", oot)
    puntaje_oot = r.puntaje(oot)

    comparacion = comparacion_modelos(r)
    psi, csi = estabilidad(r)
    tablas = {
        "particion": r.particion.resumen(),
        "exclusiones": r.exclusiones,
        "iv": r.agrupador.tabla_iv(),
        "tramos": tabla_tramos(r),
        "seleccion": r.scorecard.bitacora,
        "coeficientes": r.scorecard.tabla_coeficientes().rename_axis("variable").reset_index(),
        "tabla_puntos": r.tabla_puntos,
        "comparacion_modelos": comparacion,
        "diferencias_auc_oot": diferencias_oot(r),
        "deciles_oot": m.tabla_deciles(y_oot, pd_oot),
        "calibracion_oot": m.tabla_calibracion(y_oot, pd_oot),
        "calibracion_validacion": m.tabla_calibracion(
            r.particion.validacion[EVENTO], r.pd("scorecard", r.particion.validacion)
        ),
        "psi": psi,
        "csi": csi,
        "monitoreo_trimestral": monitoreo_trimestral(r),
        "cortes_oot": puntaje.tabla_cortes(y_oot, puntaje_oot),
        "importancia": importancia(r),
        "puntaje_oot": pd.DataFrame({"puntaje": puntaje_oot, EVENTO: y_oot}),
    }
    tablas["roc_oot"], tablas["ks_oot"] = curvas(r)
    for nombre, tabla in tablas.items():
        tabla.to_csv(DIR_ARTEFACTOS / f"{nombre}.csv", index=False)

    definicion = puntaje.exportar(r.scorecard, r.tabla_puntos)
    puntaje.guardar(definicion, DIR_ARTEFACTOS / "scorecard.json")
    resumen = resumen_global(r, comparacion, psi)
    with open(DIR_ARTEFACTOS / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    print(r.particion.resumen().to_string(index=False))
    print("\nVariables del scorecard:", ", ".join(r.scorecard.variables))
    print("\nModelos en la muestra fuera de tiempo (2011):")
    columnas = ["modelo", "auc", "gini", "ks", "brier"]
    print(comparacion.query("muestra == 'oot'")[columnas].round(3).to_string(index=False))
    print(f"\nPSI del puntaje 2007-2010 contra 2011: {resumen['psi_puntaje_oot']:.3f}")
    print(f"Artefactos en {DIR_ARTEFACTOS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
