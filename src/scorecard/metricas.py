"""Métricas de validación: discriminación, calibración y estabilidad."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def auc(y, pd_) -> float:
    return float(roc_auc_score(y, pd_))


def gini(y, pd_) -> float:
    return 2 * auc(y, pd_) - 1


def ks(y, pd_) -> float:
    """Máxima distancia entre las distribuciones acumuladas de malos y buenos."""
    fpr, tpr, _ = roc_curve(y, pd_)
    return float(np.max(tpr - fpr))


def resumen(y, pd_) -> dict:
    y = np.asarray(y)
    pd_ = np.asarray(pd_)
    return {
        "auc": auc(y, pd_),
        "gini": gini(y, pd_),
        "ks": ks(y, pd_),
        "brier": float(brier_score_loss(y, pd_)),
        "tasa_observada": float(y.mean()),
        "pd_promedio": float(pd_.mean()),
    }


def intervalo_auc(y, pd_, repeticiones: int = 1000, semilla: int = 42) -> tuple[float, float]:
    """Intervalo de 95% del AUC por bootstrap."""
    y, pd_ = np.asarray(y), np.asarray(pd_)
    rng = np.random.default_rng(semilla)
    valores = []
    for _ in range(repeticiones):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        valores.append(roc_auc_score(y[i], pd_[i]))
    return float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))


def diferencia_auc(y, pd_a, pd_b, repeticiones: int = 1000, semilla: int = 42) -> dict:
    """AUC(a) - AUC(b) con intervalo de 95% por bootstrap pareado."""
    y, pd_a, pd_b = np.asarray(y), np.asarray(pd_a), np.asarray(pd_b)
    rng = np.random.default_rng(semilla)
    difs = []
    for _ in range(repeticiones):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        difs.append(roc_auc_score(y[i], pd_a[i]) - roc_auc_score(y[i], pd_b[i]))
    difs = np.array(difs)
    return {
        "diferencia": auc(y, pd_a) - auc(y, pd_b),
        "ic_inferior": float(np.percentile(difs, 2.5)),
        "ic_superior": float(np.percentile(difs, 97.5)),
        "prob_mayor": float((difs > 0).mean()),
    }


def tabla_deciles(y, pd_, grupos: int = 10) -> pd.DataFrame:
    """Deciles de riesgo: el 1 es el de mayor PD. Incluye captura acumulada y lift."""
    df = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(pd_)})
    rango = df["pd"].rank(method="first", ascending=False)
    df["decil"] = pd.qcut(rango, grupos, labels=False) + 1
    tabla = df.groupby("decil").agg(
        n=("y", "size"),
        malos=("y", "sum"),
        pd_promedio=("pd", "mean"),
        pd_minima=("pd", "min"),
        pd_maxima=("pd", "max"),
    )
    tabla["tasa_observada"] = tabla["malos"] / tabla["n"]
    tabla["captura_acumulada"] = tabla["malos"].cumsum() / tabla["malos"].sum()
    tabla["lift"] = tabla["tasa_observada"] / df["y"].mean()
    return tabla.reset_index()


def tabla_calibracion(y, pd_, grupos: int = 10) -> pd.DataFrame:
    """PD estimada contra tasa observada por decil, con prueba binomial de cada decil.

    El p-valor contrasta si la tasa observada es compatible con la PD promedio del decil.
    """
    tabla = tabla_deciles(y, pd_, grupos)
    tabla["diferencia"] = tabla["tasa_observada"] - tabla["pd_promedio"]
    tabla["p_valor_binomial"] = [
        stats.binomtest(int(m), int(n), float(p)).pvalue
        for m, n, p in zip(tabla["malos"], tabla["n"], tabla["pd_promedio"], strict=True)
    ]
    columnas = ["decil", "n", "malos", "pd_promedio", "tasa_observada", "diferencia"]
    return tabla[[*columnas, "p_valor_binomial"]]


def error_calibracion(y, pd_, grupos: int = 10) -> float:
    """Diferencia absoluta promedio entre PD y tasa observada, ponderada por decil."""
    t = tabla_calibracion(y, pd_, grupos)
    return float(np.average(np.abs(t["diferencia"]), weights=t["n"]))


def hosmer_lemeshow(y, pd_, grupos: int = 10) -> dict:
    t = tabla_deciles(y, pd_, grupos)
    esperados = t["pd_promedio"] * t["n"]
    varianza = esperados * (1 - t["pd_promedio"])
    estadistico = float((((t["malos"] - esperados) ** 2) / varianza).sum())
    return {"estadistico": estadistico, "p_valor": float(stats.chi2.sf(estadistico, grupos - 2))}


def psi(esperado, actual, grupos: int = 10) -> float:
    """Population Stability Index con tramos definidos por los deciles de la base esperada.

    Lectura usual: menos de 0,10 estable; de 0,10 a 0,25 cambio moderado; más de 0,25,
    cambio importante.
    """
    esperado, actual = np.asarray(esperado, float), np.asarray(actual, float)
    bordes = np.unique(np.quantile(esperado, np.linspace(0, 1, grupos + 1)))
    bordes[0], bordes[-1] = -np.inf, np.inf
    e = np.histogram(esperado, bordes)[0] / len(esperado)
    a = np.histogram(actual, bordes)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorico(esperado: pd.Series, actual: pd.Series) -> float:
    """PSI de una variable ya agrupada en tramos (por ejemplo, las etiquetas de WoE)."""
    e = esperado.value_counts(normalize=True)
    a = actual.value_counts(normalize=True)
    tramos = e.index.union(a.index)
    e = e.reindex(tramos, fill_value=0).clip(lower=1e-6)
    a = a.reindex(tramos, fill_value=0).clip(lower=1e-6)
    return float(((a - e) * np.log(a / e)).sum())


def lectura_psi(valor: float) -> str:
    if valor < 0.10:
        return "Estable"
    if valor < 0.25:
        return "Cambio moderado"
    return "Cambio importante"


def curva_roc(y, pd_, puntos: int = 101) -> pd.DataFrame:
    fpr, tpr, _ = roc_curve(y, pd_)
    malla = np.linspace(0, 1, puntos)
    return pd.DataFrame({"fpr": malla, "tpr": np.interp(malla, fpr, tpr)})


def curva_ks(y, pd_, puntos: int = 101) -> pd.DataFrame:
    """Distribución acumulada de malos y buenos según la población ordenada por riesgo."""
    df = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(pd_)})
    df = df.sort_values("pd", ascending=False)
    poblacion = np.arange(1, len(df) + 1) / len(df)
    malos = df["y"].cumsum().values / df["y"].sum()
    buenos = (1 - df["y"]).cumsum().values / (1 - df["y"]).sum()
    malla = np.linspace(0, 1, puntos)
    return pd.DataFrame(
        {
            "poblacion": malla,
            "malos_acumulados": np.interp(malla, poblacion, malos),
            "buenos_acumulados": np.interp(malla, poblacion, buenos),
        }
    )
