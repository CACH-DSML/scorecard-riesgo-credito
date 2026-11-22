"""Pruebas con datos sintéticos: no dependen de la base real."""

import numpy as np
import pandas as pd
import pytest

from scorecard import metricas as m
from scorecard.puntaje import Escala
from scorecard.woe import FALTANTE, OTRAS, ajustar_variable


@pytest.fixture
def sintetico():
    rng = np.random.default_rng(0)
    n = 5000
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2 + 1.0 * x)))).astype(int)
    return pd.Series(x), pd.Series(y)


def test_woe_numerico_es_monotono_y_respeta_tamano_minimo(sintetico):
    x, y = sintetico
    tramos = ajustar_variable(x, y, "x", "numerica")
    tabla = tramos.tabla.drop(index=FALTANTE, errors="ignore")
    tasas = tabla["tasa_incumplimiento"].values
    assert np.all(np.diff(tasas) > 0)  # el riesgo sube con x
    assert (tabla["participacion"] >= 0.05).all()
    assert tramos.iv > 0.1


def test_woe_positivo_significa_menos_riesgo(sintetico):
    x, y = sintetico
    tabla = ajustar_variable(x, y, "x", "numerica").tabla
    menor_riesgo = tabla["tasa_incumplimiento"].idxmin()
    assert tabla.loc[menor_riesgo, "woe"] == tabla["woe"].max()


def test_categorias_pequenas_y_nuevas():
    x = pd.Series(["a"] * 600 + ["b"] * 390 + ["c"] * 10)
    y = pd.Series([0, 1] * 500)
    tramos = ajustar_variable(x, y, "x", "categorica")
    assert tramos.grupos["c"] == OTRAS
    # Una categoría que no se vio en entrenamiento cae en OTRAS.
    assert tramos.etiquetas(pd.Series(["z"])).iloc[0] == OTRAS


def test_faltantes_escasos_reciben_woe_cero():
    x = pd.Series([1.0] * 500 + [2.0] * 495 + [np.nan] * 5)
    y = pd.Series([0] * 450 + [1] * 50 + [0] * 400 + [1] * 95 + [1] * 5)
    tramos = ajustar_variable(x, y, "x", "numerica")
    assert tramos.woe[FALTANTE] == 0.0


def test_metricas_en_casos_extremos():
    y = np.array([0, 0, 1, 1])
    perfecto = [0.1, 0.2, 0.8, 0.9]
    assert m.auc(y, perfecto) == 1.0
    assert m.ks(y, perfecto) == 1.0
    assert m.gini(y, perfecto) == 1.0
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 20000)
    assert abs(m.auc(y, rng.random(20000)) - 0.5) < 0.02


def test_psi():
    rng = np.random.default_rng(2)
    base = rng.normal(size=10000)
    assert m.psi(base, rng.normal(size=10000)) < 0.01
    assert m.psi(base, rng.normal(loc=1.0, size=10000)) > 0.25


def test_tabla_deciles():
    rng = np.random.default_rng(3)
    pd_ = rng.random(1000)
    y = (rng.random(1000) < pd_).astype(int)
    tabla = m.tabla_deciles(y, pd_)
    assert tabla["n"].sum() == 1000
    assert tabla["captura_acumulada"].iloc[-1] == pytest.approx(1.0)
    assert tabla["pd_promedio"].is_monotonic_decreasing  # el decil 1 es el más riesgoso


def test_escala_de_puntos():
    escala = Escala(base=600, odds_base=50, pdo=20)
    pd_base = 1 / 51  # odds de 50 buenos por cada malo
    assert escala.puntaje_desde_pd(pd_base) == pytest.approx(600)
    assert escala.puntaje_desde_pd(1 / 101) == pytest.approx(620)  # odds dobles: +20 puntos
    assert escala.pd_desde_puntaje(600) == pytest.approx(pd_base)
