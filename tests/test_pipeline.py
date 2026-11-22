"""Pruebas con la base real: partición, fugas de información, signos y reproducibilidad."""

import numpy as np
import pytest

from scorecard import metricas as m
from scorecard import puntaje
from scorecard.datos import BENCHMARK, CATEGORICAS, EVENTO, NUMERICAS, VARIABLES
from scorecard.modelos import P_VALOR_MAXIMO, Scorecard
from scorecard.pipeline import DIR_ARTEFACTOS, entrenar
from scorecard.woe import AgrupadorWoE


@pytest.fixture(scope="module")
def r():
    return entrenar()


def test_particion_fuera_de_tiempo(r):
    p = r.particion
    assert p.entrenamiento["anio"].max() <= 2010
    assert p.validacion["anio"].max() <= 2010
    assert set(p.oot["anio"]) == {2011}
    ids = [set(p.entrenamiento["id"]), set(p.validacion["id"]), set(p.oot["id"])]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert len(p.entrenamiento) + len(p.validacion) + len(p.oot) == 19905


def test_no_se_usan_variables_fijadas_por_el_modelo_de_lendingclub(r):
    # La tasa de interés se fija con el modelo de riesgo de LendingClub: no es un insumo.
    assert "tasa" not in VARIABLES
    assert "tasa" not in r.scorecard.variables
    assert set(r.scorecard.variables) <= set(VARIABLES) | {BENCHMARK}


def test_coeficientes_con_signo_esperado_y_significativos(r):
    assert (r.scorecard.coeficientes < 0).all()
    pvalores = r.scorecard.resultado.pvalues.drop("const")
    assert (pvalores <= P_VALOR_MAXIMO).all()


def test_con_el_evento_barajado_no_hay_discriminacion(r):
    tr = r.particion.entrenamiento.copy()
    tr[EVENTO] = np.random.default_rng(0).permutation(tr[EVENTO].values)
    agrupador = AgrupadorWoE(NUMERICAS, CATEGORICAS + [BENCHMARK]).ajustar(tr, tr[EVENTO])
    barajado = Scorecard(agrupador, VARIABLES + [BENCHMARK]).ajustar(tr)
    oot = r.particion.oot
    if barajado.variables:
        assert m.auc(oot[EVENTO], barajado.predecir(oot)) < 0.56


def test_puntaje_por_puntos_reproduce_la_pd_del_modelo(r):
    oot = r.particion.oot.head(500)
    desde_puntos = r.puntaje(oot)
    desde_pd = puntaje.ESCALA.puntaje_desde_pd(r.pd("scorecard", oot))
    # Cada variable se redondea a un entero: el error máximo es 0,5 por variable.
    assert np.abs(desde_puntos - desde_pd).max() <= 0.5 * len(r.scorecard.variables) + 1e-9


def test_la_definicion_exportada_puntua_igual_que_el_modelo(r):
    definicion = puntaje.exportar(r.scorecard, r.tabla_puntos)
    oot = r.particion.oot.head(200)
    esperado = r.puntaje(oot).tolist()
    obtenido = [
        puntaje.puntuar_solicitud(definicion, fila)["puntaje"]
        for fila in oot.to_dict(orient="records")
    ]
    assert obtenido == esperado


def test_los_artefactos_publicados_coinciden_con_el_entrenamiento(r):
    ruta = DIR_ARTEFACTOS / "scorecard.json"
    if not ruta.exists():
        pytest.skip("No hay artefactos: ejecuta python -m scorecard.pipeline")
    publicado = puntaje.cargar_definicion(ruta)
    actual = puntaje.exportar(r.scorecard, r.tabla_puntos)
    assert publicado["variables"].keys() == actual["variables"].keys()
    for v in actual["variables"]:
        assert publicado["variables"][v]["puntos"] == actual["variables"][v]["puntos"]


def test_el_scorecard_supera_al_benchmark_fuera_de_tiempo(r):
    oot = r.particion.oot
    diferencia = m.diferencia_auc(oot[EVENTO], r.pd("scorecard", oot), r.pd("benchmark", oot))
    assert diferencia["ic_inferior"] > 0
