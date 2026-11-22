"""Escalamiento a puntos y análisis de punto de corte.

Escala estándar de la industria: 600 puntos equivalen a odds de 50 buenos por cada malo, y
cada 20 puntos (PDO, points to double the odds) se duplican esas odds. Más puntos, menos
riesgo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from scorecard.woe import TramosVariable

if TYPE_CHECKING:  # evita cargar LightGBM y statsmodels en la aplicación
    from scorecard.modelos import Scorecard

PUNTAJE_BASE = 600
ODDS_BASE = 50
PDO = 20


@dataclass(frozen=True)
class Escala:
    base: float = PUNTAJE_BASE
    odds_base: float = ODDS_BASE
    pdo: float = PDO

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.base - self.factor * np.log(self.odds_base)

    def puntaje_desde_pd(self, pd_) -> np.ndarray:
        pd_ = np.clip(np.asarray(pd_, float), 1e-9, 1 - 1e-9)
        return self.offset + self.factor * np.log((1 - pd_) / pd_)

    def pd_desde_puntaje(self, puntaje) -> np.ndarray:
        log_odds_buenos = (np.asarray(puntaje, float) - self.offset) / self.factor
        return 1 / (1 + np.exp(log_odds_buenos))


ESCALA = Escala()


def _puntos(scorecard: Scorecard, variable: str, woe: float, escala: Escala) -> float:
    k = len(scorecard.variables)
    b = scorecard.coeficientes[variable]
    return -(b * woe + scorecard.intercepto / k) * escala.factor + escala.offset / k


def tabla_puntos(scorecard: Scorecard, escala: Escala = ESCALA) -> pd.DataFrame:
    """Puntos por tramo de cada variable, redondeados a enteros.

    puntos = -(b_j * WoE + a / k) * factor + offset / k, con a el intercepto, b_j el
    coeficiente de la variable y k el número de variables.
    """
    filas = []
    for v in scorecard.variables:
        for tramo, fila in scorecard.agrupador.tramos[v].tabla.iterrows():
            filas.append(
                {
                    "variable": v,
                    "tramo": str(tramo),
                    "n_entrenamiento": int(fila["n"]),
                    "tasa_incumplimiento": float(fila["tasa_incumplimiento"]),
                    "woe": float(fila["woe"]),
                    "puntos": int(round(_puntos(scorecard, v, fila["woe"], escala))),
                }
            )
    return pd.DataFrame(filas)


def puntos_desconocido(scorecard: Scorecard, variable: str, escala: Escala = ESCALA) -> int:
    """Puntos de un tramo no visto en entrenamiento: los de WoE 0 (riesgo promedio)."""
    return int(round(_puntos(scorecard, variable, 0.0, escala)))


def puntuar(scorecard: Scorecard, df: pd.DataFrame, tabla: pd.DataFrame) -> pd.Series:
    """Puntaje entero de cada préstamo: suma de los puntos de sus tramos."""
    total = pd.Series(0, index=df.index, dtype=int)
    for v in scorecard.variables:
        puntos = tabla[tabla["variable"] == v].set_index("tramo")["puntos"]
        etiquetas = scorecard.agrupador.tramos[v].etiquetas(df[v]).astype(str)
        total += etiquetas.map(puntos).fillna(puntos_desconocido(scorecard, v)).astype(int)
    return total


def tabla_cortes(y, puntaje, cortes: list[int] | None = None) -> pd.DataFrame:
    """Para cada punto de corte: se aprueba si el puntaje es mayor o igual al corte."""
    y = np.asarray(y)
    puntaje = np.asarray(puntaje)
    if cortes is None:
        desde, hasta = np.percentile(puntaje, [5, 95]).astype(int)
        cortes = list(range(desde - desde % 5, hasta + 1, 5))
    filas = []
    for c in cortes:
        aprobado = puntaje >= c
        filas.append(
            {
                "corte": c,
                "tasa_aprobacion": aprobado.mean(),
                "incumplimiento_aprobados": y[aprobado].mean() if aprobado.any() else np.nan,
                "incumplimiento_rechazados": y[~aprobado].mean() if (~aprobado).any() else np.nan,
                "malos_evitados": y[~aprobado].sum() / y.sum(),
                "buenos_rechazados": (1 - y[~aprobado]).sum() / (1 - y).sum(),
            }
        )
    return pd.DataFrame(filas)


def exportar(scorecard: Scorecard, tabla: pd.DataFrame, escala: Escala = ESCALA) -> dict:
    """Todo lo necesario para puntuar una solicitud sin el código de entrenamiento."""
    variables = {}
    for v in scorecard.variables:
        t = scorecard.agrupador.tramos[v]
        variables[v] = {
            "tipo": t.tipo,
            "cortes": t.cortes,
            "grupos": t.grupos,
            "puntos": tabla[tabla["variable"] == v].set_index("tramo")["puntos"].to_dict(),
            "puntos_desconocido": puntos_desconocido(scorecard, v, escala),
        }
    return {
        "escala": {"base": escala.base, "odds_base": escala.odds_base, "pdo": escala.pdo},
        "variables": variables,
    }


def guardar(definicion: dict, ruta) -> None:
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(definicion, f, ensure_ascii=False, indent=2)


def cargar_definicion(ruta) -> dict:
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def puntuar_solicitud(definicion: dict, solicitud: dict) -> dict:
    """Puntúa una solicitud con la definición exportada, sin reentrenar nada.

    Devuelve el tramo y los puntos de cada variable, el puntaje total y la PD implícita.
    """
    detalle = []
    for v, d in definicion["variables"].items():
        tramos = TramosVariable(nombre=v, tipo=d["tipo"], cortes=d["cortes"], grupos=d["grupos"])
        valor = solicitud.get(v)
        serie = pd.Series([np.nan if valor is None else valor], dtype=object)
        if d["tipo"] == "numerica":
            serie = pd.to_numeric(serie, errors="coerce")
        tramo = str(tramos.etiquetas(serie).iloc[0])
        puntos = d["puntos"].get(tramo, d["puntos_desconocido"])
        detalle.append({"variable": v, "valor": valor, "tramo": tramo, "puntos": int(puntos)})
    total = sum(f["puntos"] for f in detalle)
    escala = Escala(**definicion["escala"])
    return {
        "detalle": detalle,
        "puntaje": total,
        "pd": float(escala.pd_desde_puntaje(total)),
    }
