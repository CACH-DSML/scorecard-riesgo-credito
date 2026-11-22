"""Agrupación en tramos, Weight of Evidence (WoE) e Information Value (IV).

Convención: WoE = ln(% de buenos / % de malos). Un WoE positivo indica menor riesgo que
el promedio. Los tramos se ajustan solo con la muestra de entrenamiento.

Reglas de agrupación:
- Variables numéricas: se parte de cuantiles y se fusionan tramos vecinos hasta que cada
  tramo tenga al menos 5% de la muestra y la tasa de incumplimiento sea monótona.
- Variables categóricas: las categorías con menos de 2% de la muestra van a "OTRAS".
- Los faltantes forman un tramo propio. Si son menos de 1% de la muestra, reciben WoE 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FALTANTE = "Faltante"
OTRAS = "OTRAS"
SUAVIZADO = 0.5
MIN_FALTANTES = 0.01


def _woe(buenos: np.ndarray, malos: np.ndarray, total_buenos: float, total_malos: float):
    pb = (buenos + SUAVIZADO) / (total_buenos + SUAVIZADO * len(buenos))
    pm = (malos + SUAVIZADO) / (total_malos + SUAVIZADO * len(malos))
    woe = np.log(pb / pm)
    iv = (pb - pm) * woe
    return woe, iv


@dataclass
class TramosVariable:
    """Tramos de una variable con su WoE."""

    nombre: str
    tipo: str  # "numerica" o "categorica"
    cortes: list[float] = field(default_factory=list)  # bordes internos, tramos (a, b]
    grupos: dict[str, str] = field(default_factory=dict)  # categoría -> grupo
    woe: dict[str, float] = field(default_factory=dict)  # etiqueta de tramo -> WoE
    tabla: pd.DataFrame | None = None

    @property
    def iv(self) -> float:
        return float(self.tabla["iv"].sum()) if self.tabla is not None else 0.0

    def etiquetas(self, x: pd.Series) -> pd.Series:
        """Etiqueta de tramo de cada observación."""
        if self.tipo == "numerica":
            bordes = [-np.inf, *self.cortes, np.inf]
            nombres = _nombres_tramos(bordes)
            tramo = pd.cut(x.astype(float), bins=bordes, labels=nombres, right=True)
            return tramo.astype(object).where(x.notna(), FALTANTE)
        valores = x.astype(object).where(x.notna(), FALTANTE)
        return valores.map(lambda v: self.grupos.get(v, OTRAS if v != FALTANTE else FALTANTE))

    def transformar(self, x: pd.Series) -> pd.Series:
        """WoE de cada observación. Un tramo no visto en entrenamiento recibe WoE 0."""
        return self.etiquetas(x).map(self.woe).fillna(0.0).astype(float)


def _nombres_tramos(bordes: list[float]) -> list[str]:
    def fmt(v: float) -> str:
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.4g}"

    nombres = []
    for a, b in zip(bordes[:-1], bordes[1:], strict=True):
        if np.isinf(a):
            nombres.append(f"<= {fmt(b)}")
        elif np.isinf(b):
            nombres.append(f"> {fmt(a)}")
        else:
            nombres.append(f"({fmt(a)}, {fmt(b)}]")
    return nombres


def _conteos(etiqueta: pd.Series, y: pd.Series, orden: list[str]) -> pd.DataFrame:
    tabla = (
        pd.DataFrame({"tramo": etiqueta.values, "y": y.values})
        .groupby("tramo", sort=False)["y"]
        .agg(n="count", malos="sum")
        .reindex(orden)
        .dropna(subset=["n"])
    )
    tabla["buenos"] = tabla["n"] - tabla["malos"]
    return tabla


def _completar_tabla(tabla: pd.DataFrame, total_buenos: float, total_malos: float):
    woe, iv = _woe(tabla["buenos"].values, tabla["malos"].values, total_buenos, total_malos)
    tabla = tabla.copy()
    tabla["participacion"] = tabla["n"] / tabla["n"].sum()
    tabla["tasa_incumplimiento"] = tabla["malos"] / tabla["n"]
    tabla["woe"] = woe
    tabla["iv"] = iv
    return tabla


def _ajustar_numerica(
    x: pd.Series, y: pd.Series, max_tramos: int, min_participacion: float
) -> list[float]:
    """Cortes internos de una variable numérica."""
    presentes = x.notna()
    xv, yv = x[presentes].astype(float), y[presentes]
    if xv.nunique() <= 1:
        return []
    cuantiles = np.unique(np.quantile(xv, np.linspace(0, 1, max_tramos + 1)[1:-1]))
    cortes = [c for c in cuantiles if xv.min() <= c < xv.max()]
    minimo = min_participacion * len(x)

    def estadisticas(cortes_: list[float]) -> pd.DataFrame:
        bordes = [-np.inf, *cortes_, np.inf]
        idx = np.searchsorted(bordes, xv.values, side="left") - 1
        return (
            pd.DataFrame({"t": idx, "y": yv.values})
            .groupby("t")["y"]
            .agg(n="count", malos="sum")
            .reindex(range(len(bordes) - 1), fill_value=0)
        )

    # 1. Fusionar tramos pequeños o sin buenos/malos con el vecino de tasa más parecida.
    while cortes:
        est = estadisticas(cortes)
        tasa = est["malos"] / est["n"].replace(0, np.nan)
        problema = (est["n"] < minimo) | (est["malos"] == 0) | (est["malos"] == est["n"])
        if not problema.any():
            break
        i = int(np.flatnonzero(problema.values)[0])
        if i == 0:
            j = 0
        elif i == len(est) - 1:
            j = i - 1
        else:
            izq = np.nan_to_num(abs(tasa.iloc[i] - tasa.iloc[i - 1]), nan=np.inf)
            der = np.nan_to_num(abs(tasa.iloc[i] - tasa.iloc[i + 1]), nan=np.inf)
            j = i - 1 if izq <= der else i
        del cortes[j]

    # 2. Forzar monotonía en la dirección de la tendencia global.
    while cortes:
        est = estadisticas(cortes)
        tasa = (est["malos"] / est["n"]).values
        signo = np.sign(tasa[-1] - tasa[0]) or 1.0
        dif = np.diff(tasa) * signo
        if (dif > 0).all():
            break
        i = int(np.argmin(dif))  # el par de tramos que más viola la monotonía
        del cortes[i]
    return [float(c) for c in cortes]


def ajustar_variable(
    x: pd.Series,
    y: pd.Series,
    nombre: str,
    tipo: str,
    max_tramos: int = 8,
    min_participacion: float = 0.05,
    min_categoria: float = 0.02,
) -> TramosVariable:
    total_buenos = float((y == 0).sum())
    total_malos = float((y == 1).sum())
    tramos = TramosVariable(nombre=nombre, tipo=tipo)

    if tipo == "numerica":
        tramos.cortes = _ajustar_numerica(x, y, max_tramos, min_participacion)
        orden = _nombres_tramos([-np.inf, *tramos.cortes, np.inf]) + [FALTANTE]
    else:
        frecuencia = x.value_counts(normalize=True)
        tramos.grupos = {
            str(c): (str(c) if p >= min_categoria else OTRAS) for c, p in frecuencia.items()
        }
        orden = sorted(set(tramos.grupos.values())) + [FALTANTE]

    etiqueta = tramos.etiquetas(x)
    tabla = _completar_tabla(_conteos(etiqueta, y, orden), total_buenos, total_malos)
    # Un tramo de faltantes con muy pocos casos no permite estimar su riesgo: se le
    # asigna riesgo promedio (WoE 0).
    if FALTANTE in tabla.index and tabla.loc[FALTANTE, "participacion"] < MIN_FALTANTES:
        tabla.loc[FALTANTE, ["woe", "iv"]] = 0.0
    tramos.tabla = tabla
    tramos.woe = tabla["woe"].to_dict()
    return tramos


class AgrupadorWoE:
    """Ajusta los tramos de varias variables y transforma a WoE."""

    def __init__(self, numericas: list[str], categoricas: list[str], **opciones):
        self.numericas = list(numericas)
        self.categoricas = list(categoricas)
        self.opciones = opciones
        self.tramos: dict[str, TramosVariable] = {}

    def ajustar(self, df: pd.DataFrame, y: pd.Series) -> AgrupadorWoE:
        for nombre in self.numericas:
            self.tramos[nombre] = ajustar_variable(
                df[nombre], y, nombre, "numerica", **self.opciones
            )
        for nombre in self.categoricas:
            self.tramos[nombre] = ajustar_variable(
                df[nombre], y, nombre, "categorica", **self.opciones
            )
        return self

    def transformar(self, df: pd.DataFrame, variables: list[str] | None = None) -> pd.DataFrame:
        variables = list(self.tramos) if variables is None else variables
        return pd.DataFrame(
            {f"woe_{v}": self.tramos[v].transformar(df[v]) for v in variables}, index=df.index
        )

    def tabla_iv(self) -> pd.DataFrame:
        filas = [
            {"variable": v, "iv": t.iv, "tramos": len(t.tabla)} for v, t in self.tramos.items()
        ]
        tabla = pd.DataFrame(filas).sort_values("iv", ascending=False).reset_index(drop=True)
        tabla["poder"] = pd.cut(
            tabla["iv"],
            bins=[-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
            labels=["Sin poder", "Débil", "Medio", "Fuerte", "Sospechoso"],
            right=False,
        ).astype(str)
        return tabla
