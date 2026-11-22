"""Carga, limpieza y partición de la base de préstamos.

La base es una muestra de 19 908 préstamos de LendingClub otorgados entre 2007 y 2011,
todos ya terminados: pagados por completo (Fully Paid) o castigados (Charged Off).

Partición fuera de tiempo, como en banca:
- Desarrollo: préstamos otorgados de 2007 a 2010. Se divide al azar en entrenamiento (75%)
  y validación (25%), estratificando por el evento.
- Fuera de tiempo (OOT): préstamos otorgados en 2011. El modelo no los ve hasta la prueba.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[2]
RUTA_DATOS = RAIZ / "data" / "prestamos_lendingclub_2007_2011.csv"

EVENTO = "incumple"
SEMILLA = 42
ANIO_CORTE_DESARROLLO = 2010

ESTADOS_VALIDOS = {"CA", "NY", "FL", "TX", "NJ", "IL"}

# Variables disponibles al momento de la solicitud. No se usan la calificación (grade) ni la
# tasa (int_rate): las fija LendingClub a partir de su propio modelo de riesgo, así que
# usarlas sería reutilizar ese modelo. La calificación se guarda como benchmark.
NUMERICAS = [
    "monto",
    "plazo",
    "cuota",
    "ingreso_anual",
    "dti",
    "moras_2_anios",
    "uso_linea_revolvente",
    "quiebras",
    "antiguedad_laboral",
    "cuota_ingreso",
    "monto_ingreso",
]
CATEGORICAS = ["vivienda", "verificacion_ingreso", "proposito", "estado"]
VARIABLES = NUMERICAS + CATEGORICAS
BENCHMARK = "calificacion"

ETIQUETAS = {
    "monto": "Monto solicitado (USD)",
    "plazo": "Plazo (meses)",
    "cuota": "Cuota mensual (USD)",
    "ingreso_anual": "Ingreso anual (USD)",
    "dti": "Deuda / ingreso (%)",
    "moras_2_anios": "Moras de 30+ días en 2 años",
    "uso_linea_revolvente": "Uso de línea revolvente (%)",
    "quiebras": "Quiebras registradas",
    "antiguedad_laboral": "Antigüedad laboral (años)",
    "cuota_ingreso": "Cuota anual / ingreso",
    "monto_ingreso": "Monto / ingreso anual",
    "vivienda": "Tenencia de vivienda",
    "verificacion_ingreso": "Verificación del ingreso",
    "proposito": "Propósito del préstamo",
    "estado": "Estado de residencia",
    "calificacion": "Calificación de LendingClub",
}

_RENOMBRAR = {
    "loan_amnt": "monto",
    "term": "plazo",
    "installment": "cuota",
    "annual_inc": "ingreso_anual",
    "dti": "dti",
    "delinq_2yrs": "moras_2_anios",
    "revol_util": "uso_linea_revolvente",
    "pub_rec_bankruptcies": "quiebras",
    "home_ownership": "vivienda",
    "verification_status": "verificacion_ingreso",
    "purpose": "proposito",
    "addr_state": "estado",
    "grade": "calificacion",
    "int_rate": "tasa",
}


@dataclass(frozen=True)
class Particion:
    entrenamiento: pd.DataFrame
    validacion: pd.DataFrame
    oot: pd.DataFrame

    @property
    def desarrollo(self) -> pd.DataFrame:
        return pd.concat([self.entrenamiento, self.validacion])

    def resumen(self) -> pd.DataFrame:
        filas = []
        for nombre, parte in [
            ("Entrenamiento (2007-2010)", self.entrenamiento),
            ("Validación (2007-2010)", self.validacion),
            ("Fuera de tiempo (2011)", self.oot),
        ]:
            filas.append(
                {
                    "parte": nombre,
                    "prestamos": len(parte),
                    "incumplimientos": int(parte[EVENTO].sum()),
                    "tasa_incumplimiento": parte[EVENTO].mean(),
                }
            )
        return pd.DataFrame(filas)


def _fecha(valor) -> pd.Timestamp:
    """La columna mezcla fechas de Excel y textos como 'Dec-2011'."""
    if pd.isna(valor):
        return pd.NaT
    texto = str(valor)
    for formato in ("%b-%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return pd.to_datetime(texto, format=formato)
        except ValueError:
            continue
    return pd.NaT


def _antiguedad(texto) -> float:
    if pd.isna(texto):
        return np.nan
    texto = str(texto)
    if texto.startswith("<"):
        return 0.0
    return float(texto.split()[0].rstrip("+"))


def cargar(ruta: Path = RUTA_DATOS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve la base limpia y una tabla con cuántos préstamos quedan tras cada filtro."""
    crudo = pd.read_csv(ruta)
    pasos = [("Base original", len(crudo))]

    df = crudo.rename(columns=_RENOMBRAR)
    df["fecha_emision"] = df["issue_d"].map(_fecha)

    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
    pasos.append(("Con estado final conocido (pagado o castigado)", len(df)))
    df = df[df["fecha_emision"].notna()].copy()
    pasos.append(("Con fecha de emisión válida", len(df)))

    df[EVENTO] = (df["loan_status"] == "Charged Off").astype(int)
    df["anio"] = df["fecha_emision"].dt.year
    df["trimestre"] = df["fecha_emision"].dt.to_period("Q").astype(str)

    df["antiguedad_laboral"] = df["emp_length"].map(_antiguedad)
    df["vivienda"] = df["vivienda"].replace({"OTHER": "OTRA", "NONE": "OTRA"})
    df.loc[~df["estado"].isin(ESTADOS_VALIDOS), "estado"] = np.nan
    # Dos valores de uso de línea muy por encima de 100% son errores de captura.
    df.loc[df["uso_linea_revolvente"] > 150, "uso_linea_revolvente"] = np.nan
    ingreso = df["ingreso_anual"].where(df["ingreso_anual"] > 0)
    df["cuota_ingreso"] = df["cuota"] * 12 / ingreso
    df["monto_ingreso"] = df["monto"] / ingreso

    columnas = ["index", "fecha_emision", "anio", "trimestre", EVENTO, BENCHMARK, "tasa"]
    df = df[columnas + VARIABLES].rename(columns={"index": "id"}).reset_index(drop=True)
    for columna in CATEGORICAS + [BENCHMARK]:
        df[columna] = df[columna].astype("object")

    return df, pd.DataFrame(pasos, columns=["filtro", "prestamos"])


def particionar(df: pd.DataFrame, semilla: int = SEMILLA) -> Particion:
    desarrollo = df[df["anio"] <= ANIO_CORTE_DESARROLLO]
    oot = df[df["anio"] > ANIO_CORTE_DESARROLLO]
    entrenamiento, validacion = train_test_split(
        desarrollo, test_size=0.25, stratify=desarrollo[EVENTO], random_state=semilla
    )
    return Particion(
        entrenamiento.reset_index(drop=True),
        validacion.reset_index(drop=True),
        oot.reset_index(drop=True),
    )
