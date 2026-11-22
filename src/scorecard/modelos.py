"""Scorecard logístico sobre WoE, benchmark y modelos retadores.

- Scorecard: regresión logística sobre las variables transformadas a WoE. Selección por
  IV, correlación y eliminación hacia atrás (p-valor y signo esperado).
- Benchmark: la calificación de LendingClub sola. Responde si el scorecard agrega valor
  sobre la calificación que ya existe.
- Retadores: LightGBM y una red neuronal (1 capa oculta de 16 neuronas, sigmoide), sobre
  las mismas variables. Sirven para medir cuánto se pierde por usar un modelo lineal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scorecard.datos import BENCHMARK, CATEGORICAS, EVENTO, NUMERICAS, SEMILLA
from scorecard.woe import AgrupadorWoE

IV_MINIMO = 0.02
CORRELACION_MAXIMA = 0.6
P_VALOR_MAXIMO = 0.05


@dataclass
class Seleccion:
    variables: list[str]
    bitacora: pd.DataFrame = field(default_factory=pd.DataFrame)


def seleccionar_por_iv_y_correlacion(
    agrupador: AgrupadorWoE, df: pd.DataFrame, candidatas: list[str]
) -> Seleccion:
    """Filtra por IV y, entre pares muy correlacionados, conserva la de mayor IV."""
    iv = agrupador.tabla_iv().set_index("variable")["iv"]
    bitacora = []
    vivas = []
    for v in sorted(candidatas, key=lambda c: -iv[c]):
        if iv[v] < IV_MINIMO:
            bitacora.append((v, f"IV {iv[v]:.3f} menor que {IV_MINIMO}"))
        else:
            vivas.append(v)
    correlacion = agrupador.transformar(df, vivas).corr().abs()
    elegidas: list[str] = []
    for v in vivas:  # ya ordenadas por IV descendente
        choque = [
            e for e in elegidas if correlacion.loc[f"woe_{v}", f"woe_{e}"] > CORRELACION_MAXIMA
        ]
        if choque:
            r = correlacion.loc[f"woe_{v}", f"woe_{choque[0]}"]
            bitacora.append((v, f"correlación {r:.2f} con {choque[0]}, que tiene más IV"))
        else:
            elegidas.append(v)
    return Seleccion(elegidas, pd.DataFrame(bitacora, columns=["variable", "motivo"]))


class Scorecard:
    """Regresión logística sobre WoE con eliminación hacia atrás."""

    def __init__(self, agrupador: AgrupadorWoE, candidatas: list[str]):
        self.agrupador = agrupador
        self.candidatas = candidatas
        self.variables: list[str] = []
        self.resultado = None
        self.bitacora = pd.DataFrame(columns=["variable", "motivo"])

    def ajustar(self, df: pd.DataFrame) -> Scorecard:
        self._df_ajuste = df
        y = df[EVENTO]
        seleccion = seleccionar_por_iv_y_correlacion(self.agrupador, df, self.candidatas)
        variables = list(seleccion.variables)
        bitacora = list(seleccion.bitacora.itertuples(index=False, name=None))
        while variables:
            X = sm.add_constant(self.agrupador.transformar(df, variables))
            resultado = sm.Logit(y, X).fit(disp=0)
            coef = resultado.params.drop("const")
            pval = resultado.pvalues.drop("const")
            # Con WoE = ln(buenos/malos), el coeficiente esperado es negativo.
            signo_errado = coef[coef >= 0]
            if len(signo_errado):
                peor = signo_errado.index[0][4:]
                bitacora.append((peor, "signo contrario al esperado"))
                variables.remove(peor)
                continue
            if pval.max() > P_VALOR_MAXIMO:
                peor = pval.idxmax()[4:]
                bitacora.append((peor, f"p-valor {pval.max():.3f} mayor que {P_VALOR_MAXIMO}"))
                variables.remove(peor)
                continue
            break
        if not variables:  # ninguna variable sobrevivió: modelo solo con intercepto
            constante = pd.DataFrame({"const": np.ones(len(df))}, index=df.index)
            resultado = sm.Logit(y, constante).fit(disp=0)
        self.variables = variables
        self.resultado = resultado
        self.bitacora = pd.DataFrame(bitacora, columns=["variable", "motivo"])
        return self

    @property
    def intercepto(self) -> float:
        return float(self.resultado.params["const"])

    @property
    def coeficientes(self) -> pd.Series:
        return self.resultado.params.drop("const").rename(lambda c: c[4:])

    def tabla_coeficientes(self) -> pd.DataFrame:
        r = self.resultado
        tabla = pd.DataFrame(
            {
                "coeficiente": r.params,
                "error_estandar": r.bse,
                "z": r.tvalues,
                "p_valor": r.pvalues,
            }
        )
        tabla.index = [i[4:] if i.startswith("woe_") else "intercepto" for i in tabla.index]
        X = self.agrupador.transformar(self._df_ajuste, self.variables).values
        tabla["vif"] = [np.nan, *[_vif(X, i) for i in range(X.shape[1])]]
        return tabla

    def logit(self, df: pd.DataFrame) -> np.ndarray:
        if not self.variables:
            return np.full(len(df), self.intercepto)
        X = self.agrupador.transformar(df, self.variables)
        return self.intercepto + X.values @ self.coeficientes.values

    def predecir(self, df: pd.DataFrame) -> np.ndarray:
        """Probabilidad de incumplimiento (PD)."""
        return 1 / (1 + np.exp(-self.logit(df)))


def _vif(X: np.ndarray, i: int) -> float:
    if X.shape[1] == 1:
        return 1.0
    otros = np.column_stack([np.ones(len(X)), np.delete(X, i, axis=1)])
    beta, *_ = np.linalg.lstsq(otros, X[:, i], rcond=None)
    residuo = X[:, i] - otros @ beta
    r2 = 1 - residuo.var() / X[:, i].var()
    return float(1 / (1 - r2)) if r2 < 1 else np.inf


class Benchmark:
    """PD por calificación de LendingClub, estimada en entrenamiento."""

    def ajustar(self, df: pd.DataFrame) -> Benchmark:
        self.tasas = df.groupby(BENCHMARK)[EVENTO].mean()
        self.promedio = float(df[EVENTO].mean())
        return self

    def predecir(self, df: pd.DataFrame) -> np.ndarray:
        return df[BENCHMARK].map(self.tasas).fillna(self.promedio).astype(float).values


def _matriz_arbol(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    X = df[variables].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category")
    return X


class RetadorLightGBM:
    def __init__(self, variables: list[str]):
        self.variables = variables

    def ajustar(self, entrenamiento: pd.DataFrame, validacion: pd.DataFrame) -> RetadorLightGBM:
        Xe = _matriz_arbol(entrenamiento, self.variables)
        self.categorias = {c: Xe[c].cat.categories for c in Xe.select_dtypes("category")}
        self.modelo = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.02,
            num_leaves=8,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            random_state=SEMILLA,
            n_jobs=1,
            verbose=-1,
        )
        self.modelo.fit(
            Xe,
            entrenamiento[EVENTO],
            eval_set=[(self.matriz(validacion), validacion[EVENTO])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        return self

    def matriz(self, df: pd.DataFrame) -> pd.DataFrame:
        X = _matriz_arbol(df, self.variables)
        for c, cats in self.categorias.items():
            X[c] = pd.Categorical(X[c], categories=cats)
        return X

    def predecir(self, df: pd.DataFrame) -> np.ndarray:
        return self.modelo.predict_proba(self.matriz(df))[:, 1]


class RetadorRedNeuronal:
    """1 capa oculta de 16 neuronas con activación sigmoide: la mejor arquitectura del
    trabajo académico previo. El preprocesamiento se ajusta solo con entrenamiento."""

    def __init__(self, variables: list[str]):
        self.variables = variables
        self.numericas = [v for v in variables if v in NUMERICAS]
        self.categoricas = [v for v in variables if v in CATEGORICAS or v == BENCHMARK]

    def _matriz(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.variables].copy()
        for c in self.categoricas:
            X[c] = X[c].astype(object).where(X[c].notna(), "Faltante")
        return X

    def ajustar(self, entrenamiento: pd.DataFrame) -> RetadorRedNeuronal:
        numerico = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        categorico = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.02)
        preprocesar = ColumnTransformer(
            [("num", numerico, self.numericas), ("cat", categorico, self.categoricas)]
        )
        red = MLPClassifier(
            hidden_layer_sizes=(16,),
            activation="logistic",
            alpha=1e-3,
            learning_rate_init=3e-3,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            max_iter=500,
            random_state=SEMILLA,
        )
        self.modelo = Pipeline([("prep", preprocesar), ("red", red)])
        self.modelo.fit(self._matriz(entrenamiento), entrenamiento[EVENTO])
        return self

    def predecir(self, df: pd.DataFrame) -> np.ndarray:
        return self.modelo.predict_proba(self._matriz(df))[:, 1]
