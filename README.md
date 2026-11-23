# Scorecard de riesgo de crédito

Un scorecard de probabilidad de incumplimiento para préstamos de consumo, desarrollado y
validado como lo haría el equipo de riesgos de un banco: tramos con WoE, regresión
logística, puntos, validación fuera de tiempo, calibración, estabilidad y punto de corte.

- **Dashboard:** https://scorecard-riesgo-credito-7zghwzwtgbkkpfcipxupty.streamlit.app/
- **Documento de desarrollo y validación:** [notebooks/01_desarrollo_y_validacion.ipynb](notebooks/01_desarrollo_y_validacion.ipynb)
- **Model card:** [docs/model_card.md](docs/model_card.md)

## El problema

Una entidad que evalúa préstamos de LendingClub conoce los datos de la solicitud y la
calificación (A a G) que asigna LendingClub. ¿Puede ordenar mejor el riesgo que con la
calificación sola, y con un modelo que un comité de riesgos pueda leer?

## Resultados

Validación fuera de tiempo: el modelo se desarrolla con préstamos de 2007 a 2010 y se
prueba con los de 2011, que no vio.

| Modelo | AUC 2011 | Gini | KS |
|---|---|---|---|
| **Scorecard (logístico sobre WoE)** | **0,691** | **0,382** | **0,291** |
| Retador: LightGBM | 0,691 | 0,381 | 0,288 |
| Scorecard sin calificación | 0,675 | 0,350 | 0,268 |
| Retador: red neuronal | 0,669 | 0,338 | 0,247 |
| Benchmark: calificación de LendingClub | 0,660 | 0,320 | 0,246 |

**Supera a la calificación existente.** +0,031 de AUC, con intervalo de 95% de +0,021 a
+0,040 por bootstrap pareado.

**Un modelo lineal basta.** LightGBM empata (+0,000, IC de −0,005 a +0,005) y sobreajusta
(0,745 en entrenamiento contra 0,678 en validación). La red neuronal queda por debajo.

**El puntaje es estable** (PSI 0,04 entre 2007-2010 y 2011), aunque los préstamos a 60
meses pasaron de 16% a 33% de la cartera.

**Un hallazgo incómodo: subestima el riesgo en 2011.** PD promedio 14,6% contra 15,7%
observado; Hosmer-Lemeshow rechaza la calibración. Un ajuste del intercepto estimado con
el primer semestre mejora el segundo, pero no cierra toda la brecha. Antes de usar la PD
para provisiones o precios habría que recalibrarla.

## El scorecard

Cinco variables, cada una con signo esperado, p-valor menor que 0,05 y VIF cercano a 1:
calificación de LendingClub, plazo, propósito, ingreso anual y estado. Escala: 600 puntos
equivalen a odds de 50 buenos por malo y cada 20 puntos se duplican las odds.

| Variable | Tramo de menor riesgo | Puntos | Tramo de mayor riesgo | Puntos |
|---|---|---|---|---|
| Calificación | A | 133 | F y G | 87 |
| Plazo | 36 meses | 111 | 60 meses | 98 |
| Propósito | Tarjeta de crédito | 117 | Pequeño negocio | 88 |
| Ingreso anual | Más de USD 83 000 | 114 | Hasta USD 31 002 | 95 |
| Estado | Nueva York | 114 | Florida | 101 |

La tabla completa está en [artefactos/tabla_puntos.csv](artefactos/tabla_puntos.csv).

## Método

1. **Datos.** 19 905 préstamos terminados (pagados o castigados) de seis estados. Se
   excluyen la tasa de interés, que LendingClub fija con su propio modelo de riesgo, y el
   cargo declarado, que es texto libre.
2. **Partición fuera de tiempo.** Desarrollo 2007-2010 (75% entrenamiento, 25%
   validación); prueba con 2011.
3. **Tramos y WoE.** Tramos monótonos con al menos 5% de la muestra; categorías con menos
   de 2% agrupadas.
4. **Selección.** IV de 0,02 o más; entre pares con correlación mayor que 0,6 se queda la
   de mayor IV; eliminación hacia atrás por signo y p-valor.
5. **Validación.** Discriminación (AUC, Gini, KS, deciles), comparación con benchmark y
   retadores por bootstrap pareado, calibración (binomial por decil, Hosmer-Lemeshow),
   estabilidad (PSI y CSI por trimestre), importancia (SHAP en el retador) y análisis de
   punto de corte.

Las decisiones y sus alternativas están en [docs/decisiones.md](docs/decisiones.md).

## Estructura

```
src/scorecard/
  datos.py      carga, limpieza y partición fuera de tiempo
  woe.py        tramos, WoE e IV
  modelos.py    scorecard, benchmark y retadores
  metricas.py   AUC, KS, PSI, calibración, bootstrap
  puntaje.py    escala de puntos, punto de corte y puntuación de una solicitud
  pipeline.py   entrena, valida y guarda artefactos/
app/app.py      dashboard en Streamlit (lee solo artefactos/)
notebooks/      documento de desarrollo y validación
tests/          16 pruebas: unitarias y con la base real
```

## Cómo reproducir

Requiere Python 3.11 o superior.

```bash
python -m venv .venv
source .venv/bin/activate            # en Windows: .venv\Scripts\activate
pip install -e ".[dev,app,notebooks]"

python -m scorecard.pipeline         # entrena y valida en ~30 s; escribe artefactos/
pytest                               # 16 pruebas
streamlit run app/app.py             # dashboard
```

## Limitaciones

- **Sin información de buró.** No hay puntaje FICO ni historial de pagos; por eso la
  discriminación es moderada.
- **Una sola ventana de prueba (2011).** No se sabe cómo se comporta en otro momento del
  ciclo.
- **Muestra parcial:** seis estados, no la cartera completa de LendingClub.
- **Solo préstamos aprobados.** No se aplicó inferencia de rechazados.
- **Sin pérdida dada el incumplimiento.** El punto de corte se analiza en tasas, no en
  montos.

Es un proyecto de portafolio. No sirve para decidir créditos reales.

## Datos y licencia

`data/prestamos_lendingclub_2007_2011.csv` es una muestra de los datos públicos de
préstamos de LendingClub (2007-2011), preparada para un curso de Machine Learning de la
PUCP; se quitó la columna de cargo declarado. Los datos pertenecen a LendingClub y se
incluyen solo para reproducir el análisis.

El código se publica bajo licencia MIT.
