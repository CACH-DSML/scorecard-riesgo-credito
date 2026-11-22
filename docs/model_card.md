# Model card — Scorecard de riesgo de crédito

Las cifras salen de `artefactos/` y se regeneran con `python -m scorecard.pipeline`.

## Qué es

Un scorecard que estima la probabilidad de que un préstamo de consumo termine castigado
(*Charged Off*), expresada en puntos: 600 puntos equivalen a odds de 50 buenos por malo y
cada 20 puntos se duplican las odds.

## Uso previsto

- **Sirve para:** ordenar solicitudes de préstamos de LendingClub por riesgo y analizar
  puntos de corte, por parte de quien conoce la solicitud y la calificación de
  LendingClub.
- **No sirve para:** decidir créditos reales, otros productos u otros países, ni calcular
  provisiones sin recalibrar la PD.

## Datos

- **Fuente:** muestra de los datos públicos de préstamos de LendingClub, 2007-2011.
- **Población:** 19 905 préstamos terminados (pagados o castigados) de CA, NY, FL, TX, NJ
  e IL.
- **Evento:** el préstamo terminó castigado. Ventana: la vida completa del préstamo.

| Parte | Años | Préstamos | Tasa de incumplimiento |
|---|---|---|---|
| Entrenamiento | 2007-2010 | 6 819 | 13,7% |
| Validación | 2007-2010 | 2 273 | 13,7% |
| Prueba fuera de tiempo | 2011 | 10 813 | 15,7% |

## Variables

Cinco, todas transformadas a WoE: calificación de LendingClub, plazo, propósito, ingreso
anual y estado.

**Descartadas y por qué** (detalle en `artefactos/seleccion.csv`):

- Ocho variables con IV menor que 0,02: antigüedad laboral, quiebras, verificación del
  ingreso, monto, deuda/ingreso, cuota, vivienda y moras en 2 años.
- Cuota anual / ingreso: correlación de 0,87 con monto / ingreso.
- Uso de línea revolvente: signo contrario al esperado con la calificación en el modelo.
- Monto / ingreso: p-valor 0,38.

**No candidatas:** la tasa de interés, porque LendingClub la fija con su propio modelo de
riesgo, y el cargo declarado, que es texto libre.

## Resultados en 2011 (fuera de tiempo)

| Métrica | Valor |
|---|---|
| AUC | 0,691 (IC 95%: 0,678 a 0,705) |
| Gini | 0,382 |
| KS | 0,291 |
| Brier | 0,124 |
| PD promedio estimada | 14,6% |
| Tasa observada | 15,7% |
| Hosmer-Lemeshow | p = 0,002 |
| PSI del puntaje | 0,042 |

**Benchmark y retadores** (diferencia de AUC del scorecard, bootstrap pareado):

| Contra | Diferencia | IC 95% |
|---|---|---|
| Calificación de LendingClub sola | +0,031 | +0,021 a +0,040 |
| Scorecard sin calificación | +0,016 | +0,009 a +0,024 |
| LightGBM | +0,000 | −0,005 a +0,005 |
| Red neuronal | +0,022 | +0,012 a +0,033 |

## Calibración

En validación (2007-2010) la calibración es buena (Hosmer-Lemeshow p = 0,13). En 2011 el
modelo subestima: la tasa subió de 13,7% a 15,7%, en parte porque los préstamos a 60
meses, más riesgosos, pasaron de 16% a 33% de la cartera.

Un ajuste del intercepto estimado con 2011T1-T2 y evaluado con 2011T3-T4 reduce el error
por decil de 1,8 a 1,5 puntos, pero no cierra la brecha: la tasa siguió subiendo en el
cuarto trimestre (16,9%).

## Estabilidad

| Muestra | PSI del puntaje |
|---|---|
| Validación | 0,005 |
| 2011 | 0,042 |
| Trimestre más alto (2011T1) | 0,072 |

El plazo es la única variable con cambio moderado (CSI 0,16).

## Revisión de fugas de información

- Ninguna variable tiene IV mayor que 0,5.
- Las variables son de la solicitud; no hay pagos ni recuperaciones posteriores.
- La tasa de interés queda fuera.
- Los tramos, la imputación y la escala de la red neuronal se ajustan solo con
  entrenamiento.
- Con el evento barajado, el modelo no discrimina (prueba
  `test_con_el_evento_barajado_no_hay_discriminacion`).

## Limitaciones

1. Sin información de buró: la discriminación es moderada.
2. Subestima el riesgo en 2011: requiere recalibración antes de usar la PD.
3. Una sola ventana de prueba.
4. Muestra parcial: seis estados.
5. Solo préstamos aprobados: sin inferencia de rechazados.
6. Sin pérdida dada el incumplimiento.

## Monitoreo propuesto

| Indicador | Frecuencia | Alerta | Acción |
|---|---|---|---|
| PSI del puntaje | Mensual | Mayor que 0,10 | Revisar la mezcla de la cartera |
| CSI del plazo | Mensual | Mayor que 0,25 | Evaluar reajustar los tramos |
| Tasa observada contra PD promedio | Trimestral | Diferencia mayor que 2 puntos | Recalibrar el intercepto |
| AUC y KS | Trimestral | AUC menor que 0,65 | Revisar el modelo completo |
