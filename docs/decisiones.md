# Registro de decisiones

Cada decisión con la alternativa que se descartó y el motivo.

## D-01. Validación fuera de tiempo, no partición aleatoria

- **Decisión:** desarrollo con 2007-2010, prueba con 2011.
- **Alternativa:** partición aleatoria 80/20 de toda la base.
- **Motivo:** un banco aplica el modelo a solicitudes futuras. La partición aleatoria
  mezcla años y esconde cambios en la cartera, como el aumento de préstamos a 60 meses
  en 2011, que es justo lo que la validación debe detectar.

## D-02. Excluir la tasa de interés

- **Decisión:** no es candidata.
- **Alternativa:** usarla como variable.
- **Motivo:** LendingClub fija la tasa a partir de su propio modelo de riesgo. Usarla
  sería reutilizar ese modelo y no medir el aporte propio.

## D-03. Incluir la calificación de LendingClub como variable y como benchmark

- **Decisión:** es candidata del scorecard y, sola, es el benchmark.
- **Alternativa:** dejarla fuera para un modelo solo con datos de la solicitud.
- **Motivo:** el usuario previsto la conoce al decidir, igual que un banco conoce el
  puntaje de buró. El benchmark responde la pregunta que haría un comité: ¿el modelo
  agrega valor sobre lo que ya existe? El modelo sin calificación se reporta como
  sensibilidad.

## D-04. Scorecard logístico sobre WoE como modelo final

- **Decisión:** regresión logística sobre WoE.
- **Alternativa:** LightGBM o la red neuronal.
- **Motivo:** LightGBM empata en 2011 (diferencia de AUC +0,000) y sobreajusta; la red
  queda por debajo. Sin ganancia predictiva, gana el modelo que se explica en una tabla
  de puntos, se audita variable por variable y se implementa sin código de
  entrenamiento.

## D-05. Tramos monótonos con tamaño mínimo

- **Decisión:** tramos numéricos con al menos 5% de la muestra y tasa monótona.
- **Alternativa:** tramos por cuantiles sin restricciones.
- **Motivo:** un tramo pequeño tiene un WoE ruidoso, y un patrón no monótono (más
  ingreso, más riesgo en un tramo intermedio) no se puede defender ante un comité ni
  ante un cliente.

## D-06. Faltantes escasos con WoE 0

- **Decisión:** si los faltantes son menos de 1% de la muestra, su tramo recibe WoE 0.
- **Alternativa:** estimar su WoE como cualquier tramo.
- **Motivo:** el estado faltante tenía un solo caso; su WoE estimado (−0,74) era ruido.
  Con WoE 0 se le asigna el riesgo promedio.

## D-07. Eliminación hacia atrás por signo y p-valor

- **Decisión:** sale primero la variable con signo contrario al esperado y luego la de
  p-valor mayor que 0,05.
- **Alternativa:** mantener todas las variables con IV suficiente.
- **Motivo:** con la calificación en el modelo, el uso de línea revolvente cambia de
  signo. Mantenerla daría más puntos a quien usa más su línea, lo que no tiene sentido
  de negocio.

## D-08. Recalibrar solo el intercepto, y evaluarlo con otro semestre

- **Decisión:** ajuste de tendencia central estimado con 2011T1-T2 y evaluado con
  2011T3-T4.
- **Alternativa:** reestimar el modelo completo con 2011.
- **Motivo:** el orden del modelo sigue siendo bueno; lo que cambió es el nivel de
  riesgo. Evaluar el ajuste con los mismos datos con que se estimó habría exagerado su
  efecto.

## D-09. El dashboard lee solo artefactos

- **Decisión:** la aplicación usa `artefactos/` y `puntaje.puntuar_solicitud`, sin datos
  ni librerías de entrenamiento.
- **Alternativa:** que la aplicación entrene el modelo al arrancar.
- **Motivo:** se publica gratis en Streamlit Community Cloud con dependencias mínimas, y
  una prueba verifica que la definición exportada puntúa igual que el modelo.
