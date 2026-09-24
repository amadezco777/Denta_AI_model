# Taxonomía de clases

> ⚠️ **PROVISIONAL.** Esta taxonomía es una propuesta inicial, anterior al análisis de los datasets. **Está sujeta a cambios** una vez se conozca qué clases están realmente disponibles, con cuántos ejemplos y con qué calidad de anotación en los datasets públicos seleccionados. Ninguna decisión de entrenamiento debe tomarse asumiendo que esta lista es definitiva.

## Estado

| | |
|---|---|
| Versión | `0.1.0-draft` |
| Fecha | 2026-09-24 |
| Basada en | Criterio clínico previo, **sin** validación contra datos |
| Congelada | No |

La taxonomía se considerará **congelada** cuando se haya completado el análisis individual de todos los datasets registrados en [`dataset_registry.md`](dataset_registry.md) y se haya verificado que cada clase tiene ejemplos suficientes. Hasta entonces, cualquier cambio se documenta en el historial al final de este archivo.

## Clases propuestas

| id | nombre | tipo | descripción | criterio de anotación previsto |
|---|---|---|---|---|
| 0 | `caries` | patología | Lesión del tejido dental duro por proceso cariogénico, visible como radiolucidez en corona o cuello. | Caja sobre la zona radiolúcida de la lesión, no sobre el diente completo. |
| 1 | `lesion_periapical` | patología | Radiolucidez periapical asociada al ápice radicular (granuloma, quiste, absceso). | Caja sobre la radiolucidez apical, incluyendo el ápice implicado. |
| 2 | `diente_impactado` | anomalía | Pieza que no ha erupcionado a su posición normal y permanece retenida en el hueso. Frecuentemente terceros molares. | Caja sobre la pieza retenida completa. |
| 3 | `restauracion` | tratamiento | Material restaurador radiopaco (amalgama, resina, incrustación) en corona. | Caja sobre el material restaurador. |
| 4 | `endodoncia` | tratamiento | Tratamiento de conducto: obturación radicular radiopaca. | Caja sobre la obturación del conducto. |
| 5 | `corona` | tratamiento | Prótesis fija de recubrimiento total de la corona dental. | Caja sobre la corona protésica. |

Los `id` numéricos son provisionales y solo se fijarán al congelar la taxonomía, porque el formato YOLO depende del orden.

## Agrupación conceptual

- **Patologías:** `caries`, `lesion_periapical` — lo que el modelo debe detectar como hallazgo clínico relevante.
- **Anomalías de desarrollo/posición:** `diente_impactado`.
- **Tratamientos previos:** `restauracion`, `endodoncia`, `corona` — describen el estado dental, no una patología.

Esta separación importa a la hora de evaluar: un fallo en una clase de patología tiene más peso clínico que un fallo en una clase de tratamiento.

## Criterios de inclusión de una clase

Una clase entra en la taxonomía definitiva si cumple:

1. **Disponibilidad.** Aparece anotada en al menos un dataset público utilizable.
2. **Volumen mínimo.** Tiene suficientes instancias para entrenar y evaluar (umbral a definir tras el EDA; orientativamente, no menos de unos cientos de instancias).
3. **Consistencia.** El criterio de anotación es razonablemente homogéneo entre las fuentes que la incluyen, o se puede homogeneizar mediante mapeo.
4. **Distinguibilidad.** Es visualmente separable de las demás clases en una radiografía panorámica.

## Riesgos y ambigüedades conocidas

Puntos a resolver durante el análisis de datasets:

- **`restauracion` vs `corona`.** Ambas son radiopacidades en la corona dental; los datasets pueden no distinguirlas o hacerlo con criterios distintos. Posible fusión en una clase única.
- **`endodoncia` vs `restauracion`.** Un diente tratado endodónticamente suele llevar además restauración; puede haber solapamiento de cajas y criterios de anotación distintos por dataset.
- **`caries`.** Es la clase con mayor variabilidad entre anotadores y la de menor contraste radiográfico. Es probable que sea la de peor rendimiento del modelo.
- **Granularidad.** Algunos datasets subdividen (p. ej. caries por profundidad, lesiones por tipo). El mapeo tendrá que colapsar esos niveles.
- **Clases ausentes.** Puede ser necesario añadir clases frecuentes en los datasets y útiles para el proyecto (p. ej. `implante`, `diente_ausente`, `resto_radicular`, `reabsorcion`).
- **Etiquetas negativas.** Definir si las imágenes sin hallazgos se incluyen como ejemplos negativos y en qué proporción.

## Convenciones

- Nombres en `snake_case`, en español, sin tildes ni caracteres especiales, para evitar problemas de codificación en rutas y archivos de configuración.
- El nombre canónico de cada clase vive en el archivo de configuración del dataset maestro (`configs/data/`); este documento es la referencia conceptual, la configuración es la fuente de verdad para el código.

## Historial de cambios

| fecha | versión | cambio |
|---|---|---|
| 2026-09-24 | `0.1.0-draft` | Propuesta inicial de seis clases, previa al análisis de datasets. |
