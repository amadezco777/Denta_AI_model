# Taxonomía de clases

> ⚠️ **PROVISIONAL.** Esta taxonomía es una propuesta inicial, anterior al análisis de los datasets. **Está sujeta a cambios** una vez se conozca qué clases están realmente disponibles, con cuántos ejemplos y con qué calidad de anotación en los datasets públicos seleccionados. Ninguna decisión de entrenamiento debe tomarse asumiendo que esta lista es definitiva.

## Estado

| | |
|---|---|
| Versión | `0.1.3-draft` |
| Fecha | 2026-09-27 |
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
- **`caries` vs `deep caries` (DENTEX).** Revisión manual de 10 imágenes de muestra (2026-09-27, ver `scripts/dataset_analysis/dentex_visualize.py`): ni el usuario ni el asistente, sin formación clínica, pudieron distinguir a simple vista qué hacía "profunda" a una caries frente a otra. Esto **no es evidencia suficiente para fusionar las clases**: el criterio de un anotador experto puede basarse en información no evidente en una imagen reducida (profundidad hacia la pulpa, densidad). Además, fusionar **empeora** el desbalance frente a `lesion_periapical` (pasa de 13,9x a 17,5x sobre los datos de DENTEX). **Decisión: mantener ambas clases separadas por ahora.** Se revisará con evidencia objetiva (matriz de confusión del primer modelo baseline) en vez de con intuición visual; si el modelo las confunde sistemáticamente, se reconsiderará la fusión en ese momento.
- **`caries` vs `restauracion` (riesgo entre datasets).** En imágenes de muestra de DENTEX, algunas cajas de `Caries` rodean zonas con manchas radiopacas (brillantes), compatibles con una restauración existente sin etiquetar como tal (DENTEX no tiene esa clase) — posiblemente caries secundaria junto a una restauración antigua. Esto no es un problema mientras se entrene solo con DENTEX. **Se vuelve un riesgo real si se integra un dataset que sí anota `restauracion` (p. ej. OralXrays-9):** el mismo patrón visual podría llegar etiquetado como `caries` en un dataset y como `restauracion` en otro, enseñándole al modelo señales contradictorias. Pendiente de verificar comparando ejemplos reales de ambas clases si se integra ese dataset. No bloquea el trabajo actual.
- **Criterio de caja: diente vs lesión (conflicto entre DENTEX, Mendeley y este documento).** El EDA y la revisión visual (2026-09-27) muestran que cada dataset encierra algo distinto con la misma clase:
  - **DENTEX** encierra **el diente completo** que tiene el diagnóstico. `Periapical Lesion`: mediana del 1,4 % de la imagen y ~190 px de lado menor; las cuatro clases miden casi lo mismo.
  - **Mendeley** encierra **solo la zona del ápice** con la lesión: mediana del 0,1 % de la imagen y ~56 px de lado menor, unas 14 veces menos área.
  - **Este documento** preveía cajas sobre la lesión y no sobre el diente (ver `caries` y `lesion_periapical` arriba). Eso coincide con Mendeley y **contradice a DENTEX**, que es hoy la fuente principal.

  Mezclar ambos criterios en una misma clase le daría al modelo dos respuestas correctas contradictorias. Además, Mendeley no anota caries ni dientes impactados, así que en un modelo multiclase esas zonas contarían como fondo. **Propuesta (pendiente de confirmar por el usuario):** modelo principal solo con DENTEX (4 clases, caja = diente) y, más adelante, un modelo especialista separado de `lesion_periapical` con Mendeley (caja = ápice). Dentro de ese modelo se probaría como experimento usar todos los grados o solo los grados 4 y 5. Si se adopta, el criterio de anotación de la tabla de clases se reescribirá para reflejar la caja a nivel de diente del modelo principal.
- **Nueva alternativa (2026-09-27, pendiente de decidir): Muresanu como dataset principal.** El dataset de Muresanu et al. (`muresanu_cluj`, ver registro) cubre las **6 clases** de esta taxonomía, con caja **a nivel de lesión o estructura**. Eso coincide con Mendeley y con el criterio original de este documento, no con DENTEX. Si se adopta, el plan pasaría a ser: modelo principal con Muresanu (6 clases), Mendeley como posible refuerzo de `lesion_periapical` (mismo criterio de caja, pero sin las demás clases anotadas) y DENTEX como el que no encaja (quizá solo para `diente_impactado`, que sí es a nivel de diente en los tres, o como prueba externa). Mapeo provisional: `caries` ← Carious lesion; `lesion_periapical` ← Apical periodontitis; `diente_impactado` ← Impacted tooth; `restauracion` ← Obturation; `corona` ← Prosthetic restoration; `endodoncia` ← Endodontic treatment. El acuerdo entre anotadores medido en sus duplicados (caries 49 %, periodontitis apical 40 %, impactado 87 %) confirma que `caries` y `lesion_periapical` serán las clases más difíciles.
- **Revisión visual de caries y periodontitis apical en Muresanu (2026-09-27, 8 recortes por clase, ojo no experto; figuras en `reports/figures/muresanu_samples/class_04_*` y `class_07_*`).** Opinión compartida por el usuario:
  - **Caries:** cajas en la corona, sobre zonas oscuras, a nivel de lesión. Varias están **pegadas a empastes o coronas** (caries secundaria: mismo riesgo visual `caries`/`restauracion` que en DENTEX, pero aquí ambas clases están anotadas y el modelo puede aprender a separarlas). Caries superficiales y profundas van en la misma clase. Un caso dudoso (`train/837`, imagen borrosa).
  - **Periodontitis apical:** mancha oscura en la punta de la raíz, criterio de caja **coherente con Mendeley** (ambos a nivel de ápice), así que podrían combinarse para `lesion_periapical`; DENTEX no. Casos dudosos en el maxilar superior, donde se superponen paladar y seno (`train/529`, `train/636`); coincide con el bajo acuerdo entre anotadores.
  - **Riesgo nuevo: atajo endodoncia → lesión apical.** La mayoría de las lesiones aparecen en dientes con endodoncia. El modelo podría aprender "si hay endodoncia, hay lesión" en vez de mirar la radiolucidez. **En la evaluación hay que medir la detección de lesiones apicales en dientes sin endodoncia.**
  - Limitación: 8 recortes por clase es poco. Pendiente, si hace falta: galerías de 24 y comparación lado a lado Muresanu/Mendeley.
- **Grados de `lesion_periapical` (Mendeley).** El campo `<name>` vale 3, 4 o 5. Hipótesis: grado del índice PAI (Ørstavik et al., 1986, escala 1–5), no confirmada por los autores; el paper del PAI aún no se ha revisado. El tamaño de caja crece con el grado, coherente con la hipótesis. El grado 3 es difícil de distinguir a ojo no experto. **No se crean subclases:** todos los grados mapean a `lesion_periapical`.
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
| 2026-09-27 | `0.1.1-draft` | EDA y revisión visual de DENTEX (705 imágenes con diagnóstico): confirma desbalance fuerte en `lesion_periapical` (158 cajas vs 2.189 de `caries`, 13,9x). Se documentan dos riesgos nuevos (`caries` vs `deep caries` indistinguibles a ojo no experto; posible solapamiento visual `caries`/`restauracion` entre datasets) y se decide **no fusionar** `caries`/`deep caries` por ahora. Taxonomía sigue sin congelar. |
| 2026-09-27 | `0.1.2-draft` | EDA y revisión visual de Mendeley (3.924 originales, 6.741 cajas de lesión periapical, grados 3/4/5). Se documenta el conflicto de criterio de caja (DENTEX = diente, Mendeley = ápice, este documento = lesión) y la propuesta, pendiente de confirmar, de dos modelos separados. Los grados de Mendeley no generan subclases. Taxonomía sigue sin congelar. |
| 2026-09-27 | `0.1.3-draft` | EDA, limpieza y revisión visual de Muresanu (1.582 radiografías limpias). Se documenta la alternativa de usarlo como dataset principal (cubre las 6 clases, caja a nivel de lesión), con un mapeo provisional, y el acuerdo entre anotadores por clase. Pendiente de decidir. Taxonomía sigue sin congelar. |
