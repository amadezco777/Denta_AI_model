# Glosario

Términos usados en este repositorio. La idea es que cualquier persona que lea el código o la documentación entienda lo mismo por cada palabra.

---

## Datos y anotación

**Clase / categoría**
Cada tipo de hallazgo que el modelo debe distinguir (`caries`, `corona`, …). Internamente el modelo trabaja con índices numéricos; el nombre legible se define en la configuración del dataset. Ver [`taxonomy.md`](taxonomy.md).

**Taxonomía**
Conjunto completo de clases del proyecto, junto con su definición y el criterio para decidir qué entra en cada una. No es solo la lista de nombres: incluye las reglas de asignación. Cambiar la taxonomía obliga a reconstruir el dataset maestro y a reentrenar.

**Label / etiqueta**
La anotación asignada a un objeto concreto de una imagen: qué clase es y dónde está. En formato YOLO, una línea de un archivo `.txt` con `clase x_centro y_centro ancho alto`, con coordenadas normalizadas entre 0 y 1 respecto al tamaño de la imagen. "Etiqueta original" se refiere a la del dataset de procedencia, antes del mapeo.

**Bounding box (caja delimitadora)**
Rectángulo que encierra un objeto en la imagen. Es el tipo de anotación que usa la detección de objetos. Se expresa en distintos formatos según la convención:
- `[x_min, y_min, x_max, y_max]` — esquinas (Pascal VOC)
- `[x_min, y_min, ancho, alto]` — esquina + tamaño, en píxeles (COCO)
- `[x_centro, y_centro, ancho, alto]` — centro + tamaño, normalizado (YOLO)

Confundir formatos es el error más habitual al convertir datasets; de ahí que exista `scripts/dataset_validation/`.

**Mapeo de etiquetas**
Tabla que traduce cada etiqueta original de un dataset a una clase de la taxonomía propia, o la marca como descartada. Ejemplo: `"Caries"`, `"caries_profunda"` y `"decay"` de tres datasets distintos pueden mapearse todos a `caries`. Es el paso que hace posible unir fuentes heterogéneas, y donde se introducen los errores más difíciles de detectar.

**EDA (Exploratory Data Analysis, análisis exploratorio de datos)**
Inspección inicial de un dataset antes de usarlo: cuántas imágenes hay, de qué resolución, cómo se reparten las clases, qué tamaño tienen las cajas, si hay imágenes corruptas o anotaciones incoherentes. Sirve para decidir si el dataset es utilizable y para anticipar problemas de entrenamiento.

**Dataset maestro**
Resultado de unir todos los datasets integrados, ya limpios, mapeados a la taxonomía propia y normalizados a un único formato. Es el único dataset del que se entrena. Vive en `data/processed/` y se genera siempre mediante scripts, nunca a mano.

**Train / Validation / Test**
Las tres particiones del dataset maestro:
- **Train (entrenamiento).** El modelo aprende de estas imágenes ajustando sus pesos.
- **Validation (validación).** Se usa durante el entrenamiento para medir cómo va y decidir hiperparámetros o cuándo parar. El modelo no aprende de ellas, pero sí influyen en las decisiones.
- **Test (prueba).** Se reserva y no se toca hasta el final. Da la medida honesta del rendimiento.

Una imagen solo puede estar en una partición. Si imágenes del mismo paciente o duplicados caen en particiones distintas se produce **fuga de datos** (*data leakage*) y las métricas salen infladas.

---

## Modelado y evaluación

**Baseline (línea base)**
Primer modelo funcional, entrenado con una configuración simple y sin optimizar. No busca ser bueno: busca ser la referencia contra la que comparar todo lo que venga después. Sin baseline no hay forma de saber si una mejora posterior realmente mejora algo.

**Inferencia**
Usar el modelo ya entrenado para predecir sobre imágenes nuevas. A diferencia del entrenamiento, no modifica los pesos del modelo.

**IoU (Intersection over Union)**
Medida de solapamiento entre la caja predicha y la caja real: área de la intersección dividida entre el área de la unión. Va de 0 (sin solapamiento) a 1 (coincidencia exacta). Se fija un umbral (habitualmente 0.5) para decidir si una predicción cuenta como acierto.

**Verdadero positivo / falso positivo / falso negativo**
- **VP:** el modelo detecta un hallazgo que efectivamente está ahí.
- **FP:** el modelo detecta algo que no está (falsa alarma).
- **FN:** el modelo no detecta un hallazgo que sí está (omisión).

**Precision (precisión)**
`VP / (VP + FP)`. De todo lo que el modelo detectó, qué proporción era correcta. Precisión baja = muchas falsas alarmas.

**Recall (exhaustividad / sensibilidad)**
`VP / (VP + FN)`. De todo lo que había que detectar, qué proporción encontró el modelo. Recall bajo = se le escapan hallazgos.

En un contexto de apoyo al diagnóstico, un falso negativo (patología no detectada) suele ser más grave que un falso positivo (región que el profesional descarta al revisarla). Esto condiciona qué métrica priorizar.

**F1-score**
Media armónica de precision y recall: `2 · (P · R) / (P + R)`. Resume ambas en un solo número y penaliza que una de las dos sea muy baja. Útil para comparar modelos de un vistazo, pero oculta el equilibrio concreto entre falsas alarmas y omisiones.

**mAP (mean Average Precision)**
Métrica estándar en detección de objetos. Para cada clase se calcula el *Average Precision* (área bajo la curva precision-recall) y después se promedia entre clases.
- **mAP@0.5:** con umbral de IoU de 0.5. Es la más citada y la más permisiva con la localización.
- **mAP@0.5:0.95:** promedio sobre umbrales de IoU de 0.5 a 0.95 en pasos de 0.05. Más exigente, penaliza cajas mal ajustadas.

Conviene mirar siempre el AP **por clase** y no solo el promedio: con clases desbalanceadas, un mAP aceptable puede esconder una clase que el modelo no detecta prácticamente nunca.

---

## Otros términos del flujo

**Anotación normalizada**
Anotación ya convertida al formato único del proyecto, independientemente del formato en que viniera el dataset original.

**Desbalance de clases**
Situación en la que unas clases tienen muchos más ejemplos que otras. Es lo esperable aquí: las restauraciones son mucho más frecuentes que las lesiones periapicales. Sesga el entrenamiento hacia las clases mayoritarias.

**Radiografía panorámica (ortopantomografía, OPG)**
Radiografía extraoral que captura en una sola imagen ambas arcadas dentarias completas, junto con estructuras óseas adyacentes. Es el tipo de imagen de entrada de este proyecto.

**Data leakage (fuga de datos)**
Cualquier vía por la que información de las particiones de validación o test llega al entrenamiento. Produce métricas optimistas que no se sostienen en uso real. Causas típicas aquí: imágenes duplicadas entre datasets, o varias radiografías del mismo paciente repartidas entre particiones.

**Reproducibilidad**
Que cualquier persona pueda regenerar el dataset maestro y reentrenar el modelo partiendo de este repositorio y de los datasets originales, obteniendo el mismo resultado. Por eso todo paso de transformación es un script versionado y no una edición manual.
