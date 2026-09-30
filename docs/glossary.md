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

---

## Entrenamiento

**Pesos (weights)**
Los números internos del modelo, que son millones. El entrenamiento los ajusta poco a poco hasta que el modelo acierta, así que los pesos son "lo aprendido". Se guardan en un archivo `.pt`.

**Modelo preentrenado (transfer learning)**
YOLO11 se publica ya entrenado sobre COCO, un dataset de fotos cotidianas (personas, coches, animales). No sabe nada de dientes, pero ya reconoce bordes, texturas y formas. Si se parte de esos pesos en lugar de empezar de cero, hacen falta muchas menos imágenes propias, y aquí solo hay 979 en train.

**YOLO, ultralytics, YOLO11n y YOLO11s**
YOLO (*You Only Look Once*) es una familia de detectores que predicen todas las cajas y sus clases en una sola pasada por la imagen. **ultralytics** es la librería de Python que lo implementa: entrena, valida y predice. La letra final indica el tamaño del modelo:
- **n** (*nano*): unos 2,6 millones de parámetros.
- **s** (*small*): unos 9,4 millones de parámetros.
- Después vienen m, l y x.

Un modelo más grande puede aprender detalles más finos, pero consume más VRAM, entrena más lento y con pocos datos tiene más riesgo de sobreajuste.

**Época (epoch)**
Una pasada completa por todas las imágenes de train. Con 979 imágenes y un batch de 16, una época son 62 pasos. Entrenar 50 épocas significa que el modelo ve cada imagen 50 veces, y cada vez un poco distinta por el aumento de datos.

**Batch (lote)**
Número de imágenes que el modelo procesa juntas antes de corregir sus pesos. Un batch más grande da correcciones más estables y épocas más rápidas, pero ocupa más VRAM. Si no cabe, aparece el error `CUDA out of memory`.

**Resolución de entrada (`imgsz`)**
Tamaño al que se reduce cada imagen antes de entrar al modelo. Con `imgsz: 640`, el lado largo de la panorámica queda en 640 píxeles. Si la radiografía original mide 2.800 px de ancho, todo se reduce unas 4,4 veces: una caries de 40 px queda en unos 9 px, y a ese tamaño es difícil de ver. Subir la resolución ayuda con los hallazgos pequeños, pero la memoria crece con el cuadrado: 960 px usa unas 2,25 veces la memoria de 640 px.

**Hiperparámetros**
Ajustes que decide la persona antes de entrenar: modelo, épocas, batch, `imgsz`, learning rate, aumento de datos… A diferencia de los pesos, el modelo no los aprende. Se eligen fuera del entrenamiento y se comparan midiendo sobre validación.

**Learning rate (`lr0`, tasa de aprendizaje)**
El tamaño del paso con que se corrigen los pesos en cada iteración. Si es demasiado grande, el entrenamiento oscila y no converge. Si es demasiado pequeño, aprende muy despacio.

**Patience (paciencia)**
Número de épocas que se espera sin que mejore la métrica de validación antes de detener el entrenamiento (*early stopping*). Así no se gasta tiempo y se evita seguir sobreajustando.

**Aumento de datos (data augmentation)**
Cambios aleatorios que se aplican a las imágenes de train en cada época, como voltear, rotar un poco, cambiar el brillo o la escala. Así el modelo ve más variedad y memoriza menos. Solo se usan cambios que podrían darse en una radiografía real:
- `fliplr` (voltear de izquierda a derecha) **sí**, porque la boca es aproximadamente simétrica.
- `flipud` (poner la imagen de cabeza) **no**, porque una panorámica nunca llega así.

Al voltear de izquierda a derecha, un diente del lado derecho pasa a parecer del izquierdo. Para las 7 clases actuales da igual, pero importaría si algún día se numeraran los dientes (FDI).

**Sobreajuste (overfitting)**
El modelo memoriza las imágenes de train en lugar de aprender el patrón general, y falla con imágenes nuevas. Se nota cuando las métricas de train siguen mejorando mientras las de validación se estancan o empeoran.

**Semilla (seed)**
Número que inicia el generador de números aleatorios. Con la misma semilla se repite el mismo orden de imágenes, los mismos aumentos y la misma inicialización. Por eso se anota siempre (aquí, `seed: 42`).

**No determinismo en GPU**
Aunque la semilla sea la misma, dos entrenamientos idénticos en GPU dan resultados algo distintos. La GPU hace miles de sumas en paralelo y no siempre en el mismo orden, y con decimales `(a+b)+c` no da exactamente lo mismo que `a+(b+c)`. Esas diferencias mínimas se acumulan durante el entrenamiento. Por eso se entrena con 2 o 3 semillas y se reporta la media con su variación. Si una "mejora" es menor que esa variación, no es una mejora real.

---

## Hardware y entorno

**GPU (tarjeta gráfica)**
Procesador con miles de núcleos sencillos que hacen muchas operaciones a la vez. Entrenar una red neuronal consiste sobre todo en multiplicar matrices enormes, y en eso la GPU es muchísimo más rápida que la CPU. La del proyecto es una NVIDIA RTX 4050 Laptop.

**VRAM**
Memoria propia de la GPU; la nuestra tiene 6 GB. Allí tienen que caber al mismo tiempo el modelo, el lote de imágenes y los cálculos intermedios. Es el límite práctico que decide el tamaño del modelo, el batch y `imgsz`. Si se supera, aparece el error `CUDA out of memory`.

**Driver**
Programa del sistema operativo que permite comunicarse con la GPU. Si es demasiado viejo, las versiones recientes de CUDA no funcionan. Aquí, el driver 528.97 provocaba `cudaErrorDevicesUnavailable`; se resolvió actualizando a 616.92.

**CUDA**
Plataforma de NVIDIA para usar la GPU en cálculos generales y no solo en gráficos. PyTorch la usa internamente. En `torch 2.14.0+cu126`, el sufijo `cu126` significa que esa versión está compilada para CUDA 12.6, así que el driver tiene que soportar esa versión o una posterior.

**PyTorch (`torch`)**
Librería de Python para construir y entrenar redes neuronales; ultralytics funciona encima de ella. `torch.cuda.is_available()` indica si PyTorch puede usar la GPU.

**float32 y redondeo**
Formato numérico que guarda unas 7 cifras significativas. Si la CPU y la GPU hacen el mismo cálculo, pueden diferir en las últimas cifras porque suman en otro orden. Una diferencia de 5·10⁻⁴ sobre valores de decenas es normal y no indica ningún error.

**TFLOPS**
Billones (10¹²) de operaciones con decimales por segundo, una medida de velocidad de cálculo. Multiplicar dos matrices de n×n cuesta unas 2·n³ operaciones. Una medición rápida sirve para confirmar que la GPU realmente está calculando. No es una prueba de rendimiento (*benchmark*) rigurosa.

---

## Reproducibilidad y procedencia

**Huella (hash SHA-256 o MD5)**
Texto corto calculado a partir del contenido de uno o varios archivos; en SHA-256 son 64 caracteres. Si cambia un solo píxel o un solo número de una etiqueta, la huella cambia por completo. Si dos huellas coinciden, el contenido es idéntico. `2f7eae71…db16` es la huella del dataset maestro. MD5 es un algoritmo más antiguo del mismo tipo, y es el que publica Zenodo para sus descargas.

**Commit**
Una "foto" del código en git, identificada por un código como `b416567`. Si el código tenía cambios sin guardar en un commit, se marca como `dirty: true` y no se puede recuperar exactamente.

**Procedencia (provenance)**
Registro de dónde sale un resultado: con qué código (commit), a partir de qué datos (huella) y con qué versiones de las librerías. Sirve para responder meses después a la pregunta "¿con qué se entrenó exactamente este modelo?".

**Prueba de humo (smoke test)**
Ejecución corta y barata que solo comprueba que todo funciona y mide los recursos (VRAM, tiempo por época) antes de lanzar el entrenamiento largo. Sus métricas no significan nada.

**Datos sintéticos y errores provocados**
Para probar un script se crean datos pequeños y falsos cuya respuesta correcta se conoce de antemano. Después se introducen errores a propósito, por ejemplo alterando un archivo o quitando una línea del código, para comprobar que las pruebas los detectan. Si una prueba sigue pasando después de romper el código, no estaba protegiendo nada.
