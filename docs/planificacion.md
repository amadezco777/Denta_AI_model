# Planificación del proyecto Dental AI

Cristhian Pazmino Parra · Universidad Espíritu Santo · Maestría en Inteligencia Artificial

Tutora Gladys Villegas · 26 de octubre de 2026

## Planteamiento de la investigación

### Problema de investigación

La interpretación de una panorámica requiere localizar múltiples regiones y relacionarlas con anatomía, antecedentes y otras exploraciones. Un clasificador global de imagen no muestra qué región sustenta una categoría. Las cajas delimitadoras ofrecen una representación operativa para estudiar la localización, pero su valor depende de que las anotaciones describan objetos comparables y de que la evaluación cuente correctamente coincidencias, omisiones y detecciones sobrantes.

La heterogeneidad afecta tanto a las fuentes como a las clases. Una caja que encierra un diente con diagnóstico no equivale necesariamente a una caja sobre la lesión. Una etiqueta ausente puede indicar ausencia real, omisión o categoría fuera del alcance del anotador. Mezclar esos casos sin control convierte regiones sin verificar en fondo de entrenamiento y limita la interpretación del recall. El proyecto mantuvo separadas las fuentes con geometría o cobertura incompatibles.

La duplicación plantea otra amenaza. Una copia exacta o una reexportación de una radiografía puede cambiar de nombre y aparecer en entrenamiento y evaluación. Las versiones de etiqueta también pueden diferir sobre la misma imagen. En esta investigación se adoptó una política conservadora de exclusión y agrupación, sin convertir similitud visual en identidad de paciente ni resolver discrepancias mediante una adjudicación automática.

El problema consiste, por tanto, en determinar qué desempeño conserva un detector al aplicar controles reproducibles de datos y al someterlo a condiciones de evaluación que exceden la validación usada para seleccionar parámetros. El estudio incorpora también una aplicación para comprobar el trayecto completo desde el archivo de entrada hasta la visualización. Esa comprobación operativa no sustituye un ensayo de utilidad clínica.

### Pregunta de investigación

¿Qué desempeño por clase y qué robustez alcanza un detector de siete hallazgos dentales, optimizado con una cohorte de anotaciones publicadas y evaluado en prueba interna, imágenes de origen externo y escenarios simulados, al integrarse en una aplicación local reproducible?

### Objetivo general

Desarrollar, optimizar y evaluar retrospectivamente un sistema académico de detección de hallazgos en radiografías panorámicas, mediante aprendizaje por transferencia, controles de integridad de datos y una interfaz local, delimitando el alcance de sus resultados frente a las anotaciones de origen.

### Objetivos específicos

1. Definir y conservar una taxonomía de siete clases con correspondencias verificables respecto a la fuente y documentar las limitaciones del criterio de caja.
2. Recuperar y analizar el corpus, registrar problemas de integridad, excluir ambigüedades y separar los grupos de similitud entre las particiones de desarrollo y evaluación.
3. Ejecutar un baseline y una búsqueda de hiperparámetros con presupuesto explícito, y estudiar sensibilidad, dependencia parcial e interacciones sobre los ensayos realizados.
4. Confirmar el candidato con varias semillas y comparar configuraciones de arquitectura y aumento de datos bajo condiciones documentadas.
5. Evaluar detección por clase, incertidumbre, errores, tamaño de objeto y robustez; fijar el umbral operativo en validación antes de abrir las pruebas reservadas.
6. Integrar y comprobar una aplicación local con el mismo preprocesamiento, controles de entradas y comunicación de límites, y analizar privacidad, equidad y responsabilidad.

### Justificación

El aporte académico se sitúa en la relación entre datos, experimentos y conclusiones. Un entrenamiento exitoso solo demuestra que un algoritmo puede ajustar sus parámetros a unos archivos. Para interpretar el desempeño es necesario identificar qué anotaciones actuaron como referencia, cómo se separaron las particiones y qué decisiones se tomaron después de observar la validación. El sistema conserva esos elementos mediante contratos, manifiestos, configuraciones y hashes.

La evaluación por clase responde al costo asimétrico de los errores. Una caja omitida en una región compatible con lesión y una caja sobrante sobre una restauración no tienen la misma interpretación profesional. Esta investigación priorizó caries y lesión periapical en la función de búsqueda, pero mantuvo métricas para las siete categorías. Así evita que la presencia de tratamientos relativamente fáciles determine por completo la conclusión.

La aplicación convierte decisiones metodológicas en comportamiento observable: acepta formatos concretos, rechaza entradas inválidas, aplica una transformación conocida y muestra puntuaciones con una explicación de su significado. Este recorrido permite comprobar fallos que un archivo de métricas no revela. También facilita una demostración educativa sin instalar servicios externos ni transferir las radiografías cargadas a un proveedor de inferencia.

La relevancia social permanece como una motivación para investigar. El estudio no midió reducción de errores humanos, ahorro de tiempo de lectura, cambios terapéuticos ni desenlaces de pacientes. Sus resultados pueden orientar investigación y docencia; no permiten atribuir beneficios asistenciales ya obtenidos.

### Alcance y delimitaciones

La tarea comprende detección de cajas en panorámicas de adultos con dentición permanente, dentro de las condiciones de la fuente. Se excluyen segmentación, numeración dental completa, identificación de pacientes, evaluación pediátrica, planificación terapéutica y modalidades distintas de la panorámica. Las imágenes de origen externo provienen de otros centros descritos por la fuente, pero continúan perteneciendo al mismo recurso publicado y a su proceso de anotación (Mureșanu et al., 2025).

El referente son las etiquetas publicadas. No se construyó una nueva verdad clínica mediante lectores independientes, examen del paciente o confirmación histológica. La evaluación cuantifica coincidencia espacial y de categoría con ese referente. Los grupos de similitud reducen una forma observable de fuga, pero la independencia por paciente no está establecida.

La selección conservadora excluye una proporción importante del corpus. Por consiguiente, los resultados corresponden a la cohorte aceptada y no a todos los archivos originales ni a una población clínica consecutiva. El tamaño de muestra se determinó por disponibilidad y elegibilidad; no se realizó un cálculo prospectivo de potencia para demostrar superioridad.

### Supuestos y criterio de interpretación

Se consideró plausible que el desempeño difiriera según tamaño, contraste, frecuencia y consistencia de anotación. Estas variables pueden actuar simultáneamente. Las diferencias descriptivas no identifican un efecto causal de una de ellas. La búsqueda y las comparaciones se tratan como exploratorias, con decisiones congeladas antes de abrir las vistas de prueba.

La comparación de configuraciones se interpreta dentro de un presupuesto finito. La mejor combinación entre doce ensayos de diez épocas no constituye el óptimo global ni garantiza el mejor modelo a cincuenta épocas. La repetición de semillas cuantifica una parte de la variación del entrenamiento; no crea nuevas cohortes ni representa incertidumbre de la población por sí sola.

## Metodología

### Enfoque y diseño

La investigación es aplicada, cuantitativa y retrospectiva. Desarrolla un prototipo de detección y mide su comportamiento sobre imágenes previamente recolectadas. Su componente experimental incluye búsqueda de hiperparámetros, confirmación con semillas y alteraciones simuladas de la imagen. No constituye un ensayo prospectivo ni un estudio con lectores que mida el efecto de la asistencia sobre decisiones clínicas.

Se distinguieron tres niveles de evidencia. El registro v1 aporta antecedentes históricos. La cohorte v2 aporta una referencia reproducible de imágenes y anotaciones de origen. Las pruebas de software y de aplicación aportan evidencia de funcionamiento técnico. Ningún nivel se atribuyó automáticamente las propiedades de otro: pasar pruebas de código no establece sensibilidad clínica, y conservar un hash no autentica una aprobación odontológica.

### Unidades de análisis y referente

La radiografía es la unidad de entrada y remuestreo. Cada caja es una instancia de referencia o detección. Una imagen puede contener varias clases y muchas instancias correlacionadas. Por ello, el bootstrap remuestreó imágenes completas y conservó sus referencias y predicciones, en lugar de tratar cada caja como una observación independiente.

Los identificadores de paciente y las visitas no estaban disponibles como metadatos verificables. Se construyeron componentes de similitud visual para separar archivos relacionados según una heurística conservadora. Estos componentes no se denominaron pacientes. El remuestreo por imagen puede subestimar incertidumbre si persiste dependencia entre adquisiciones; se declara esta limitación en los informes y en la interpretación de intervalos.

Las etiquetas originales se utilizaron como anotaciones de origen sin adjudicación local. Las tres personas calibradas y el acuerdo descritos en el recurso corresponden al estudio original, no a una nueva revisión de este proyecto (Mureșanu et al., 2025). La consistencia entre copias de archivo tampoco representa concordancia entre observadores independientes.

### Recuperación y control de integridad

La descarga utilizó el enlace oficial del registro de Zenodo. Se comprobaron tamaño y MD5 publicados, se calculó SHA-256 y se verificaron los CRC del archivo comprimido. La extracción rechazó rutas que pudieran escapar del destino y enlaces simbólicos. El corpus se almacenó fuera de la carpeta sincronizada de trabajo para conservar los datos localmente.

Cada imagen y etiqueta recibió un hash de procedencia. El análisis exploratorio se ejecutó sobre esos archivos y comprobó emparejamientos, geometría, clases, resolución y duplicados. Los manifiestos conservan la ruta de origen, la partición original, las dimensiones y la transformación aplicada. La vista de investigación es inmutable: una modificación de sus bytes impide verificar el contrato.

### Criterios de inclusión y exclusión

Se incluyeron archivos decodificables con etiquetas YOLO válidas y al menos un objeto de las siete categorías seleccionadas. La sintaxis requirió cinco campos, clase entera conocida, coordenadas finitas, dimensiones positivas y cajas dentro de la imagen. Una etiqueta vacía después del mapeo no se aceptó automáticamente como ejemplo negativo porque la cobertura negativa de las siete clases no estaba verificada.

Se excluyeron bloques históricos con semántica ambigua y remapeos sin confirmar. Las copias exactas con anotaciones distintas se pusieron fuera de la cohorte en conjunto; no se eligió la versión que favoreciera una métrica. En copias exactas concordantes se retuvo una instancia, priorizando la reserva de evaluación cuando correspondía. Las reexportaciones visuales con correlación superior a 0,98 y bytes distintos también se excluyeron por requerir adjudicación.

La agrupación se calculó antes de aplicar exclusiones, mediante miniaturas en gris de 96 × 48 píxeles normalizadas y componentes conexas con correlación superior a 0,90. De esta manera, una copia excluida no podía ocultar una relación observable entre otros archivos de desarrollo y una reserva. El umbral expresa una regla operativa de similitud, sin certificar identidad ni ausencia de relaciones por debajo del umbral.

Las reglas contemplaron además cajas intersectadas por la máscara de bordes y relaciones entre reservas internas y externas. El informe de exclusiones registra una razón final por archivo; no debe interpretarse como un conteo exhaustivo de todas las razones que podrían concurrir en el mismo caso. La política de exclusión se definió sin utilizar predicciones del detector.

### Particionado y preprocesamiento

Las particiones de prueba interna y de origen externo se conservaron desde la organización original, tras aplicar elegibilidad y controles de similitud. Los archivos restantes de desarrollo se reagruparon en entrenamiento y validación con una fracción objetivo de 15 %, semilla 42 y estratificación aproximada por frecuencia de cajas. Ningún componente de similitud aceptado cruza las cuatro particiones.

El preprocesamiento convirtió la imagen a gris de ocho bits, aplicó una máscara negra del 3 % izquierdo, 3 % derecho y 4 % inferior, y redujo el lado máximo a 1280 píxeles conservando proporciones. Las etiquetas permanecieron en coordenadas normalizadas. La transformación tiene versión propia y no se atribuye retrospectivamente a los pesos v1.

La máscara busca reducir texto periférico, pero no garantiza desidentificación completa y puede ocultar anatomía. La exclusión de cajas que la intersectan disminuye un conflicto explícito entre imagen y referencia, a costa de restringir la cohorte. El decodificador y la función de preparación son compartidos por la construcción del corpus y la inferencia de archivos originales. Una prueba comprobó igualdad de los derivados obtenidos por ambos recorridos.

### Entrenamiento y selección

El protocolo se guardó con hash y fecha antes de iniciar el ajuste. Se empleó aprendizaje por transferencia desde pesos COCO oficiales. El piloto verificó funcionamiento, pérdidas, recursos y duración sobre imágenes reales. La búsqueda posterior utilizó TPE con semilla 42, doce ensayos de diez épocas y cuatro hiperparámetros. Su objetivo fue el AP50 medio de caries y lesión periapical en validación.

En cada corrida, Ultralytics conservó el checkpoint con mejor mAP50–95 global de validación, de acuerdo con su función de fitness instalada. La función de búsqueda se calculó sobre ese checkpoint. Por tanto, el criterio de conservar una época y el criterio de comparar ensayos no son idénticos. Esta elección aprovecha el flujo estándar, pero puede omitir una época que maximice específicamente las dos clases patológicas.

El candidato de la búsqueda se entrenó con las semillas 42, 43 y 44, hasta cincuenta épocas con paciencia 15. Se ejecutaron también un baseline de configuración explícita, una comparación YOLO26n con los parámetros seleccionados y una ablación con mosaic 0,5. Estas comparaciones usan validación; no se seleccionó arquitectura ni semilla a partir del resultado en prueba. El modelo final para la aplicación y las reservas se fijó como YOLO11n candidato con semilla 42.

### Métricas y umbral operativo

Una coincidencia exige clase correcta y un emparejamiento uno a uno por IoU. Se calcularon AP50 y AP50–95, este último con diez umbrales de IoU entre 0,50 y 0,95. AP utiliza envolvente de precisión, 101 niveles de recall e integración trapezoidal según la versión instalada de Ultralytics. La integración y el emparejamiento se contrastaron mediante pruebas numéricas. La inferencia de evaluación conservó puntuaciones desde 0,001, IoU de NMS 0,70, máximo 300 detecciones y letterbox cuadrado con rect=False.

Precisión, recall, F1 y falsos positivos por imagen se calcularon en un umbral fijo. Se exploró en validación una cuadrícula de 0,05 a 0,80, con pasos de 0,05, maximizando el F1 medio de las dos clases patológicas; un empate favoreció el umbral mayor. La decisión quedó registrada antes de generar predicciones de prueba interna y externa. Esas medidas operativas no son intercambiables con precisión y recall que un validador informa al elegir otro punto de su curva.

La aplicación y las predicciones almacenadas reutilizan el posprocesamiento de validación de Ultralytics, incluida la política multi_label de NMS. Así se evita comparar métricas de una política con una interfaz que utiliza otra. Esta política puede conservar propuestas de categorías diferentes para una región, sujetas a la supresión y al umbral fijados.

Antes de abrir las reservas se contrastaron las seis configuraciones con su validación nativa, tanto en mAP50 y mAP50–95 globales como en AP por clase. La tolerancia absoluta fue 0,0001. Este control incluye la arquitectura de comparación, cuyo flujo de salida puede diferir del de YOLO11. La versión final y el umbral ya estaban fijados y no se reeligieron durante ese contraste.

La ausencia de referencia de una clase se representó como no estimable para recall y AP, no como desempeño perfecto. Los intervalos globales del bootstrap conservaron el estimando de siete clases: una réplica sin alguna clase de referencia no aportó un mAP de composición diferente. Los informes registran cuántas réplicas válidas sustentan cada intervalo.

### Incertidumbre y comparaciones

Se utilizaron mil réplicas de bootstrap por imagen, semilla 42 e intervalos percentiles del 95 %. En comparaciones pareadas, los dos modelos compartieron exactamente imágenes, referencias e índices remuestreados. Se recalcularon AP y métricas agregadas en cada réplica. Los intervalos de desarrollo no corrigen el optimismo introducido por buscar parámetros en la misma validación.

La media y desviación estándar muestral entre tres semillas describen estabilidad del entrenamiento en la partición fija. No se interpretan como intervalos de generalización a otros pacientes o centros. Las comparaciones adicionales y los escenarios se consideran exploratorios; no se formularon conclusiones confirmatorias a partir de múltiples contrastes sin corrección.

### Robustez y análisis de errores

Las pruebas de estrés se aplicaron exclusivamente a validación después de seleccionar el modelo. Se alteraron desenfoque, ruido, compresión JPEG, brillo y resolución, conservando dimensiones y coordenadas de referencia. El ruido fue reproducible por identificador de imagen para permitir comparaciones pareadas. Los escenarios representan degradaciones controladas, sin afirmar que reproduzcan toda la variación de una adquisición real.

Las transformaciones se aplicaron al derivado con lado máximo 1280, antes del letterbox. Se usó desenfoque gaussiano con σ de 1,2 y 2 píxeles; ruido normal aditivo con σ de 5 y 12 niveles de intensidad de ocho bits; JPEG con parámetro de calidad 40; brillo multiplicado por 0,7 y 1,3 con recorte al rango 0 a 255; y reducción al 50 % seguida de reconstrucción a las dimensiones originales. La reducción utiliza interpolación de área y la reconstrucción, bilineal. Estos parámetros describen operaciones de imagen, no una calibración física del equipo radiográfico.

El análisis de error contabilizó objetos omitidos y detecciones sobrantes por clase e imagen. La matriz de confusión empleó emparejamiento espacial independiente de clase para distinguir confusiones de categoría y fondo. Ese emparejamiento tiene un propósito diferente al de las métricas primarias, que sí exigen coincidencia de clase. Los errores se refieren al referente publicado; una detección contada como falsa positiva podría requerir revisión profesional antes de considerarse clínicamente incorrecta.

Se describió recall según fracción de área del objeto y se evaluó confiabilidad de las puntuaciones entre detecciones con score de al menos 0,05. Esta última comparación usa como objetivo el emparejamiento a IoU 0,50. No incluye lesiones omitidas en su denominador ni calibra probabilidad de enfermedad por paciente.

### Verificación técnica y reproducibilidad

El cierre incluyó pruebas automatizadas, verificación de hashes, comparación de métricas con el validador oficial, exportación ONNX y llamadas HTTP con los pesos finales. La comprobación ONNX comparó tensores de salida FP32; la aplicación conserva PyTorch y su posprocesamiento verificado. No se atribuyó a una prueba de tensores la equivalencia de cualquier implementación de NMS.

Los registros conservan configuración solicitada y efectiva, semilla, entorno, hashes de datos y pesos, recursos y estado de cada corrida. El repositorio tenía cambios sin commit al comenzar, por lo que el SHA de HEAD se acompaña de una instantánea del código efectivo. La reproducibilidad se plantea sobre ese conjunto de artefactos y las condiciones de acceso al corpus, sin asumir identidad bit a bit entre plataformas o versiones distintas.

## Cronograma y seguimiento

No se conserva un cronograma inicial fechado para medir desviaciones del trabajo histórico. La fase v2 sí conserva un protocolo congelado antes de ajustar pesos. La tabla distingue hitos documentados y el orden metodológico realmente ejecutado; no reconstruye fechas planificadas inexistentes.

| Hito | Criterio de planificación conservado | Ejecución documentada |
|---|---|---|
| Taxonomía | Registro histórico | 28 de septiembre de 2026 |
| Auditoría de entorno | Registro histórico | 6 de octubre de 2026 |
| Baseline v1 | Registro histórico | 10 y 11 de octubre de 2026 |
| Recuperación y cohorte v2 | Antes de ajustar pesos | 20 de octubre de 2026, hashes y manifiesto |
| Búsqueda | Doce ensayos de diez épocas | Ejecutada, resultados por ensayo |
| Confirmación | Tres semillas y comparaciones previstas | Ejecutada, seis corridas |
| Evaluación reservada | Después de fijar modelo y umbral | Prueba interna y origen externo evaluados |
| Robustez y despliegue | Pesos y umbral fijos | Ocho escenarios, exportación y API comprobados |
| Documentación | Derivada de los artefactos completados | Tesis, guía, cuadernos y manual actualizados |

Recursos utilizados: corpus autorizado en almacenamiento local, Python y bibliotecas fijadas, RTX 5070 Ti Laptop y RAM de 16 GB. Los tiempos y memoria están en run.json, sin imputar costos de nube que no se contrataron. El uso clínico prospectivo queda fuera del estudio retrospectivo concluido. Las diez preguntas oficiales se recibieron el 26 de octubre. El usuario confirmó el reparto en pareja: Cristhian responde las impares y Anthony las pares; el guion y las evidencias se conservan en docs/defensa/. La grabación personal en cámara todavía no existe.

## Referencias

Mureșanu, S., Hedeșiu, M., & Iacob, L.-M. (2025). *Dataset for automating dental condition detection on panoramic radiographs* (Versión 1) [Conjunto de datos]. Zenodo. https://doi.org/10.5281/zenodo.15487430
