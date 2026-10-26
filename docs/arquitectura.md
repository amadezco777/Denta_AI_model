# Arquitectura de Dental AI

Cristhian Pazmino Parra · Universidad Espíritu Santo · Maestría en Inteligencia Artificial

Tutora Gladys Villegas · 26 de octubre de 2026

## Desarrollo del sistema

### Arquitectura general

El sistema separa preparación de datos, ajuste, evaluación e inferencia. Los módulos de datos validan geometría, procedencia y particiones. El ejecutor de investigación verifica un contrato explícitamente marcado como anotaciones de origen. Los evaluadores almacenan predicciones para recalcular métricas sin volver a entrenar. La aplicación carga únicamente un modelo cuya huella y taxonomía coinciden con sus metadatos.

La ruta de revisión odontológica con Label Studio continúa disponible como un flujo distinto. Sus contratos exigen declaraciones humanas de revisión y cobertura que no se fabricaron para completar el estudio retrospectivo. Si en otra fase se construye una referencia adjudicada, sus datos y resultados deberán recibir otra versión. Un archivo de aprobación generado automáticamente no sustituiría ese trabajo.



El desarrollo y la inferencia comparten los componentes representados en la Figura 1.

**Figura 1**

*Flujo de desarrollo e inferencia del sistema*

![Flujo de desarrollo e inferencia del sistema](../output/tesis/figures/v2_17_flujo.png)

*Nota.* Elaboración propia. El entrenamiento usa una cohorte congelada; la aplicación comparte preparación, geometría y posprocesamiento de evaluación.

### Arquitectura del detector

YOLO11n procesa la imagen mediante una red convolucional con extracción de características, combinación de escalas y una cabeza de detección adaptada a siete clases. El aprendizaje por transferencia conserva parámetros compatibles de COCO y ajusta los de la tarea. Se eligió la variante nano para permitir múltiples corridas en el equipo disponible, sin asumir que su capacidad sea suficiente para todas las lesiones (Ultralytics, s. f.).

La comparación con YOLO26n usa la implementación instalada y los mismos datos y parámetros seleccionados para YOLO11n. Es una comparación exploratoria de arquitectura bajo una configuración transferida; no representa una búsqueda igualmente extensa para ambas familias. El registro histórico YOLO11s se informa como antecedente sobre otra cohorte y no como un participante de esta comparación controlada.

### Aumentos y reproducibilidad

El baseline fijó AdamW, lr0 0,001, weight decay 0,0005, resolución 640 y scale 0,2. Se utilizaron giro de hasta cinco grados, traslación 0,1, inversión horizontal 0,5 y cambios de valor de imagen 0,2, sin inversión vertical ni variaciones de tono o saturación. Mosaic y mixup permanecieron desactivados salvo en la ablación especificada.

La inversión horizontal cambia la lateralidad visual y por ello debe interpretarse respecto a una tarea sin identificación de lado ni numeración. No se extrapola su conveniencia a aplicaciones que sí dependan de esas variables. Los aumentos transforman simultáneamente imagen y cajas a través del entrenador, mientras que los escenarios de estrés de evaluación mantienen la geometría espacial.

Se mantuvieron AMP desactivado, batch 8, workers 0 y caché de imagen desactivada. La decisión limita uso de RAM y conserva una configuración estable en el equipo. El modo determinista y las semillas reducen variación no controlada; no garantizan reproducibilidad absoluta entre versiones de bibliotecas y hardware.

### Entorno y almacenamiento

Las corridas se ejecutaron en Windows con una GPU NVIDIA GeForce RTX 5070 Ti Laptop de aproximadamente 12 GB de VRAM y 16 GB de RAM del sistema. El entorno de investigación utiliza Python 3.12.14, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128 y Ultralytics 8.4.173. Las versiones completas de análisis, API y exportación se conservan en el lock del estudio.

Los datos originales, las vistas y las corridas se guardaron bajo la carpeta local DentalAI de AppData. El repositorio conserva código e informes agregados. Las imágenes médicas y los checkpoints no se publicaron automáticamente. Esta organización disminuye transferencias involuntarias del corpus, pero no equivale a una certificación de seguridad del equipo.

### Aplicación local

La interfaz utiliza FastAPI y HTML con JavaScript. El usuario selecciona una imagen, solicita el análisis y recibe una visualización de cajas, categorías y puntuaciones, junto con una tabla descargable en CSV. La página muestra información del modelo, métricas por conjunto y advertencias de alcance. Los ejemplos precargados se obtienen del corpus local autorizado mediante identificadores cerrados, sin aceptar rutas arbitrarias del navegador.

Se aceptan JPEG y PNG de hasta 12 MB y veinte megapíxeles, con al menos 32 píxeles por lado. Se comprueba la cabecera y se decodifica la imagen antes de inferir. DICOM y otras modalidades quedan fuera del alcance implementado. El indicador de procesamiento bloquea solicitudes repetidas desde la interfaz y los errores se presentan con mensajes comprensibles.

El servicio se vincula a 127.0.0.1. Las imágenes cargadas no se conservan de forma permanente y las respuestas llevan instrucciones de no almacenamiento en caché. Las coordenadas de salida corresponden a la imagen preparada, cuyas dimensiones se informan, no necesariamente al tamaño original. La inversión de la transformación conserva las propuestas numéricas sin recortarlas antes de evaluar IoU; el dibujo queda limitado por la superficie de la imagen. El preprocesamiento y el umbral congelados permanecen visibles en los metadatos.

La advertencia principal indica que las puntuaciones no son probabilidades de enfermedad y que una imagen sin detecciones no descarta patología. No se generan recomendaciones de tratamiento. Las cajas permiten inspección espacial de una propuesta; no explican por sí solas la causa interna de la predicción. No se atribuye al proyecto una implementación de SHAP, LIME o un estudio de explicabilidad con lectores que no se realizó.

### Exportación y comprobación del despliegue

El checkpoint final se exportó a ONNX FP32 con tamaño estático, batch uno y sin NMS embebida. La comparación numérica utilizó imágenes de validación y tolerancias documentadas. Este artefacto permite estudiar portabilidad, mientras que la aplicación entregada continúa usando la ruta PyTorch sometida a pruebas HTTP.

La tolerancia global de tensores fue atol 0,02 y rtol 0,001. Las puntuaciones de clase, cuyo rango difiere del de las coordenadas, se comprobaron adicionalmente con atol y rtol de 0,0001. El informe separa el error absoluto máximo de coordenadas y puntuaciones. Esta comprobación evita aplicar a una confianza pequeña únicamente una tolerancia concebida también para posiciones expresadas en píxeles.

Las pruebas integradas ejecutaron decodificación, preparación, inferencia, supresión de duplicados, dibujo de cajas, codificación PNG y serialización de respuesta. También comprobaron rechazo de archivos inválidos y de identificadores que intentaran representar rutas. La latencia se informa para ese recorrido completo, con calentamiento, mediana y percentil 95, diferenciándola del tiempo de red neuronal aislada.

## Referencias

Ultralytics. (s. f.). *Ultralytics YOLO11*. Recuperado el 20 de octubre de 2026, de https://docs.ultralytics.com/models/yolo11
