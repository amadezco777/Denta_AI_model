# Optimización y validación de Dental AI

Cristhian Pazmino Parra · Universidad Espíritu Santo · Maestría en Inteligencia Artificial

Tutora Gladys Villegas · 26 de octubre de 2026

## Optimización y validación en contextos simulados

### Relación con la evaluación de la semana 5

Los materiales de la semana 5 proponen separar métricas, incertidumbre, generalización, sensibilidad y escenarios críticos. En este proyecto se adaptaron esas ideas a detección multiclase. La unidad de coincidencia espacial impide trasladar directamente una prueba de clasificación binaria por paciente. Se utilizaron AP por clase, bootstrap por imagen y comparaciones sobre las mismas referencias, en lugar de aplicar mecánicamente métricas de otro tipo de tarea (*Validación en contextos reales o simulados*, s. f.; *Semana5 Workshops semana5*, s. f.).

El estudio ejecutó una búsqueda real con doce ensayos TPE mediante Optuna (Akiba et al., 2019). Cada ensayo recibió diez épocas, la misma semilla y la misma cohorte de desarrollo. Los archivos de configuración y resultados permiten identificar intentos favorables y desfavorables. Las visualizaciones de importancia y dependencia parcial se derivan de esos ensayos; no representan entrenamientos adicionales ni efectos clínicos.

### Función objetivo y presupuesto

La función objetivo fue la media de AP50 de caries y lesión periapical en validación, calculada sobre el checkpoint conservado por el entrenador. Ambas clases recibieron igual peso, independientemente del número de cajas. Este criterio responde al interés en hallazgos patológicos y evita que la cantidad de restauraciones determine por sí sola la búsqueda. No incorpora costos clínicos individuales ni un margen mínimo de mejora acordado con especialistas.

TPE usó seis ensayos iniciales antes de orientar la exploración con los resultados observados. El presupuesto reducido permite comparar configuraciones con los recursos disponibles, pero favorece las que aprenden con rapidez y puede penalizar tasas de aprendizaje pequeñas que necesitarían más épocas. La confirmación a cincuenta épocas reduce esa limitación para el candidato elegido, sin evaluar todas las configuraciones al presupuesto largo.


La Tabla 1 resume el espacio de búsqueda ejecutado.

**Tabla 1**

*Espacio de búsqueda ejecutado*

| Parámetro | Rango o valores | Muestreo |
| --- | --- | --- |
| lr0 | 0,0002 a 0,003 | Logarítmico |
| weight_decay | 0,00001 a 0,002 | Logarítmico |
| scale | 0,05 a 0,40 | Uniforme |
| imgsz | 640 y 960 | Categórico |
*Nota.* Protocolo congelado antes del ajuste. Doce ensayos completos de diez épocas; AdamW, batch 8 y las demás condiciones comunes permanecieron explícitas.



La Tabla 2 resume el resultado de los doce ensayos.

**Tabla 2**

*Resultado de los doce ensayos*

| Ensayo | lr0 | Decay | Scale | Resolución | AP50 patológico |
| --- | --- | --- | --- | --- | --- |
| 1 | 0,000551 | 0,001540 | 0,306 | 640 | 0,022 |
| 2 | 0,000305 | 0,000014 | 0,353 | 960 | 0,048 |
| 3 | 0,000211 | 0,001705 | 0,341 | 640 | 0,006 |
| 4 | 0,000329 | 0,000050 | 0,234 | 640 | 0,018 |
| 5 | 0,001049 | 0,000021 | 0,152 | 960 | 0,143 |
| 6 | 0,001677 | 0,000029 | 0,230 | 640 | 0,070 |
| 7 | 0,000945 | 0,000118 | 0,052 | 960 | 0,150 |
| 8 | 0,001262 | 0,000267 | 0,153 | 960 | 0,159 |
| 9 | 0,000981 | 0,000401 | 0,386 | 960 | 0,117 |
| 10 | 0,002239 | 0,001922 | 0,228 | 960 | 0,179 |
| 11 | 0,000277 | 0,001120 | 0,230 | 960 | 0,060 |
| 12 | 0,002650 | 0,001011 | 0,251 | 960 | 0,162 |
*Nota.* Elaboración propia a partir de search_trials.json. La métrica objetivo promedia dos AP50 de validación. No se utilizó prueba interna o externa para optimizar.



La Figura 1 registra los ensayos y el mejor valor acumulado.

**Figura 1**

*Trayectoria de la búsqueda TPE*

![Trayectoria de la búsqueda TPE](../output/tesis/figures/v2_08_busqueda.png)

*Nota.* Elaboración propia. El mejor acumulado corresponde al presupuesto de diez épocas; no certifica un óptimo global.

### Configuración seleccionada

El ensayo 10 obtuvo el objetivo mayor, 0,179. Sus parámetros fueron lr0 0,002239, weight decay 0,001922, scale 0,228 y resolución 960. Esa combinación se trasladó a las tres semillas de confirmación. La elección no se modificó después de observar las reservas.

El baseline a 640 mantiene parámetros explícitos de referencia y el candidato refleja una búsqueda conjunta. Por ello, una diferencia entre ambos no permite aislar el efecto de la resolución o de una única tasa de aprendizaje. La ablación mosaic y la comparación de arquitectura responden a preguntas adicionales con el mismo conjunto de desarrollo. La confirmación se presenta junto con los resultados, incluso si una variante adicional alcanza un valor mayor.

### Importancia y dependencia parcial

La importancia relativa se calculó mediante fANOVA sobre los ensayos completos. Este análisis describe la variación aprendida por un modelo de sustitución dentro de los rangos explorados. No cuantifica causalidad, no incorpora la incertidumbre clínica de las etiquetas y puede cambiar con otros ensayos. Una importancia pequeña no demuestra que un parámetro carezca de efecto fuera de estos rangos.


El ranking exploratorio se representa en la Figura 2.

**Figura 2**

*Importancia exploratoria de hiperparámetros*

![Importancia exploratoria de hiperparámetros](../output/tesis/figures/v2_09_importancia.png)

*Nota.* Elaboración propia. fANOVA, 128 árboles, profundidad máxima ocho y semilla 42. Doce ensayos completos.

Para los PDP e interacciones se ajustó un bosque aleatorio de 256 árboles, profundidad máxima cuatro y mínimo dos observaciones por hoja. La validación dejando un ensayo fuera produjo R² 0,632 y RMSE 0,037 sobre el objetivo. El valor describe capacidad predictiva entre estos pocos ensayos, sin validar extrapolaciones ni eliminar la incertidumbre del entrenamiento. Los gráficos se interpretan como tendencias exploratorias.

Un PDP modifica un parámetro sobre los datos de ensayo y promedia la predicción del sustituto respecto a los demás. Esa operación permite visualizar una dependencia marginal estimada. Si el espacio contiene pocas combinaciones o los parámetros se concentran por la adaptación de TPE, el gráfico incluye regiones escasamente apoyadas. Por ello se muestran rangos y valores, sin presentar una curva suave como evidencia de un efecto físico.


La Figura 3 reúne las dependencias parciales estimadas.

**Figura 3**

*Dependencia parcial estimada de la función objetivo*

![Dependencia parcial estimada de la función objetivo](../output/tesis/figures/v2_10_pdp.png)

*Nota.* Elaboración propia. Predicciones del bosque sustituto entrenado con doce ensayos; su calidad fuera de muestra se informa mediante R² y RMSE.

### Interacciones

La superficie de lr0 y scale cambia ambos parámetros, promediando sobre las demás características de los ensayos. Los puntos señalan combinaciones realmente observadas. La variación de color fuera de esos puntos procede del sustituto y no de nuevas corridas. Las mesetas y discontinuidades son compatibles con la estructura de árboles y el soporte reducido; no deben interpretarse como fronteras precisas de desempeño.

La configuración elegida sigue el resultado medido de TPE. El análisis de sensibilidad sirve para describir y cuestionar esa elección, no para alterar parámetros después de mirar test. Un estudio más amplio podría comparar otros presupuestos y repetir ensayos; esta posibilidad es una recomendación de investigación derivada del tamaño del experimento, no una evaluación que se declare realizada.


La Figura 4 distingue la superficie estimada y los ensayos realmente observados.

**Figura 4**

*Interacción exploratoria entre tasa de aprendizaje y escala*

![Interacción exploratoria entre tasa de aprendizaje y escala](../output/tesis/figures/v2_11_interaccion.png)

*Nota.* Elaboración propia. Dependencia parcial bidimensional del sustituto; los puntos indican ensayos observados. La superficie no contiene nuevas mediciones de entrenamiento.

### Resultados de escenarios críticos



La Tabla 3 resume la robustez del candidato en validación.

**Tabla 3**

*Robustez del candidato en validación*

| Escenario | mAP50 | Cambio mAP50 | Cambio recall patológico |
| --- | --- | --- | --- |
| Desenfoque σ = 1,2 | 0,604 | -0,029 | -0,021 |
| Desenfoque σ = 2,0 | 0,509 | -0,123 | -0,142 |
| Ruido σ = 5 | 0,615 | -0,018 | -0,025 |
| Ruido σ = 12 | 0,517 | -0,116 | -0,144 |
| JPEG calidad 40 | 0,627 | -0,006 | 0,002 |
| Brillo × 0,7 | 0,594 | -0,039 | -0,063 |
| Brillo × 1,3 | 0,566 | -0,067 | -0,021 |
| Resolución × 0,5 | 0,618 | -0,015 | -0,011 |
*Nota.* Elaboración propia. Cambio respecto a las mismas imágenes sin degradación, con pesos y umbral fijos. Recall patológico: media de caries y lesión periapical.



La Figura 5 permite identificar las degradaciones con mayor cambio de mAP50.

**Figura 5**

*Cambio de mAP50 bajo degradaciones controladas*

![Cambio de mAP50 bajo degradaciones controladas](../output/tesis/figures/v2_14_estres.png)

*Nota.* Elaboración propia. Ocho escenarios sobre 121 imágenes de validación; no se reentrena ni se recalibra el umbral para cada escenario.

El menor cambio observado de mAP50 correspondió a desenfoque σ = 2,0, con valor -0,123. Cada escenario conservó referencias y dimensiones, y el remuestreo pareado mantuvo los mismos índices de imagen. El cambio describe sensibilidad a esa transformación concreta, sin demostrar cómo se comportaría toda una cadena de adquisición clínica.

La compresión y los cambios de brillo pueden preservar parte de las estructuras grandes y afectar detalles pequeños de otra manera. El desenfoque y el ruido tampoco representan la misma pérdida de información. Por ello se informan resultados separados y no se resume robustez con un único escenario favorable. Los intervalos pareados de cada escenario se conservan en el informe de estrés para distinguir variación de muestra y magnitud observada.

### Entradas ausentes y desplazamiento de distribución

Los archivos vacíos, no decodificables, demasiado grandes o incompatibles se comprobaron como errores de entrada de la aplicación. No se les asignó una detección ficticia ni se mezclaron con la métrica radiográfica. La prueba externa contrasta un origen de adquisición diferente y el análisis por clase muestra el efecto de soportes desiguales. Estas comprobaciones abordan aspectos distintos de datos faltantes, desbalance y cambio de dominio.

La ausencia de variables demográficas o de una referencia adjudicada no se resolvió imputando identidades o diagnósticos. La evaluación de sesgo se limita a variables observables y a una discusión explícita de quiénes quedan fuera de la fuente. Esta decisión mantiene la correspondencia entre los escenarios realmente ejecutados y las afirmaciones del estudio.

## Referencias

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). *Optuna: A next-generation hyperparameter optimization framework*. arXiv. https://doi.org/10.48550/arXiv.1907.10902

*Semana5 Workshops semana5* [Cuaderno Jupyter de ejercicios]. (s. f.). Material de clase proporcionado por el autor.

*Validación en contextos reales o simulados* [Diapositivas de la semana 5]. (s. f.). Material de clase proporcionado por el autor.
