# Análisis de datos de Dental AI

Cristhian Pazmino Parra · Universidad Espíritu Santo · Maestría en Inteligencia Artificial

Tutora Gladys Villegas · 26 de octubre de 2026

## Datos y análisis exploratorio

### Fuente y selección de categorías

Se recuperó el recurso publicado por Mureșanu et al. (2025). La fuente original contiene 1.808 archivos distribuidos en 1.375 de entrenamiento, 153 de validación, 100 de prueba y 180 de origen externo. El nuevo análisis reprodujo esos conteos y registró 24.888 cajas en catorce identificadores originales. Encontró tres cajas con ancho o alto no positivo, 42 grupos de copias exactas entre particiones y 145 grupos dentro de una partición. Cero archivos sin etiqueta y cero cabeceras ilegibles describen integridad de emparejamiento, sin acreditar exhaustividad clínica.

Se seleccionaron siete categorías mediante el mapeo 4→0, 7→1, 6→2, 2→3, 3→4, 1→5 y 0→6. Ese mapeo corresponde al orden fuente y al orden canónico de Dental AI. No se aplicó a los bloques cuyo significado no estaba confirmado. Las fuentes DENTEX y Mendeley permanecieron fuera del ajuste multiclase por las diferencias de geometría y cobertura ya descritas en los antecedentes.

### Construcción de la cohorte v2

La política conservadora aceptó 1.066 imágenes y excluyó 742, equivalentes al 41,04 % de los archivos originales. Esta reducción es una característica del estimando: los resultados describen las imágenes elegibles según estas reglas. No se afirma que los archivos excluidos carezcan de valor clínico, sino que no ofrecían una referencia suficientemente determinada para esta comparación automatizada.

Las discrepancias entre copias exactas motivaron la exclusión más numerosa. Comparar listas de cajas permite detectar una diferencia de anotación, pero no decidir cuál versión es correcta. Por ello, el proceso no promedió coordenadas ni fabricó una revisión de consenso. La exclusión conserva una pregunta clínica abierta como limitación del recurso y evita convertirla en una certeza artificial del entrenamiento.


La Tabla 1 resume la composición efectiva de la cohorte de investigación.

**Tabla 1**

*Composición efectiva de la cohorte de investigación*

| Partición | Imágenes | Cajas | Uso |
| --- | --- | --- | --- |
| Entrenamiento | 681 | 7.128 | Ajuste de pesos |
| Validación | 121 | 1.254 | Búsqueda y umbral |
| Prueba interna | 94 | 1.033 | Evaluación reservada |
| Origen externo | 170 | 2.046 | Evaluación reservada |
*Nota.* Elaboración propia a partir del manifiesto y cohort_report.json de la vista v2. Las cajas no equivalen a pacientes.



La Figura 1 permite comparar el tamaño de las cuatro particiones.

**Figura 1**

*Distribución de imágenes aceptadas en la cohorte v2*

![Distribución de imágenes aceptadas en la cohorte v2](../output/tesis/figures/v2_01_cohorte.png)

*Nota.* Elaboración propia. N = 1.066 imágenes. Los grupos son de similitud no verificada como identidad de paciente.



Las discrepancias entre copias concentran la exclusión más frecuente, como muestra la Figura 2.

**Figura 2**

*Motivo final de exclusión de los archivos originales*

![Motivo final de exclusión de los archivos originales](../output/tesis/figures/v2_02_exclusiones.png)

*Nota.* Elaboración propia. N = 742 archivos. Una razón final por archivo; pueden existir problemas concurrentes.

### Distribución y desbalance

El número de instancias difiere entre clases y particiones. Restauración y endodoncia aportan muchas cajas; implante y lesión periapical tienen soportes más reducidos. Un gran número de cajas en una imagen no aumenta en igual medida la diversidad de personas o adquisiciones. Por eso se informa el soporte y se evita interpretar las proporciones de anotaciones como prevalencias epidemiológicas.


La Tabla 2 resume el soporte de referencia por clase y partición.

**Tabla 2**

*Soporte de referencia por clase y partición*

| Clase | Train | Val | Test | Externo |
| --- | --- | --- | --- | --- |
| Caries | 1.426 | 263 | 137 | 294 |
| Lesión periapical | 471 | 88 | 80 | 105 |
| Diente impactado | 371 | 69 | 56 | 111 |
| Restauración | 2.298 | 385 | 385 | 840 |
| Endodoncia | 1.617 | 277 | 261 | 497 |
| Prótesis fija | 760 | 138 | 89 | 183 |
| Implante | 185 | 34 | 25 | 16 |
*Nota.* Elaboración propia. Soporte expresado en cajas aceptadas y no en pacientes con una condición.



La Figura 3 muestra el desbalance de cajas entre categorías y orígenes.

**Figura 3**

*Distribución de cajas de las siete categorías*

![Distribución de cajas de las siete categorías](../output/tesis/figures/v2_03_clases.png)

*Nota.* Elaboración propia. Las barras suman particiones de la cohorte v2 y no representan prevalencia clínica.

### Tamaño de las regiones

La resolución nominal de una panorámica no coincide con la resolución efectiva que recibe la red. El letterbox conserva proporciones dentro de una entrada cuadrada. Una caja pequeña puede reducirse a pocos píxeles en su lado menor, aunque la imagen original tenga varios miles de píxeles de anchura. Se analizó esta geometría sobre entrenamiento para describir la dificultad sin utilizar el resultado de prueba como criterio de diseño.


La Tabla 3 resume la geometría de las cajas de entrenamiento.

**Tabla 3**

*Geometría de las cajas de entrenamiento*

| Clase | Mediana lado menor a 640 | Cajas ≤ 8 px a 640 | Cajas ≤ 8 px a 960 |
| --- | --- | --- | --- |
| Caries | 13,7 | 198 | 76 |
| Lesión periapical | 18,2 | 3 | 0 |
| Diente impactado | 44,9 | 0 | 0 |
| Restauración | 17,5 | 95 | 17 |
| Endodoncia | 16,6 | 100 | 33 |
| Prótesis fija | 32,9 | 1 | 0 |
| Implante | 23,4 | 0 | 0 |
*Nota.* Elaboración propia. Tamaños de referencia calculados con el redimensionamiento y redondeo por eje del validador, sin aumentos aleatorios.

En caries, 198 de 1.426 cajas de entrenamiento tienen un lado menor de hasta ocho píxeles al preparar una entrada de 640. La cifra se reduce a 76 a 960. Esta diferencia justifica explorar resolución, pero no demuestra una mejora de AP: el efecto final depende del aprendizaje, la referencia y el resto de la configuración. Las cajas de diente impactado presentan una escala espacial mayor.


Las diferencias de escala espacial se observan en la Figura 4.

**Figura 4**

*Tamaño efectivo de los objetos de entrenamiento*

![Tamaño efectivo de los objetos de entrenamiento](../output/tesis/figures/v2_04_tamanos.png)

*Nota.* Elaboración propia. Mediana y cuartiles; escala logarítmica y puntos atípicos omitidos del dibujo. La línea de ocho píxeles es una referencia descriptiva, no un umbral clínico. El cálculo conserva todas las cajas.

### Coocurrencia y carga de anotación

Las categorías no son excluyentes. Una misma imagen puede incluir caries y tratamientos, o varios tipos de tratamiento. La matriz de coocurrencia cuenta imágenes con ambas categorías y su diagonal indica presencia de cada clase. No mide asociación causal ni correlación entre personas, y sus valores dependen de la cobertura de anotación disponible.

La concentración de varias cajas por radiografía explica por qué un error de adquisición puede afectar simultáneamente muchos objetos. También impide interpretar un intervalo calculado sobre cajas independientes como si describiera el número de pacientes estudiados. La distribución de objetos por imagen complementa el conteo de imágenes y clases.


La Figura 5 conserva la coocurrencia sin asumir categorías excluyentes.

**Figura 5**

*Coocurrencia de categorías en entrenamiento*

![Coocurrencia de categorías en entrenamiento](../output/tesis/figures/v2_05_coocurrencia.png)

*Nota.* Elaboración propia. Cada celda cuenta imágenes con presencia de las dos categorías; la diagonal es presencia individual.

En entrenamiento, la mediana fue 9 cajas por imagen, con cuartiles 5 y 14, y máximo 45. La regla descriptiva de 1,5 rangos intercuartílicos produjo una cota superior de 27,5 y señaló 17 imágenes. Esos casos no se excluyeron por cantidad: una carga elevada puede reflejar múltiples objetos válidos y no demuestra corrupción del dato.



La carga de cajas por imagen se presenta en la Figura 6.

**Figura 6**

*Cantidad de cajas por imagen de entrenamiento*

![Cantidad de cajas por imagen de entrenamiento](../output/tesis/figures/v2_07_objetos.png)

*Nota.* Elaboración propia. N = 681 imágenes; las instancias de una misma radiografía pueden estar correlacionadas.

### Resolución y origen

Las dimensiones originales varían entre archivos y orígenes. El lado máximo de 1280 de la vista reduce almacenamiento y costo de decodificación, pero constituye una decisión experimental que puede retirar detalle fino. La comparación de 640 y 960 ocurre después de esa transformación común. Por ello, este estudio no evalúa conservar íntegramente la resolución de adquisición en la entrada de la red.


La Tabla 4 resume la anchura original de las imágenes aceptadas.

**Tabla 4**

*Anchura original de las imágenes aceptadas*

| Conjunto | Mínimo | Mediana | Máximo |
| --- | --- | --- | --- |
| Entrenamiento | 804 | 2.652 | 2.976 |
| Validación | 1.758 | 2.652 | 2.896 |
| Prueba interna | 2.382 | 2.775 | 2.902 |
| Origen externo | 2.398 | 2.760 | 5.538 |
*Nota.* Elaboración propia. Píxeles antes de preparar el derivado. Los extremos se conservan en el histograma y no se ocultan mediante límites fijos de visualización.



La Figura 7 compara las distribuciones de anchura original.

**Figura 7**

*Anchura original por partición en la cohorte aceptada*

![Anchura original por partición en la cohorte aceptada](../output/tesis/figures/v2_06_resoluciones.png)

*Nota.* Elaboración propia. Histograma de densidad para comparar formas con tamaños de muestra diferentes. No identifica equipos o pacientes individuales.

### Datos ausentes y alcance del referente

La ausencia de variables demográficas verificadas impide estudiar equidad por edad o sexo. La falta de identificadores clínicos impide demostrar independencia por paciente. Las etiquetas publicadas tampoco permiten distinguir exhaustivamente ausencia y omisión. Estas ausencias se conservan como propiedades del diseño y no se completan por inferencia del modelo.

La nueva cohorte ofrece trazabilidad de sus exclusiones y particiones. Su contrato no pretende corregir todas las limitaciones de la fuente: una anotación con geometría válida puede seguir siendo clínicamente discutible. El análisis posterior se refiere a ese referente delimitado y declara los denominadores utilizados.

La cohorte aceptada contiene al menos una caja objetivo por imagen y no aporta negativos sanos con cobertura exhaustivamente verificada. Por tanto, los falsos positivos por imagen se estiman sobre esa muestra positiva y no establecen especificidad en panorámicas normales. Esta restricción afecta cualquier extrapolación hacia cribado o descarte de enfermedad.

## Referencias

Mureșanu, S., Hedeșiu, M., & Iacob, L.-M. (2025). *Dataset for automating dental condition detection on panoramic radiographs* (Versión 1) [Conjunto de datos]. Zenodo. https://doi.org/10.5281/zenodo.15487430
