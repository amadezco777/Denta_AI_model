# Referencias bibliográficas

Papers y fuentes de datos que usamos para justificar las decisiones del proyecto. Cada entrada dice **qué justifica** en nuestro trabajo, no solo qué es: así, al redactar el informe, se sabe dónde citar cada una.

Los datasets en sí (licencia, estado, observaciones técnicas) se registran en [`dataset_registry.md`](dataset_registry.md). Aquí va la bibliografía.

> Regla: solo se registra lo que se ha abierto y comprobado. Lo que está citado pero todavía no se ha leído se marca **(pendiente de revisar)**.

## Papers

### [R1] DENTEX: benchmark de detección y diagnóstico
- **Cita:** Hamamci IE, Er S, Durugol OF, et al. *DENTEX: Dental Enumeration and Tooth Pathosis Detection Benchmark for Panoramic X-ray*. MICCAI 2023. arXiv:2305.19112.
- **URL:** https://arxiv.org/abs/2305.19112
- **Dataset asociado:** DENTEX (Zenodo, https://zenodo.org/record/7812323).
- **Qué justifica en el proyecto:**
  - Es la fuente de nuestro dataset principal y es obligatorio citarla (licencia CC-BY / CC-BY-NC-SA).
  - Explica por qué las cajas de DENTEX encierran el **diente completo**: el problema se plantea de forma jerárquica (cuadrante → número de diente → diagnóstico). Nuestro EDA lo confirma (todas las clases miden ~1,1–1,5 % de la imagen).
  - Sus resultados servirán de referencia para comparar nuestro modelo base.

### [R2] Mureșanu et al. 2024: detección de condiciones dentales con YOLOv8
- **Cita:** Mureșanu S, Hedeșiu M, Iacob L, Eftimie R, Olariu E, Dinu C, Jacobs R. *Automating Dental Condition Detection on Panoramic Radiographs: Challenges, Pitfalls, and Opportunities*. Diagnostics (Basel). 2024;14(20):2336. DOI: 10.3390/diagnostics14202336.
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11507083/
- **Dataset asociado:** [D3] (Zenodo 15487430), ver abajo.
- **Qué justifica en el proyecto:**
  - **Metodología muy cercana a la nuestra:** YOLOv8 sobre panorámicas, con cajas y clases de patología y de tratamiento. Es un precedente directo para elegir YOLO.
  - **Referencia de métricas por clase.** En su prueba interna, caries tiene precisión 0,626 y recall 0,511, y pérdida ósea periodontal 0,285 / 0,169, frente a ~0,88–0,92 en implantes y endodoncias. Respalda nuestra hipótesis de que `caries` será la clase más difícil (ver `taxonomy.md`).
  - **Evidencia del problema de generalizar entre centros:** con 180 imágenes de otros centros, el recall (mAP@50) baja de 0,657 a 0,451. Justifica nuestra preocupación por el atajo por equipo de origen (resoluciones, logos RAYSCAN) y la conveniencia de evaluar con datos de otra fuente.
  - **Parámetros de partida:** entrada de 1024×1024, 75 épocas, *batch* de 8, *patience* de 5. Es una referencia razonable para nuestro primer entrenamiento, dado que nuestros objetos son pequeños.
  - **Criterios de exclusión documentados** (quistes grandes, tumores, artefactos metálicos, desenfoque, errores de técnica). Modelo a seguir para nuestra fase de limpieza.
  - **Protocolo de anotación:** tres anotadores calibrados, acuerdo ICC = 0,91. Es el tipo de calidad que debemos pedir a un dataset.
  - **Advertencia (verificación propia, 2026-09-27):** el dataset publicado en Zenodo tiene duplicados entre train y valid, anotaciones independientes de la misma radiografía que solo coinciden en ~55 % de las cajas, y lotes etiquetados con otro orden de clases (detalle en `dataset_registry.md`). **No sabemos** si el modelo del paper se entrenó con estos mismos archivos, así que sus métricas se citan como referencia, sin afirmar que estén afectadas. Sus métricas de test y de validación externa no tienen fuga por duplicados.

### [R3] Índice periapical (PAI) (pendiente de revisar)
- **Cita:** Ørstavik D, Kerekes K, Eriksen HM. *The periapical index: a scoring system for radiographic assessment of apical periodontitis*. Endod Dent Traumatol. 1986;2(1):20–34. (Cita tomada de memoria: **verificar** antes de usarla en el informe.)
- **Qué justificaría:** nuestra hipótesis de que los valores 3/4/5 del campo `<name>` de Mendeley son grados PAI. El EDA es coherente con ella (el tamaño de caja crece con el grado), pero los autores del dataset no lo confirman. Hasta leer el paper y comparar sus definiciones con nuestras galerías, en el informe se presenta **solo como hipótesis**.

### [R4] OralXrays-9 (dataset bloqueado)
- **Cita:** Chen et al. CVPR 2025. Repositorio: https://github.com/Binz-Chan/CVPR2025_OralXrays-9
- **Qué justifica:** es el único candidato de gran escala con clases de tratamiento. Está registrado porque condiciona el alcance de 3 vs 6 clases. Sin licencia explícita; correo enviado a los autores el 2026-09-26, sin respuesta.

## Fuentes de datos sin paper propio

### [D1] DENTEX, ver [R1].

### [D2] Mendeley: panorámicas con lesiones periapicales
- **Cita:** Do V. *Panoramic radiographs with periapical lesions Dataset*. Mendeley Data, v3, 2024. DOI: 10.17632/kx52tk2ddj.3. Hanoi Medical University. CC BY 4.0.
- **Qué justifica:** fuente de 6.741 cajas de lesión periapical (a nivel de ápice). **No tiene paper asociado:** el criterio de anotación y el significado de los grados 3/4/5 son deducciones propias del EDA y la revisión visual (2026-09-27), y así deben presentarse en el informe.

### [D3] Mureșanu et al.: dataset de 14 condiciones dentales
- **Cita:** Mureșanu S, Hedeșiu M, Iacob L-M. *Dataset for Automating Dental Condition Detection on Panoramic Radiographs*. Zenodo, v1, 2025-05-22. DOI: 10.5281/zenodo.15487430.
- **URL:** https://zenodo.org/records/15487430
- **Paper asociado:** [R2].
- **Qué justifica:** posible fuente de las clases de tratamiento (restauración, endodoncia, corona) sin depender de OralXrays-9. Detalle y riesgos en `dataset_registry.md`.

## Historial

| fecha | cambio |
|---|---|
| 2026-09-27 | Creación. R1 (verificado en arXiv), R2 y D3 (verificados en PMC y Zenodo), R3 (pendiente de revisar), R4, D1, D2. |
