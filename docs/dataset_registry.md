# Registro de datasets

Inventario de todos los datasets públicos evaluados para el proyecto, se hayan integrado o no. Un dataset descartado **también** se registra, con el motivo del descarte, para no volver a evaluarlo desde cero.

> Estado actual: **primera revisión sobre fuentes web (2026-09-25). Nada está descargado.** Lo marcado "por confirmar" no se ha podido verificar todavía: no debe tratarse como dato.

## Tabla de registro

| Dataset | Fuente/Paper | URL | Nº imágenes | Clases | Tipo de anotación | Formato | Licencia | Estado | Observaciones |
|---|---|---|---|---|---|---|---|---|---|
| DENTEX | Hamamci et al., MICCAI 2023 (arXiv 2305.19112) | https://zenodo.org/record/7812323 (espejos: Hugging Face `ibrahimhamamci/DENTEX`, GitHub `ibrahimethemhamamci/DENTEX`) | 1.005 totalmente anotadas (705 train / 50 val / 250 test oculto a los participantes) + 693 solo cuadrante + 634 con numeración + 1.571 sin etiquetar | caries, deep caries, periapical lesion, impacted tooth (+ cuadrante y numeración FDI) | bounding box + polígono de segmentación, con jerarquía de 3 niveles (verificado leyendo el JSON el 2026-09-26) | JSON tipo COCO **no estándar** (3 archivos, uno por subconjunto) + PNG. `bbox` = [x, y, ancho, alto] en píxeles; cada anotación trae `category_id_1` (cuadrante), `category_id_2` (numeración) y `category_id_3` (diagnóstico: 0 Impacted, 1 Caries, 2 Periapical Lesion, 3 Deep Caries). El JSON de diagnóstico tiene 705 imágenes y 3.529 anotaciones (recuento por clase pendiente). Al extraer queda anidado (`training_data/training_data/…`). | **Conflicto entre fuentes:** CC-BY-4.0 (Zenodo, confirmado por su API el 2026-09-25) vs CC-BY-NC-SA-4.0 (Hugging Face y GitHub). Se trata como NC-SA (la más restrictiva) | candidato | Es un único dataset con tres enlaces, no tres. **Verificado el 2026-09-26** (MD5 de los dos zip coinciden con Zenodo; contenido listado sin extraer): `training_data.zip` = 705 PNG con diagnóstico + 693 solo cuadrante + 634 con numeración + 1.571 sin etiquetar, más 3 JSON; `validation_data.zip` = 50 PNG **sin ningún archivo de anotaciones**. Con diagnóstico utilizable: **705 imágenes** (no ~755). Los 250 del test no están publicados. Descomprimido: ~14,9 GB. Cubre 3 de nuestras 6 clases; ninguna de tratamiento. |
| OralXrays-9 | Chen et al., CVPR 2025 | https://github.com/Binz-Chan/CVPR2025_OralXrays-9 | 12.688 imágenes / 84.113 instancias (paper y GitHub); espejos de terceros muestran 12.449 y 9.250, sin explicar la diferencia | apical periodontitis, dental caries, wisdom tooth, missing tooth, dental filling, root canal filling, dental implant, porcelain crown, ceramic bridge | Por confirmar (bbox según resúmenes de terceros) | Por confirmar | **Sin licencia explícita** en el repositorio | bloqueado | Único candidato que cubre restauracion, endodoncia y corona. Bloqueado hasta obtener la licencia por escrito de los autores. Correo enviado por el usuario (fecha de registro: 2026-09-26) a los autores correspondientes (Minhua Lu, Yishu Liu; ✉ en el paper) con copia a Bingzhi Chen y Xiaocheng Fang; sin respuesta todavía. Acceso por Google Drive con contraseña. Ojo: `wisdom tooth` no equivale a `diente_impactado` (no toda muela del juicio está impactada). |
| Panoramic radiographs with periapical lesions | Viet Do, Hanoi Medical University. Mendeley Data v3 (2024-04-10), DOI 10.17632/kx52tk2ddj.3 | https://data.mendeley.com/datasets/kx52tk2ddj/3 | 3.926 originales (confirmado en la página); versión aumentada de cantidad no confirmada (17.004 según una tabla externa, sin verificar). Descarga: 8,31 GB en un único archivo comprimido | Lesiones periapicales (etiquetas exactas por confirmar) | Por confirmar (hay una carpeta `Image Annotations`; el formato solo se ve dentro del zip) | Imágenes JPG; anotaciones por confirmar | CC BY 4.0 (confirmado en la página el 2026-09-25) | candidato | Muy prometedor para lesion_periapical. Confirmado: 3 carpetas separadas (`Original JPG Images`, `Augmentation JPG Images`, `Image Annotations`), así que las aumentadas NO vienen mezcladas y basta con usar solo las originales. Pendiente: no está claro si las anotaciones cubren las originales, las aumentadas o ambas; ni el formato (bbox/máscara). Sin paper asociado. Para saberlo hay que descargar los 8,31 GB. |
| Tufts Dental Database | Tufts University (Dr. Karen Panetta). Artículo: "Tufts Dental Database: A Multimodal Panoramic X-Ray Dataset for Benchmarking Diagnostic Systems" (PubMed 34606466; revista y año por confirmar) | https://tdd.ece.tufts.edu/ | 1.000 panorámicas | Clasificación en 5 niveles (localización, bordes, radiodensidad, efecto en estructuras vecinas, categoría de anomalía) | Por confirmar (no se verificó que haya bounding boxes) | Por confirmar | CC BY 4.0 según fuente secundaria; **la web oficial no la declara** | candidato | No se verificó que aporte las clases de tratamiento (endodoncia, etc.) que una tabla externa le atribuía. Prioridad baja hasta revisar sus archivos. |
| DC1000 (repositorio MLUA) | Wang et al., Neurocomputing 2023 | https://github.com/Zzz512/MLUA | Sin dato | caries | Máscaras de segmentación semántica | Por confirmar | **Sin licencia explícita** | bloqueado | Requiere convertir máscara a caja. Enlace de descarga en Google Drive/Baidu; el README aún dice que el enlace se añadirá cuando el paper se acepte. Exige citar el artículo. |
| Multi-center Ningxia (impacted / periodontitis / caries) | Scientific Data 2024 (PMC11031544) | https://github.com/qinxin99/qinxini (acceso por solicitud) | 4.940 + 716 + 880 imágenes de tres hospitales | impacted teeth, periodontitis, dental caries | Segmentación y clasificación (por confirmar) | Por confirmar | No verificada | candidato | Acceso por solicitud a los autores. No se pudo abrir la fuente para verificarlo. |
| Dental Panoramic X-ray YOLO (liodon-ai) | Derivado de DENTEX + OralXrays-9 (compilación de terceros) | https://huggingface.co/datasets/liodon-ai/dental-panoramic-xray-yolo | 9.974 (9.928 train / 46 val); 39.897 cajas | caries, periapical lesion, impacted tooth | bounding box | YOLO | CC-BY-NC-4.0 declarada por el compilador; **no fiable** | descartado | Motivos: (1) mezcla dos fuentes, y usarlo junto a los originales duplicaría imágenes y provocaría fuga de datos; (2) su licencia no puede ser más permisiva que la de sus componentes, y OralXrays-9 no tiene ninguna; (3) elimina las clases de tratamiento; (4) `Impacted tooth` es el 57 % de las cajas y proviene mayormente de `wisdom tooth`; (5) validación de solo 46 imágenes. |
| Restauraciones panorámicas de Qingdao | Estudio retrospectivo, Affiliated Hospital of Qingdao University (PMC13492184) | https://pmc.ncbi.nlm.nih.gov/articles/PMC13492184/ | 2.434 | crowns, root canal filling, fillings, bridges, implants, braces | bounding box (Labelme) | Labelme | No aplica | descartado | No es público: acceso restringido a uso institucional con aprobación ética. Buenas clases, pero inaccesible. |
| Charité Berlin (restauraciones) | Diagnostics 2022, 12(6):1316 (PMC9221749) | https://pmc.ncbi.nlm.nih.gov/articles/PMC9221749/ | 1.781 | fillings, crowns, root canal treatments, implants | Segmentación pixel a pixel | Por confirmar | No verificada | bloqueado | No se logró confirmar que los datos sean públicos. |

## Cómo rellenar cada columna

| Columna | Qué poner |
|---|---|
| **Dataset** | Nombre corto y estable. Se usa como identificador en `data/raw/<dataset>/` y como prefijo de los archivos en el dataset maestro. |
| **Fuente/Paper** | Autores y publicación de origen, o institución. Necesario para la citación. |
| **URL** | Enlace de descarga o página oficial. Anotar si requiere registro o solicitud de acceso. |
| **Nº imágenes** | Total de imágenes; si están preparticionadas, desglosar (`800 (600/100/100)`). |
| **Clases** | Etiquetas **originales** del dataset, tal como vienen, sin traducir ni mapear. |
| **Tipo de anotación** | `bounding box`, `segmentación (polígono)`, `máscara`, `clasificación (imagen)`, `keypoints`, `mixto`. |
| **Formato** | `COCO JSON`, `Pascal VOC XML`, `YOLO TXT`, `CSV`, `PNG mask`, etc. |
| **Licencia** | Licencia exacta (`CC BY 4.0`, `CC BY-NC 4.0`, `MIT`, *uso académico*, *desconocida*). Si es no comercial o desconocida, señalarlo. |
| **Estado** | Ver tabla de estados. |
| **Observaciones** | Calidad de las anotaciones, resolución, sesgos, solapamiento con otros datasets, problemas encontrados. |

## Estados posibles

| Estado | Significado |
|---|---|
| `candidato` | Identificado, todavía sin revisar. |
| `en revisión` | Descargado, en análisis (EDA). |
| `integrado` | Analizado, mapeado y volcado en el dataset maestro. |
| `descartado` | Evaluado y rechazado. **Obligatorio** indicar el motivo en Observaciones. |
| `bloqueado` | Interesante pero no disponible (acceso restringido, licencia incompatible, sin respuesta). |

## Checklist por dataset

Antes de marcar un dataset como `integrado`:

- [ ] Licencia verificada y compatible con el uso previsto del proyecto.
- [ ] Atribución/citación registrada.
- [ ] Copia intacta en `data/raw/<dataset>/`, sin modificar.
- [ ] EDA completado en `notebooks/` (conteo de imágenes, distribución de clases, resolución, tamaño de cajas).
- [ ] Anotaciones validadas: sin cajas fuera de rango, sin archivos huérfanos, sin imágenes corruptas.
- [ ] Mapeo de etiquetas definido en `configs/data/` y documentado.
- [ ] Comprobado el solapamiento con datasets ya integrados (imágenes duplicadas entre fuentes).
- [ ] Script de conversión reproducible en `scripts/dataset_conversion/`.

## Notas sobre licencias

- Algunos datasets de imagen médica son **no comerciales** (`CC BY-NC`). Para un proyecto académico suele ser aceptable, pero debe quedar registrado, ya que condiciona cualquier uso posterior.
- Las licencias de los datasets **no** quedan cubiertas por la licencia MIT de este repositorio. Ver la sección de licencia del [README](../README.md).
- Si la licencia es desconocida o ambigua, el dataset se marca como `bloqueado` hasta aclararla — no como `candidato`.
