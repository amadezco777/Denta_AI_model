# Registro de datasets

Inventario de todos los datasets públicos evaluados para el proyecto, se hayan integrado o no. Un dataset descartado **también** se registra, con el motivo del descarte, para no volver a evaluarlo desde cero.

> Estado actual: **vacío**. Todavía no se ha evaluado ningún dataset.

## Tabla de registro

| Dataset | Fuente/Paper | URL | Nº imágenes | Clases | Tipo de anotación | Formato | Licencia | Estado | Observaciones |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

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
