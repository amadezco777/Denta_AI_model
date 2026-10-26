# Etiquetado dental con Label Studio

Guía operativa revisada el 6 de octubre de 2026. El alcance implementado es **detección mediante cajas**, con las siete clases de `configs/data/taxonomy.yaml`, versión `0.2.0`. El corpus no está en esta copia: se necesita su ruta para preparar el primer lote real.

## 1. Abrir el entorno instalado

En este equipo Label Studio 1.23.2 está instalado en Ubuntu/WSL, con Python 3.12. La instalación nativa de Windows encuentra un bloqueo de Application Control al cargar SQLite. El entorno WSL permite trabajar sin modificar esa política.

Desde PowerShell, en la raíz del proyecto:

```powershell
.\scripts\labeling\start_labelstudio.ps1 -Background
```

Abre [Label Studio local](http://127.0.0.1:8080). Crea tu cuenta local en el primer acceso y un proyecto llamado **Denta AI — hallazgos panorámicos**. En Settings → Labeling Interface → Code, pega el contenido de `configs/labeling/dental_detection.xml`. El XML fue validado con el esquema de Label Studio instalado.

El servicio escucha únicamente en `127.0.0.1`. Las imágenes se sirven desde `data/labelstudio/media`, no desde todo el disco. La base SQLite y los adjuntos propios del servidor están en `/home/criscri/.local/share/denta-ai/labelstudio-state`, dentro de WSL. Respalda esa carpeta con el servicio detenido y conserva también los paquetes de imágenes y las exportaciones JSON. El repositorio está en OneDrive: los archivos colocados allí pueden sincronizarse según tu configuración de OneDrive.

En otra instalación Ubuntu/WSL, ejecutar primero:

```bash
bash scripts/labeling/install_labelstudio_wsl.sh
bash scripts/labeling/start_labelstudio_wsl.sh
```

La instalación nativa opcional usa `.venv-labelstudio` y `requirements-labelstudio.txt`. No mezclar sus dependencias con las del entrenamiento. `-Backend native` selecciona esa variante del lanzador, pero el bloqueo SQLite observado en este Windows sigue pendiente.

## 2. Preparar un lote

### Imágenes nuevas, sin anotaciones

Usa imágenes PNG/JPEG de 8 bits, con orientación normalizada. Los DICOM necesitan primero un proceso específico de desidentificación y conversión que conserve la información necesaria; este importador no los convierte automáticamente.

```powershell
.\.venv\Scripts\python.exe scripts/labeling/prepare_labelstudio.py `
  --images "D:\radiografias_desidentificadas" `
  --out data/labelstudio/media/lote01
```

La carpeta de ejemplo debe sustituirse por la ruta real. El script lee subcarpetas, valida imágenes y publica el lote completo sin modificar los originales. Genera nombres opacos, derivados PNG sin metadatos EXIF, `manifest.json`, `tasks.json`, `groups.csv` y `label_config.xml`. Quitar EXIF y cambiar nombres **no elimina texto identificativo incrustado en los píxeles**: revisarlo antes de importar. Los duplicados exactos se rechazan en la entrada nueva para evitar etiquetarlos dos veces.

### Revisar un maestro YOLO existente

```powershell
.\.venv\Scripts\python.exe scripts/labeling/prepare_labelstudio.py `
  --dataset data/processed/master/dataset.yaml `
  --out data/labelstudio/media/master_revision01
```

El maestro debe tener estructura válida, etiquetas presentes y la misma taxonomía. Sus cajas entran como `predictions`, pendientes de revisión. No se convierten en anotaciones humanas aprobadas. Se conservan las particiones; una vista externa mantiene su rol de evaluación.

No importar directamente las 14 clases originales de Muresanu con la interfaz de siete clases. Primero se aplica el mapeo versionado. Los casos en cuarentena por remapeo incierto o desacuerdos requieren un lote de adjudicación explícito; no se recuperan automáticamente del maestro filtrado. Como alternativa, preparar sus imágenes sin sugerencias y anotar las siete clases desde cero, conservando la trazabilidad de origen.

En Label Studio importa **`tasks.json`** del lote. No importes a la vez sus PNG como tareas nuevas: eso duplicaría tareas y perdería los identificadores de correspondencia. Usa el XML del propio lote si la taxonomía es una vista distinta. La ruta del paquete debe permanecer estable durante el etiquetado.

**Conectar el almacenamiento del lote es obligatorio en Label Studio 1.23.2.** En el proyecto, Settings → Cloud Storage → Add Source Storage → Local Files, registra la carpeta de imágenes. En este WSL, para `lote01`, la ruta es:

```text
/mnt/c/Users/crisc/OneDrive/Documents/Denta_AI_model/data/labelstudio/media/lote01/images
```

Mantén desactivada la creación automática de tareas desde archivos y no sincronices el directorio como un segundo lote: las tareas ya vienen de `tasks.json`. La conexión autoriza al proyecto a servir esas imágenes. Activar las variables de entorno sin crear esta conexión produce HTTP 404 aunque los archivos existan. Si trabajas en otra ruta, usa la que corresponda dentro del document root del servidor.

## 3. Protocolo de anotación

La especificación clínica de las cajas está en [taxonomy.md](taxonomy.md). Antes de producción, un odontólogo debe aprobar ejemplos visuales de cada clase y resolver el criterio de `protesis_fija`: prótesis completa frente a cada unidad del puente, y separación de corona sobre implante e implante.

1. Revisar calidad y toda la panorámica con zoom. No rotar la imagen ni las cajas.
2. Dibujar una caja por hallazgo según el criterio de la clase. Las cajas de caries son de lesión; no usar automáticamente cajas del diente entero de DENTEX.
3. Corregir y completar las sugerencias. La presencia de una sugerencia no constituye evidencia clínica.
4. Para **cada clase**, marcar `complete` solo cuando se hayan revisado presencia y ausencia en toda la imagen. `uncertain` conserva la incertidumbre y bloquea la exportación a entrenamiento.
5. Elegir `approved`, `needs_review` o `unusable`; registrar el identificador del revisor y las dudas en notas.
6. Guardar la anotación. Una imagen sin cajas puede ser un negativo únicamente cuando todas sus clases están revisadas y el revisor la aprueba.

Propuesta de arranque: calibrar criterios sobre 30–50 panorámicas y hacer doble anotación independiente de una muestra del 10–20 %, con sobrerrepresentación de clases raras y casos dudosos. Es un diseño inicial propuesto, no un estándar clínico validado. Medir acuerdo por clase y geometría; adjudicar discrepancias con una persona experta. Los porcentajes históricos de acuerdo entre copias del dataset no sustituyen este trabajo.

Los textos de estado son declaraciones del revisor. La versión comunitaria y el convertidor no certifican por sí mismos su identidad o competencia profesional. El JSON completo conserva `completed_by`, y el convertidor registra también el identificador de anotación y las notas.

## 4. Confirmar grupos antes de dividir datos nuevos

Completa `groups.csv` del lote. Sus columnas son:

| Campo | Significado |
|---|---|
| `sample_id` | Identificador generado; no cambiarlo. |
| `group_id` | Mismo identificador seudónimo para todas las imágenes relacionadas que deben permanecer juntas. |
| `group_basis` | `patient_metadata`, `verified_acquisition` o `image_only`. |
| `group_verified` | `true` solo tras comprobar el fundamento; inicialmente `false`. |

No introducir nombres de pacientes. `image_only` reconoce que solo se conoce una imagen aislada: no acredita independencia entre pacientes y no puede usarse para varias imágenes del mismo grupo. Las relaciones por similitud son candidatos conservadores; si cruzan particiones o contradicen la revisión, el flujo se detiene para resolverlas.

Completar los grupos **antes** de exportar imágenes nuevas evita dividir visitas del mismo paciente entre train/val/test. Para un maestro existente las particiones permanecen fijas. Las vistas con cobertura parcial deben mantenerse separadas: un dataset periapical no demuestra ausencia de las otras seis clases.

## 5. Exportar a YOLO y revisar el informe

En Label Studio exporta **JSON completo**, con todas las tareas del lote. El exportador propio conserva metadatos que se perderían con una exportación YOLO directa.

```powershell
.\.venv\Scripts\python.exe scripts/labeling/export_labelstudio.py `
  --bundle data/labelstudio/media/lote01 `
  --export data/interim/labelstudio_lote01.json `
  --groups data/labelstudio/media/lote01/groups.csv `
  --out data/processed/labelstudio_reviewed
```

La salida contiene `dataset/`, `review.csv`, `annotation_audit.json` y `export_report.json`. Las imágenes nuevas se dividen aproximadamente 70/15/15 por grupos; hacen falta al menos tres grupos. La estratificación es aproximada: revisar los conteos y las clases sin ejemplos que señala el informe, especialmente en validación y test. No interpretar como rendimiento medible una clase sin casos de evaluación.

El convertidor rechaza tareas ausentes o duplicadas, anotaciones canceladas, cobertura incompleta, clases desconocidas, polígonos, cajas degeneradas/rotadas, cambios de imagen y discrepancias de dimensiones. Si hay varias anotaciones activas, requiere un JSON de adjudicación `sample_id → annotation_id` mediante `--annotation-ids`; nunca elige silenciosamente la última.

Los estados `needs_review` y `unusable` detienen la exportación del lote. Resuelve la revisión o prepara un lote explícito de casos elegibles, documentando las exclusiones. No borres cajas para convertir un caso dudoso en negativo. Conserva la exportación original sin editarla.

## 6. Crear el contrato y entrenar

Una exportación puede contener grupos pendientes; eso permite inspeccionarla, pero el contrato no se crea hasta completar la revisión correspondiente. El CSV de salida contiene hashes de las imágenes y las etiquetas efectivamente exportadas.

```powershell
.\.venv\Scripts\python.exe scripts/dataset_validation/lock_dataset.py `
  data/processed/labelstudio_reviewed/dataset/dataset.yaml --create `
  --review data/processed/labelstudio_reviewed/review.csv `
  --lock configs/data/contracts/labelstudio_reviewed.lock.json

.\.venv-training\Scripts\python.exe -m src.training.train_research `
  --config configs/training/yolo26_pilot.yaml --check-only
```

No completar aprobaciones ficticias para superar este control. Si los datos cambian, se revisa la nueva versión y se crea otro contrato; el anterior permanece como referencia. Para entrenar y comparar costos, seguir [training_decision.md](training_decision.md).

Referencias técnicas: [almacenamiento local](https://labelstud.io/guide/storage_local), [formato de cajas](https://labelstud.io/tags/rectanglelabels), [exportaciones](https://labelstud.io/guide/export), [preanotaciones](https://labelstud.io/guide/predictions).
