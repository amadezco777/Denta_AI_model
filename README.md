# Dental X-Ray AI

Detección de hallazgos dentales en radiografías panorámicas con YOLO: siete categorías, desde caries hasta implantes, cada una localizada con su caja y su puntuación.

Proyecto integrador de la **Maestría en Inteligencia Artificial de la Universidad Espíritu Santo (UEES)**, asignatura «Proyecto Integrador», con la tutora Gladys Villegas. Entrega: 28 de octubre de 2026.

> **Alcance:** prototipo académico de investigación y docencia. **No** es un dispositivo médico, no da diagnósticos y no sustituye el criterio del odontólogo. Que no haya cajas en una imagen **no** descarta patología.

---

## 1. En una mirada

| | |
|---|---|
| **Problema** | Leer una panorámica es manual y depende de la experiencia: las caries pequeñas y las lesiones periapicales incipientes se pasan por alto con facilidad. |
| **Solución** | Un detector que señala regiones sospechosas para que el profesional las revise. |
| **Datos** | Dataset público de Mureșanu et al. (2025), panorámicas de adultos anotadas por especialistas. |
| **Modelo en producción** | **v1**: YOLO11s a 640 px, con mAP50 de 0,634 en validación. Lo sirve la app clínica desplegada. |
| **Estudio de investigación** | **v2 de investigación**: YOLO11n a 960 px, con mAP50 de 0,722 en test interno y 0,571 en externa. |
| **Producto** | App web clínica con login, pacientes, revisión de hallazgos, reporte firmado verificable por QR y envío por correo. |

## 2. Equipo

| Integrante | Aporte |
|---|---|
| **Anthony Madezco** | Parte técnica: selección y depuración de datos, taxonomía, pipeline y modelo v1, app clínica (API y front) y etiquetador. |
| **Cristhian Pazmiño** | Documento final (tesis APA 7) y estudio de investigación v2: cohorte, búsqueda de hiperparámetros y evaluación completa. |

## 3. Ecosistema del proyecto

El proyecto se reparte en tres repositorios privados:

| Repositorio | Qué es | Estado |
|---|---|---|
| **Denta_AI_model** (este) | Datos, entrenamiento, evaluación y registro de modelos | v1 congelado; v2 de investigación integrado |
| [dental-xray-app](https://github.com/amadezco777/dental-xray-app) | App clínica: API FastAPI (inferencia) + Next.js + Supabase | **Desplegada**: front en Vercel y API en VPS. Sirve el v1 |
| [dental-labeler](https://github.com/amadezco777/dental-labeler) | Etiquetador web para que los odontólogos corrijan cajas y generen datos nuevos | MVP terminado; pendiente de despliegue |

```
 Muresanu (Zenodo)  ──►  Denta_AI_model  ──►  pesos + ficha  ──►  dental-xray-app  ──►  odontólogo
                              ▲                                                            │
                              └───────────  .zip YOLO  ◄──  dental-labeler  ◄──────────────┘
                                         (datos para futuras versiones)
```

## 4. Taxonomía

Hay siete clases, congeladas en la versión `0.2.0` ([`configs/data/taxonomy.yaml`](configs/data/taxonomy.yaml) y [`docs/taxonomy.md`](docs/taxonomy.md)):

| id | Clase | Qué encierra la caja |
|---|---|---|
| 0 | `caries` | La lesión cariosa |
| 1 | `lesion_periapical` | La región apical afectada |
| 2 | `diente_impactado` | La pieza completa |
| 3 | `restauracion` | El material restaurador |
| 4 | `endodoncia` | El tratamiento de conducto |
| 5 | `protesis_fija` | La corona o puente |
| 6 | `implante` | El implante |

El dataset principal es Muresanu. DENTEX, Mendeley y Khurshid se evaluaron y quedaron fuera del entrenamiento porque usan otro criterio de caja o cubren otras clases (ver [`docs/dataset_registry.md`](docs/dataset_registry.md)).

## 5. Modelos y resultados

| | **v1 (producción)** | **v2 de investigación** |
|---|---|---|
| Autor | Anthony Madezco | Cristhian Pazmiño |
| Arquitectura | YOLO11s, 640 px | YOLO11n, 960 px |
| Datos | Maestro: 979 train / 173 val / 98 test / 178 externa | Cohorte estricta: 681 / 121 / 94 / 170 |
| Selección | Comparación n960 contra s640 | 12 ensayos TPE + 3 semillas + YOLO26n + ablación |
| mAP50 validación | **0,634** | 0,633 |
| mAP50 test interno | sin evaluar | **0,722** (IC95 0,683–0,758) |
| mAP50 externa | sin evaluar | **0,571** (IC95 0,473–0,629) |
| Umbral operativo | — | 0,15, fijado en validación |
| Dónde se usa | dental-xray-app (`models/v1/`) | App local de demostración (`app/`) |

> **No son directamente comparables.** Usan cohortes y preprocesamiento distintos. La comparación justa (evaluar el v1 sobre el mismo test del v2) está pendiente.

**AP50 por clase.** Para el v1 se mide en validación; para el v2, en test interno:

| Clase | v1 (val) | v2 (test) |
|---|---|---|
| caries | 0,338 | 0,356 |
| lesion_periapical | 0,262 | 0,467 |
| diente_impactado | 0,857 | 0,921 |
| restauracion | 0,634 | 0,730 |
| endodoncia | 0,692 | 0,892 |
| protesis_fija | 0,753 | 0,805 |
| implante | 0,899 | 0,886 |

**Lectura:**
- Los tratamientos y el diente impactado se detectan bien.
- **Caries y lesión periapical siguen siendo el reto**: el recall está cerca de 0,39 en test, es decir, se escapan unas 6 de cada 10.
- Son también las clases con menor acuerdo entre los anotadores originales (49 % y 40 %), lo que limita el techo alcanzable.
- La referencia publicada (Mureșanu et al., 2024) da un mAP50 de 0,657.

Detalle de cada modelo:
- v1: [`docs/models.md`](docs/models.md)
- v2: [`docs/model_card_v2.md`](docs/model_card_v2.md), [`docs/optimizacion.md`](docs/optimizacion.md) y los informes en [`reports/completion/`](reports/completion/)

## 6. Cómo se construyó

**Pipeline v1** (dataset maestro y baseline):

```bash
python scripts/dataset_conversion/muresanu_dedup.py        # raw -> interim (remapeo, exclusiones, duplicados)
python scripts/dataset_conversion/build_master.py          # interim -> data/processed/master (reparto por paciente, máscara)
python -m src.data.provenance data/processed/master/dataset.yaml   # verifica la huella 2f7eae71…
python -m src.training.train_baseline --config configs/training/baseline.yaml
```

**Pipeline v2 de investigación** (cohorte con contrato externo):

```bash
python scripts/dataset_conversion/recover_source.py --root <carpeta_local>   # descarga y verifica Zenodo
python scripts/dataset_conversion/prepare_research.py                        # cohorte + contrato
python scripts/training/complete_research.py pilot                           # piloto
python scripts/training/complete_research.py experiments                     # TPE y confirmaciones
python scripts/evaluation/evaluate_research.py                               # test, externa, bootstrap, estrés
python scripts/evaluation/verify_deployment.py                               # ONNX y app
```

El v2 se ejecutó con su propio entorno: Python 3.12, PyTorch 2.11 cu128 y Ultralytics 8.4.173, más onnxruntime. Las versiones están fijadas en `requirements-research-cu128.lock`.

Los dos pipelines conviven: el v2 entrena con `src/training/train_research.py` y no modifica el del v1. Los datos del v2 viven en `%LOCALAPPDATA%/DentalAI` (configurable con `DENTAL_LOCAL_ROOT`). Los notebooks `notebooks/01`–`05` recorren el estudio v2 paso a paso.

**Principios aplicados en ambos:**
- Test y externa se evalúan una sola vez, con el modelo ya elegido.
- Las particiones se separan por paciente o por similitud, sin fugas.
- Cada dataset y cada corrida dejan su huella (SHA-256, commit y entorno).
- La misma máscara de bordes se aplica en el entrenamiento y en la inferencia.

## 7. Aplicaciones

- **App clínica (producción):** [dental-xray-app](https://github.com/amadezco777/dental-xray-app). Flujo: subir la radiografía → revisar los hallazgos de la IA (aceptar o descartar) → firmar con contraseña → PDF con QR y SHA-256 → envío por correo → verificación pública en `/verificar/<código>`. Cada doctor ve solo a sus pacientes (RLS en Supabase).
- **App local de demostración (v2):** `app/start.ps1` levanta `http://127.0.0.1:8089`. Permite cargar una imagen, ver las cajas y descargar un CSV. No guarda imágenes.

## 8. Limitaciones

- La referencia son las anotaciones publicadas: no hubo una revisión odontológica local independiente.
- No hay identificadores de paciente verificables. La separación se hizo por similitud de imagen.
- No hay negativos sanos verificados, así que la especificidad clínica no es estimable.
- Las imágenes son de un solo origen (Rumania): no representan a la población ecuatoriana.
- No se midió utilidad clínica, tiempo de lectura ni equidad demográfica.

## 9. Próximos pasos

1. Evaluar el v1 sobre el test del v2 y decidir qué modelo sirve la app.
2. Desplegar el etiquetador para que los odontólogos corrijan y generen datos propios para la siguiente versión.
3. Revisión por 3 odontólogos (criterio de la propuesta), con su feedback documentado.
4. Reforzar caries y lesión periapical con más resolución o *tiling*, y con datos de lesión de Mendeley.

## 10. Estructura del repositorio

```
configs/        taxonomía, mapeos, configuraciones de entrenamiento (v1 y v2)
scripts/        ejecutables: conversión, análisis, entrenamiento, evaluación, etiquetado
src/            lógica importable: datos, entrenamiento, evaluación, inferencia
app/            app local de demostración del v2
notebooks/      estudio v2 en cinco cuadernos
docs/           taxonomía, registro de datasets y de modelos, ética, arquitectura, manual
reports/        informes agregados (sin imágenes de pacientes)
artifacts/runs/ registros de las corridas del v2 (configuración, métricas por época)
tests/          pruebas (pytest)
data/, runs/, models/   NO versionados: datos y pesos quedan en local
```

## 11. Puesta en marcha

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest -m "not integration"     # pruebas rápidas; las de entrenamiento real: pytest -m integration
```

## 12. Datos, privacidad y licencias

- **Muresanu** solo permite uso no comercial en investigación y docencia, y prohíbe redistribuirlo. Por eso las imágenes, los datos derivados y los pesos **no se versionan**, y los repos son privados. Fuente oficial: https://doi.org/10.5281/zenodo.15487430
- **Ultralytics** es AGPL-3.0 (ver [`licenses/`](licenses/)).
- **El código propio** es MIT (ver [`LICENSE`](LICENSE)). Esa licencia no se extiende a los datos ni a las dependencias.
- Uso de IA: el desarrollo contó con asistentes de IA para código y documentación. Los resultados provienen de ejecuciones registradas.
