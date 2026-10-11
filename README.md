# Dental X-Ray AI

Modelo de visión por computadora para la detección de hallazgos dentales en radiografías panorámicas.

> **Estado:** inicialización del repositorio. Todavía no hay datasets descargados, ni modelo entrenado, ni aplicación.

---

## 1. Objetivo del proyecto

Desarrollar un modelo de detección de objetos capaz de analizar una radiografía dental panorámica y localizar, con su correspondiente caja delimitadora (*bounding box*), múltiples categorías de hallazgos dentales de forma simultánea.

El resultado esperado al final del proyecto es un modelo entrenado y evaluado, reproducible a partir de este repositorio, junto con la documentación completa del proceso de construcción del dataset.

## 2. Problema que intenta resolver

La interpretación de radiografías panorámicas es una tarea manual, repetitiva y dependiente de la experiencia del profesional. Los hallazgos pequeños o de bajo contraste (caries interproximales, lesiones periapicales incipientes) son fáciles de pasar por alto, y la concordancia entre distintos observadores es limitada.

Un modelo de detección automática puede funcionar como **apoyo al diagnóstico**: señalar regiones sospechosas para que el profesional las revise, reducir omisiones y homogeneizar el criterio de revisión.

> **Alcance clínico:** este proyecto es académico. El modelo resultante es una herramienta de asistencia y **no** constituye un diagnóstico médico ni sustituye el criterio de un profesional de la salud.

## 3. Clases inicialmente consideradas

| Clase | Descripción breve |
|---|---|
| `caries` | Lesión de tejido dental duro por proceso cariogénico. |
| `lesion_periapical` | Radiolucidez asociada al ápice radicular. |
| `diente_impactado` | Pieza que no erupciona a su posición normal. |
| `restauracion` | Material restaurador (amalgama, resina, incrustación). |
| `endodoncia` | Tratamiento de conducto / obturación radicular. |
| `corona` | Prótesis fija de recubrimiento total. |

Estas seis clases son **provisionales**. La taxonomía definitiva se fijará tras el análisis de los datasets públicos disponibles; ver [`docs/taxonomy.md`](docs/taxonomy.md).

## 4. Estrategia de integración de múltiples datasets

No existe un único dataset público que cubra todas las categorías de interés con calidad suficiente. La estrategia es **agregar varios datasets especializados** en un dataset maestro multicategoría:

1. **Recopilación.** Identificar datasets públicos de radiografía panorámica y registrarlos en [`docs/dataset_registry.md`](docs/dataset_registry.md) con su fuente, licencia y formato de anotación.
2. **Análisis individual (EDA).** Para cada dataset por separado: número de imágenes, resolución, distribución de clases, tamaño de las cajas, calidad y consistencia de las anotaciones. Un notebook por dataset.
3. **Verificación de licencias.** Solo se integran datasets cuya licencia permita el uso previsto. El estado de cada uno queda en el registro.
4. **Limpieza.** Descarte de imágenes corruptas, duplicadas o sin anotación utilizable; detección de solapamiento entre datasets.
5. **Mapeo de etiquetas.** Cada etiqueta original se mapea a una clase de la taxonomía propia, o se marca como descartada. El mapeo vive en `configs/data/` y es explícito y versionado: es la pieza crítica del proyecto.
6. **Normalización de anotaciones.** Conversión de todos los formatos (COCO, Pascal VOC, máscaras, CSV) a un formato único de trabajo, y de ahí al formato YOLO. Normalización de nombres de archivo y de coordenadas.
7. **Dataset maestro.** Unión de las fuentes normalizadas, con trazabilidad del dataset de origen de cada imagen.
8. **Particionado.** División train / validation / test estratificada por clase, evitando fugas entre particiones (imágenes del mismo paciente o duplicados no pueden caer en particiones distintas).

Cada paso es un script reproducible bajo `scripts/`, no una edición manual.

## 5. Estructura del repositorio

```
dental-xray-ai/
│
├── configs/                   # Configuración versionada (YAML)
│   ├── data/                  # Mapeos de etiquetas, definición del dataset maestro
│   └── training/              # Hiperparámetros de entrenamiento
│
├── data/                      # NO versionado (ver .gitignore)
│   ├── raw/                   # Datasets tal como se descargaron, intactos
│   ├── external/              # Fuentes auxiliares de terceros
│   ├── interim/               # Intermedios: limpieza, mapeo, normalización
│   └── processed/             # Dataset maestro listo para entrenar
│
├── docs/
│   ├── dataset_registry.md    # Registro de datasets evaluados
│   ├── taxonomy.md            # Definición de clases
│   ├── models.md              # Registro de modelos congelados (v1, v2…)
│   └── glossary.md            # Glosario de términos
│
├── models/                    # NO versionado
│   ├── checkpoints/           # Pesos entrenados (.pt)
│   └── exports/               # Modelos exportados (.onnx, etc.)
│
├── notebooks/                 # EDA y exploración (un notebook por dataset)
│
├── reports/                   # Resultados de evaluación
│   └── figures/               # Gráficas generadas
│
├── runs/                      # Salidas de entrenamiento (NO versionado)
│
├── scripts/                   # Puntos de entrada ejecutables
│   ├── dataset_analysis/      # EDA automatizado, estadísticas
│   ├── dataset_conversion/    # COCO/VOC/CSV -> formato interno -> YOLO
│   └── dataset_validation/    # Verificación de integridad y consistencia
│
├── src/                       # Código reutilizable (importable)
│   ├── data/                  # Carga, mapeo, particionado
│   ├── training/              # Lógica de entrenamiento
│   ├── evaluation/            # Métricas y reportes
│   ├── inference/             # Predicción sobre imágenes nuevas
│   └── utils/                 # Utilidades transversales
│
├── tests/                     # Pruebas (pytest)
│
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

**Criterio `src/` vs `scripts/`:** en `src/` va la lógica, importable y testeable; en `scripts/` van los ejecutables de línea de comandos que orquestan esa lógica. Los notebooks exploran, pero nada que deba reproducirse vive únicamente en un notebook.

## 6. Flujo previsto

```
Datasets públicos
        ↓
Análisis individual
        ↓
Limpieza
        ↓
Mapeo de etiquetas
        ↓
Normalización de anotaciones
        ↓
Dataset maestro
        ↓
Train / Validation / Test
        ↓
Entrenamiento
        ↓
Evaluación
        ↓
Modelo entrenado
        ↓
Inferencia
```

## 7. Tecnologías inicialmente consideradas

| Tecnología | Uso previsto |
|---|---|
| Python | Lenguaje base del proyecto. |
| PyTorch | Framework de deep learning subyacente. |
| Ultralytics YOLO | Arquitectura de detección para el baseline. |
| OpenCV | Lectura y preprocesamiento de imágenes. |
| Pandas | Manejo tabular de anotaciones y estadísticas. |
| NumPy | Operaciones numéricas. |
| Jupyter | Notebooks de análisis exploratorio. |

Las dependencias se irán añadiendo a `requirements.txt` conforme se necesiten; nada se instala por adelantado.

## 8. Puesta en marcha

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Copia `.env.example` a `.env` y ajusta las rutas locales si hace falta.

## 9. Datos y privacidad

Los datasets, las imágenes médicas y los modelos entrenados **no se versionan**: son pesados y las imágenes clínicas tienen implicaciones de privacidad y de licencia. Se mantienen en local, bajo `data/` y `models/`, cuya estructura sí está versionada mediante archivos `.gitkeep`.

Cada dataset debe usarse conforme a su licencia original, registrada en [`docs/dataset_registry.md`](docs/dataset_registry.md).

## 10. Licencia

MIT — ver [`LICENSE`](LICENSE). La licencia cubre el **código** de este repositorio; **no** se extiende a los datasets de terceros, que conservan sus propias condiciones de uso.
"# Denta_AI_model" 
