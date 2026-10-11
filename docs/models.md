# Registro de modelos

Modelos que salen de este repositorio para usarse fuera de él. Cada uno queda congelado: sus pesos ya no cambian, y una versión nueva lleva otro número.

Los pesos no se versionan aquí: viven en `runs/` (excluido por `.gitignore`) y se publican en el repo del producto, [dental-xray-app](https://github.com/amadezco777/dental-xray-app) (privado), dentro de `models/<versión>/` junto a su ficha `model.yaml`.

## v1: baseline (congelado el 2026-10-10)

| Campo | Valor |
|---|---|
| Corrida | `runs/baseline/yolo11s_640_s42` |
| Pesos | `weights/best.pt` (19 MB) |
| sha256 | `bffb9a24cb252c2a0f17fe9c562e850a04ad1bc5bce5e33533c1e0b99d75187d` |
| Arquitectura | YOLO11s preentrenado en COCO, *fine-tuning* a 640 px |
| Ultralytics | 8.4.165 (entorno completo en `requirements.lock`) |
| Commit | `dd87c4f` |
| Dataset | maestro `data/processed/master/` (train 979, val 173), huella `2f7eae71f88f5c28051bd7b14785d4bbd134b0e487798d6b4e0ce67d8184db16` |
| Taxonomía | `0.2.0` (7 clases, ver `taxonomy.md`) |
| Configuración | `configs/training/baseline.yaml`: 50 épocas, `patience` 20, batch 8, sin AMP, semilla 42 |
| Preprocesamiento | escala de grises y máscara negra en los bordes (izquierda 3 %, derecha 3 %, abajo 4 %), como en `build_master.py`. **Cualquier uso del modelo debe aplicarlo igual.** |

### Por qué este

Se comparó con `yolo11n_960_s42-2` (mAP50 0,624 en val). La diferencia está dentro del ruido de una sola semilla, así que se eligió el s640 porque su inferencia es más rápida. Las semillas 43 y 44 quedan como opcionales para medir la variación.

### Métricas en validación (173 imágenes, umbral de IoU 0,5)

Recalculadas el 2026-10-11 con `model.val(split="val")` sobre el mismo `best.pt`.

| Clase | Cajas | Precision | Recall | F1 | AP50 | AP50-95 |
|---|---|---|---|---|---|---|
| caries | 331 | 0,587 | 0,305 | 0,401 | 0,338 | 0,119 |
| lesion_periapical | 100 | 0,398 | 0,317 | 0,353 | 0,262 | 0,091 |
| diente_impactado | 86 | 0,814 | 0,884 | 0,847 | 0,857 | 0,439 |
| restauracion | 487 | 0,669 | 0,626 | 0,647 | 0,634 | 0,231 |
| endodoncia | 350 | 0,700 | 0,746 | 0,722 | 0,692 | 0,272 |
| protesis_fija | 164 | 0,672 | 0,800 | 0,731 | 0,753 | 0,410 |
| implante | 43 | 0,827 | 0,953 | 0,886 | 0,899 | 0,491 |
| **Todas** | 1.561 | **0,667** | **0,662** | — | **0,634** | **0,293** |

- Los tratamientos (implante, prótesis, endodoncia, restauración) y el diente impactado se detectan bien. Caries y lesión periapical son débiles: se escapan unas 2 de cada 3.
- Esas dos clases también son las de menor acuerdo entre anotadores en Muresanu (49 % y 40 %, ver `taxonomy.md`), lo que limita el techo alcanzable.
- Referencia: el paper de Muresanu da mAP50 0,657 (no es una comparación directa: otras clases y otra partición).
- **Test (98) y externa (178) no se han evaluado.** Se reservan para el modelo final, una sola vez.

### Dónde se usa

- [dental-xray-app](https://github.com/amadezco777/dental-xray-app), `models/v1/`: API de inferencia y front del MVP del proyecto integrador. Unos 200 ms por radiografía en CPU.

### Licencia

El modelo se entrenó con el dataset de Mureșanu et al., que solo permite uso no comercial de investigación y docencia y prohíbe compartir el acceso con terceros. Los pesos solo pueden estar en repos privados. Ultralytics es AGPL-3.0.
