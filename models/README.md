# Modelo Dental AI v2

Versión 2.0.0-source-reference. YOLO11n, siete clases, resolución 960, umbral 0.15.

SHA-256: 70b130153628b4564defae0d3542be4edef9eea147e2b2b4096b840761426d2e.

| Conjunto | Imágenes | mAP50 | mAP50-95 | FP/imagen |
|---|---:|---:|---:|---:|
| Validación | 121 | 0,633 | 0,274 | 6,636 |
| Prueba interna | 94 | 0,722 | 0,339 | 7,021 |
| Origen externo | 170 | 0,571 | 0,260 | 8,041 |


Referencia: anotaciones de origen sin adjudicación clínica local. Identidad por paciente desconocida y equidad demográfica no estimable. Investigación y educación no comercial; sin diagnóstico autónomo, triaje, descarte ni recomendaciones terapéuticas. Ver la tesis y docs/consideraciones_eticas.md.

El checkpoint y ONNX locales se identifican en model.json y reports/completion/export.json. Los pesos v1 históricos no se recuperaron; sus métricas se conservan separadas en docs/models.md.

Aviso del framework y los pesos: [licencia de Ultralytics](../licenses/Ultralytics_AGPL-3.0.txt), conservada desde la distribución instalada. La licencia del código propio no sustituye las condiciones de esos componentes.
