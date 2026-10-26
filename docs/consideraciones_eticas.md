# Consideraciones éticas de Dental AI

Cristhian Pazmino Parra · Universidad Espíritu Santo · Maestría en Inteligencia Artificial

Tutora Gladys Villegas · 26 de octubre de 2026

## Impacto social y responsabilidad

### Beneficios potenciales y límites de la evidencia

La localización automática puede servir como recurso docente para discutir hallazgos, comparar propuestas y revisar errores de un detector. Su posible uso como apoyo profesional requiere estudiar cómo personas reales interpretan y corrigen las sugerencias. Este trabajo no midió decisiones asistidas, tiempos de lectura, costos, acceso a atención ni desenlaces de pacientes. Por ello, esos beneficios se presentan como usos a investigar y no como efectos obtenidos.

La integración de evaluación y aplicación aporta un beneficio técnico verificable: permite conocer qué modelo produjo una salida y bajo qué transformación y umbral. Ese control reduce ambigüedad al depurar errores. Su alcance es trazabilidad del prototipo; una predicción trazable puede continuar siendo incorrecta.

### Sesgos de selección y representación

La fuente se concentra en una región de Rumania, adultos con dentición permanente y criterios de calidad diagnóstica. La selección excluyó imágenes con determinadas lesiones extensas, artefactos o errores graves que dificultaran interpretación, según sus autores. La cohorte del proyecto añadió exclusiones por ambigüedad de anotación y duplicación. El comportamiento sobre esta muestra no representa automáticamente la distribución de casos en otros servicios o en Ecuador (Mureșanu et al., 2025).

Las exclusiones mejoran ciertas condiciones de comparación, pero pueden retirar casos difíciles o poco consistentes. El rendimiento estimado puede ser más favorable que el de una secuencia clínica sin selección. Conservar el número y motivo de exclusiones permite evaluar ese sesgo, sin eliminarlo. La prueba de imágenes de origen externo amplía el contraste de adquisición dentro del recurso; no resuelve toda la representatividad geográfica o asistencial.

### Equidad

No se dispuso de metadatos verificables de edad individual, sexo, grupo étnico, situación socioeconómica o comorbilidad. No es posible calcular igualdad de oportunidad o comparar error por esos grupos con validez. Inferir atributos sensibles a partir de la radiografía para completar una tabla introduciría otra fuente de error y no se realizó.

Se evaluaron variables observables relacionadas con adquisición y representación, como origen interno o externo y tamaño de objeto. Esos análisis describen heterogeneidad técnica; no son una certificación de equidad demográfica. La exclusión de fuentes con cobertura incompatible y la separación de grupos de similitud reducen fallos de evaluación, pero tampoco demuestran trato equitativo entre poblaciones.

Una validación destinada al uso asistencial necesitaría grupos definidos con pertinencia clínica, permisos adecuados y tamaños suficientes. Cualquier diferencia debería interpretarse junto con cobertura y calidad de la referencia. Un grupo pequeño con intervalo amplio no permite concluir ausencia de inequidad.

### Privacidad y condiciones de acceso

La fuente declara desidentificación y una exención de consentimiento aprobada para su estudio retrospectivo (Mureșanu et al., 2025). Esas declaraciones corresponden a los autores y su comité; el proyecto no se atribuye ese aval institucional como una aprobación nueva. El tratamiento local mantiene datos y derivados fuera del repositorio de publicación y conserva únicamente los informes agregados necesarios para comunicar resultados.

La máscara de bordes reduce parte del texto periférico, pero no garantiza retirar todos los identificadores incrustados. Antes de cargar imágenes ajenas a la fuente, el usuario debe contar con autorización y anonimización adecuada. La interfaz lo comunica y evita guardar permanentemente los archivos recibidos. El servicio local no implementa un sistema hospitalario de autenticación, auditoría de usuarios, historias clínicas ni respaldo clínico.

Las medidas técnicas descritas no constituyen certificación de cumplimiento jurídico. La eventual incorporación de datos asistenciales propios requeriría determinar las obligaciones aplicables a la institución, el fundamento de tratamiento, los responsables y la retención permitida. No se declara conformidad automática con una regulación por haber usado un corpus público o una conexión local.

### Marco ecuatoriano de protección de datos y salud

El contexto de aplicación propuesto es Ecuador. La LOPDP distingue los datos de salud y establece condiciones para su tratamiento e investigación; sus artículos 25, 30 a 32 y 42 orientan la revisión de categorías especiales, base de tratamiento y evaluación de impacto. La disponibilidad de un archivo en Internet y una máscara de bordes no acreditan anonimización jurídica ni autorización local (Asamblea Nacional del Ecuador, 2021).

El Reglamento General desarrolla ese régimen, incluido el ámbito del tratamiento y la evaluación de impacto. Antes de incorporar datos de una institución ecuatoriana deben documentarse finalidad, base de legitimación, roles, acceso, conservación, derechos y la evaluación de impacto cuando corresponda. Las competencias y disposiciones de la SPDP deben consultarse con la normativa secundaria vigente. Es una agenda de verificación, no una certificación obtenida por Dental AI (Presidencia de la República del Ecuador, 2023; Superintendencia de Protección de Datos Personales, s. f.).

En investigación en salud, el MSP describe la evaluación ética, metodológica y jurídica y el seguimiento de estudios por CEISH. La eventual fase institucional debe determinar su ruta aplicable antes de iniciarse; la aprobación declarada por los autores del corpus no se transfiere automáticamente al nuevo equipo (Ministerio de Salud Pública del Ecuador, s. f.).

La ARCSA es la autoridad de referencia para dispositivos médicos. La Resolución ARCSA-DE-2026-003-DASP contempla software con fines médicos, pero su disposición final fija nueve meses desde la suscripción del 28 de abril de 2026: su entrada en vigor está prevista para el 28 de enero de 2027. Al cierre del 26 de octubre de 2026 se distingue esa transición del régimen aún aplicable y no se presenta la resolución futura como vigente. La clasificación del producto y sus requisitos dependen del uso previsto y requieren verificación competente; no se atribuyen a este prototipo registro sanitario o autorización clínica (Agencia Nacional de Regulación, Control y Vigilancia Sanitaria, 2026).

GDPR, HIPAA, FDA y normas colombianas no sustituyen el análisis ecuatoriano ni se declaran aplicables automáticamente. La licencia de datos, la AGPL de dependencias y la regulación sanitaria son obligaciones distintas. La versión local permanece en investigación y educación no comercial.

### Transparencia y riesgo de automatización

El modelo devuelve categorías, cajas y puntuaciones. La precisión aparente de una cifra puede inducir confianza excesiva si se interpreta como probabilidad clínica. La interfaz incluye una explicación explícita y comunica que la ausencia de detecciones no descarta lesión. El umbral responde a un criterio de F1 de desarrollo, sin incorporar una valoración clínica individual del costo de error.

Los errores pueden provocar omisiones, revisiones innecesarias o distracción de otras regiones de la imagen. Las siete categorías tampoco cubren toda la patología observable en una panorámica. Por eso se excluyen diagnóstico autónomo, triaje, descarte de enfermedad y recomendaciones terapéuticas del uso previsto. La orientación de la Organización Mundial de la Salud (2021) sustenta mantener supervisión, transparencia y rendición de cuentas en sistemas de IA para salud.

La explicación espacial de una caja facilita inspección, pero no prueba causalidad ni razonamiento clínico del sistema. La tesis comunica esa diferencia y conserva ejemplos de error como elementos de análisis. No presenta una visualización atractiva como evidencia de seguridad.

### Licencias y difusión

El recurso de Zenodo limita el uso a investigación y educación no comercial, prohíbe compartir la base o proporcionar acceso a terceros y restringe la presentación de imágenes a los usos de investigación indicados en sus condiciones. Por ello, los datos originales y sus vistas no se incluyeron en una publicación automática del repositorio. Los interesados deben acceder a la fuente oficial y aceptar sus términos (Mureșanu et al., 2025).

La licencia del código propio, las condiciones de los datos y las licencias de dependencias y pesos se interpretan por separado. Ultralytics ofrece AGPL-3.0 y condiciones Enterprise; una licencia MIT del repositorio no elimina las obligaciones de sus componentes (Ultralytics, s. f.). El prototipo se delimita a investigación no comercial y no se presenta como producto con autorización de explotación.

La entrega local conserva el código, resultados agregados y documentos necesarios para revisión. La publicación de un repositorio, la apertura de su visibilidad o la difusión de grabaciones con imágenes requiere respetar permisos y destinatarios. No se cambiaron esas condiciones como consecuencia automática de una consigna académica.

### Responsabilidad y seguimiento

La responsabilidad académica de las decisiones y de la entrega corresponde al autor. Los autores de la fuente y los proveedores de bibliotecas no validan por su mera participación el uso propuesto por este proyecto. La trazabilidad permite reconstruir una salida y retirar una versión de la demostración si se detecta un fallo.

Ante una discrepancia se deben conservar versión de modelo, parámetros y descripción del problema sin incorporar identificadores innecesarios. Una corrección de etiquetas o cambio de preprocesamiento define un experimento nuevo y no reemplaza silenciosamente métricas anteriores. Los controles implementados rechazan pesos y datos modificados respecto al registro congelado.

El estudio no aporta vigilancia prospectiva ni un plan institucional aprobado de incidentes clínicos. Su mecanismo operativo de responsabilidad consiste en limitar el uso, mostrar advertencias, conservar evidencia y mantener versiones diferenciadas. Las recomendaciones de investigación clínica se derivan de los límites observados y no forman parte de funcionalidades ya certificadas.

### Priorización de riesgos y controles verificables

La matriz siguiente prioriza riesgos del prototipo y de su posible uso fuera de alcance. Severidad S: 1, efecto reversible menor; 2, perjuicio relevante o carga de revisión; 3, daño potencial grave a la salud o derechos. Plausibilidad P: 1, limitada al contexto examinado; 2, posibilidad razonable sin frecuencia medida; 3, mecanismo observado en las pruebas técnicas. El producto S por P expresa prioridad: 1–2 baja, 3–4 moderada, 6 alta y 9 crítica. Es una escala de juicio de ingeniería adoptada después de la evaluación, no una probabilidad clínica, una medición de riesgo residual ni una norma sanitaria.

**Tabla 1**

*Priorización de riesgos y estado del control*

| ID y riesgo | Afectados | S | P | Prioridad | Estado y evidencia |
| --- | --- | --- | --- | --- | --- |
| R1 Omisión y falsa tranquilidad | Personas evaluadas y revisores | 3 | 3 | Crítica 9 | Recall interno 0,387 y 0,388; aviso implementado, sin eficacia clínica medida |
| R2 Propuestas adicionales | Revisores y personas evaluadas | 2 | 3 | Alta 6 | FP por imagen 7,021 interno y 8,041 externo; tabla y CSV implementados |
| R3 Representación insuficiente | Poblaciones fuera de la fuente | 3 | 2 | Alta 6 | Origen restringido y sin metadatos demográficos; equidad no estimable |
| R4 Exposición de imágenes | Titulares de datos | 3 | 2 | Alta 6 | Corpus fuera de OneDrive y del ZIP; máscara no garantiza anonimización |
| R5 Etiquetas, duplicados y trazabilidad | Investigadores y evaluadores | 2 | 2 | Moderada 4 | Grupos separados y hashes verificados; sin adjudicación local |
| R6 Uso fuera de alcance | Usuarios y personas evaluadas | 3 | 2 | Alta 6 | Uso previsto y avisos visibles; sin aprobación para atención clínica |
| R7 Licencias y difusión | Autores e institución | 2 | 2 | Moderada 4 | Condiciones de fuente y dependencias conservadas; repositorio privado |
*Nota.* Elaboración propia. R1 y R2 usan evidencia de error frente a anotaciones publicadas; los niveles no estiman tasas de daño a personas. Un control implementado no demuestra reducción cuantificada del riesgo.

Los siguientes controles fijan verificaciones concretas. Los umbrales de desempeño son propuestas posteriores a este estudio para discutir un nuevo protocolo con profesionales; no fueron criterios preregistrados, no autorizan uso clínico y no se ajustará el modelo sobre las reservas existentes para cumplirlos. Un referente nuevo y una muestra independiente serían necesarios para evaluarlos.

**Tabla 2**

*Agenda propuesta de mitigación y criterios de revisión*

| Control | Valor verificable | Responsable propuesto | Fecha objetivo |
| --- | --- | --- | --- |
| C1 Cobertura patológica | IC95 inferior de recall ≥ 0,70 por clase en referencia futura adjudicada; v2 no cumple | Cristhian Pazmino Parra | 02 nov. 2026 |
| C2 Carga de propuestas | ≤ 5 FP por imagen en muestra futura; v2 no cumple | Cristhian Pazmino Parra | 02 nov. 2026 |
| C3 Archivos | Cero radiografías del corpus en ZIP; 100 % de hashes coincidentes | Cristhian Pazmino Parra | 26 oct. 2026 |
| C4 Uso y versiones | Uso asistencial excluido; bloquear si cambia el hash | Cristhian Pazmino Parra | Antes de cada demo |
| C5 Ruta ecuatoriana | Una ficha de base, roles y ruta institucional antes de datos nuevos | Cristhian Pazmino Parra | 09 nov. 2026 |
| C6 Licencias | Inventario de código, pesos y datos antes de publicación autorizada | Cristhian Pazmino Parra | 09 nov. 2026 |
*Nota.* Las fechas futuras son objetivos propuestos y no compromisos institucionales aceptados. El responsable técnico no sustituye al profesional clínico, responsable de tratamiento, delegado de protección de datos o autoridad competente. No se atribuyen designaciones o aprobaciones a Anthony, la tutora o una institución.

Los FP se cuentan respecto a anotaciones de origen y sobre una cohorte con objetos objetivo; no equivalen a especificidad en radiografías sanas ni a una medición de equidad demográfica. La autoridad para avanzar o detener una fase institucional debe quedar formalmente asignada en su protocolo. Sin esa asignación y las condiciones de acceso y referencia, la propuesta no avanza a uso asistencial.

### Asistencia de inteligencia artificial

Se utilizó asistencia de inteligencia artificial para revisar código, organizar documentación y apoyar la redacción y el análisis. Los resultados cuantitativos proceden de ejecuciones registradas sobre los archivos y modelos identificados. No se generaron revisiones odontológicas, consentimientos, firmas, respuestas de participantes ni resultados experimentales ficticios. La responsabilidad de verificar y presentar el trabajo académico permanece en el autor conforme a las reglas de su programa.

## Referencias

Agencia Nacional de Regulación, Control y Vigilancia Sanitaria. (2026). *Normativa técnica sanitaria sustitutiva para la obtención del registro sanitario, control y vigilancia de dispositivos médicos de uso humano* (Resolución ARCSA-DE-2026-003-DASP). https://www.controlsanitario.gob.ec/wp-content/uploads/downloads/2026/04/Resolucion-No.-ARCSA-DE-2026-003-DASP-Normativa-Tecnica-Sanitaria-Sustitutiva-para-la-obtencion-del-registro-sanitario-control-y-vigilancia-de-dispositivos-medicos-de-uso-humano.pdf

Asamblea Nacional del Ecuador. (2021). *Ley Orgánica de Protección de Datos Personales*. Registro Oficial, Suplemento 459. https://www.gob.ec/sites/default/files/regulations/2025-01/01%20Ley%20Org%C3%A1nica%20de%20Protecci%C3%B3n%20de%20Datos%20Personales.pdf

Ministerio de Salud Pública del Ecuador. (s. f.). *Preguntas frecuentes CEISH*. Recuperado el 26 de octubre de 2026, de https://www.salud.gob.ec/preguntas-frecuentes-ceish/

Mureșanu, S., Hedeșiu, M., & Iacob, L.-M. (2025). *Dataset for automating dental condition detection on panoramic radiographs* (Versión 1) [Conjunto de datos]. Zenodo. https://doi.org/10.5281/zenodo.15487430

Organización Mundial de la Salud. (2021). *Ethics and governance of artificial intelligence for health*. https://www.who.int/publications/i/item/9789240029200

Presidencia de la República del Ecuador. (2023). *Reglamento General de la Ley Orgánica de Protección de Datos Personales* (Decreto Ejecutivo 904). Registro Oficial, Tercer Suplemento 435. https://www.gob.ec/sites/default/files/regulations/2025-01/02%20Reglamento%20General%20a%20la%20Ley%20Org%C3%A1nica%20de%20Protecci%C3%B3n%20de%20Datos%20Personales_0.pdf

Superintendencia de Protección de Datos Personales. (s. f.). *Marco normativo*. Recuperado el 26 de octubre de 2026, de https://www.spdp.gob.ec/marco-normativo/

Ultralytics. (s. f.). *Ultralytics license*. Recuperado el 20 de octubre de 2026, de https://www.ultralytics.com/license
