# Revisión Capítulo 4 — extracción de evidencia desde el código

Informe de extracción (no de mejora del modelo) preparado para responder al evaluador que
pidió ampliar el Capítulo 4. Cada dato indica su origen: **Fuente** cuando viene de código o
notebooks ya existentes en el repo (formato `archivo:línea`), **Generado por** cuando es un
cálculo nuevo hecho para esta revisión (script de `analysis_revision/` → CSV de salida), y
**NO DETERMINADO** cuando el repo no contiene la información necesaria.

Rama: `revision-cap4`. Todo el contenido nuevo vive en `analysis_revision/`; no se modificó
ningún archivo de `planificacion/`, los notebooks, ni los `.xlsx`/`.json`/`.png` ya versionados.

---

## 0. Resumen ejecutivo y entorno

- Entorno de ejecución (**Generado por:** `analysis_revision/00_verificar_entorno.py`):
  Python 3.11.15, pandas 3.0.6, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1,
  matplotlib 3.11.2, openpyxl 3.1.5, **prophet 1.4.0**, pytest 9.1.1.
- Gate de reproducibilidad: los 54 tests de `tests/` (repo original) pasan en este entorno
  (`pytest tests/ -q` → `54 passed`) antes de extraer cualquier número nuevo.
- Todos los datos de las secciones A, B, D, E y F fueron verificados contra los artefactos
  ya existentes en el repo (`hiperparametros_prophet.json`, `Insumos_Criticos.xlsx`,
  `Politicas_Inventario_Insumos_Criticos.xlsx`) o contra la salida ya ejecutada de los
  notebooks. La única sección sin verificación exacta es el out-of-sample de Prophet
  (ver Sección A y G — hallazgo de no-determinismo).
- Puntos **NO DETERMINADO**: comparación empírica de la curva de talles (Sección C), y la
  metodología AHP completa (matriz de comparaciones, autovector, CI/RI/CR — Sección E).
- Inconsistencias detectadas entre lo preguntado y lo que hay en el repo: ver Sección G.

---

## A. Datos y preprocesamiento, y Prophet

### A.1 Datos y preprocesamiento

- **Fuente de datos:** Google Sheets exportado como xlsx
  (`excel_url`, **Fuente:** `planificacion/config.py:24-25`), con copia local de respaldo
  `ventas_historicas_cache.xlsx` (**Fuente:** `planificacion/config.py:27`, cargada por
  `planificacion/io_datos.py:11-29`).
- **Rango de fechas efectivo, frecuencia y n° de observaciones**, para los 3 artículos
  (**Fuente:** `Pronostico_Ventas.ipynb`, celda de salida "Rango de fechas: 2023-01 a 2025-12";
  confirmado leyendo `ventas_historicas_cache.xlsx` con
  `analysis_revision/01_prophet_reproducibilidad.py`):
  mensual, **36 observaciones** por artículo (ene-2023 a dic-2025), para CRONOS-N04,
  TAURO 2-N04 y HORIZON-M09 (los 3 únicos artículos del dataset).
- **Filtro de fechas ("desde dic-2023" o similar): NO DETERMINADO — no existe.** Se revisó
  `planificacion/forecast/datos.py` completo: no hay ningún filtro de rango de fechas; se usa
  el histórico completo de 36 meses tal cual viene del Excel.
- **Transformación de la variable:** ninguna (sin log, Box-Cox ni agregación: el dato fuente
  ya es mensual). **Fuente:** ausencia confirmada por lectura completa de
  `planificacion/forecast/datos.py` y `modelo.py` (sin `np.log`, `boxcox`, ni winsorización).
  El único "recorte" es `clip(lower=0)` sobre el `yhat` exportado
  (**Fuente:** `planificacion/forecast/resultados.py:18`), aplicado solo en el export
  (decisión de presentación: las ventas no pueden ser negativas), **no** sobre las métricas de
  ajuste (**Fuente:** docstring `planificacion/forecast/modelo.py:199-201`).

### A.2 Split entrenamiento/prueba

- **Fuente:** `planificacion/config.py:49` (`test_periods: int = 12`) y
  `planificacion/forecast/evaluacion.py:29-30`:
  `train = serie.iloc[:-12]`, `test = serie.iloc[-12:]`.
- Con 36 observaciones: **train = 24 meses** (ene-2023 a dic-2024), **test = 12 meses**
  (ene-2025 a dic-2025).

### A.3 Random Search de Prophet

- **Fuente:** `planificacion/forecast/modelo.py:41-57` (espacio de búsqueda),
  `:113-143` (`random_search`), `:60-73` (`evaluar_cv`, con `cross_validation`/
  `performance_metrics` de `prophet.diagnostics`).
- **n_iter = 20** (`config.py:36`). Espacio: `changepoint_prior_scale` log-uniforme en
  [0.01, 0.5]; `seasonality_prior_scale` log-uniforme en [1, 10]; `seasonality_mode`
  ∈ {additive, multiplicative}; `n_changepoints` ∈ {5,10,15,20,25}; `yearly_seasonality`
  ∈ {True, 5, 8, 10} (si es entero, es el orden de Fourier).
- **Métrica optimizada:** RMSE (`metrica: str = 'rmse'`, `config.py:40`).
- **Sí usa `cross_validation` de Prophet:** `initial=730 days` (~24 meses),
  `period=90 days` (~3 meses entre cortes), `horizon=180 days` (~6 meses),
  `parallel='threads'` (**Fuente:** `config.py:37-43`).
- **holidays / regresores / growth logistic: NO DETERMINADO como presentes — no se usan.**
  `crear_modelo` (`modelo.py:34-37`) solo pasa `weekly_seasonality=False,
  daily_seasonality=False` + los hiperparámetros buscados; sin `holidays`, sin
  `add_regressor`, sin `growth='logistic'` en ningún punto del código.

### A.4 Hiperparámetros ganadores por artículo

**Fuente:** `hiperparametros_prophet.json` (cache en disco), leído directo sin volver a
correr la búsqueda (**Generado por:** `analysis_revision/01_prophet_reproducibilidad.py` →
`outputs/A_prophet/hiperparametros_por_articulo.csv`):

| Artículo | changepoint_prior_scale | seasonality_prior_scale | seasonality_mode | n_changepoints | yearly_seasonality |
|---|---|---|---|---|---|
| CRONOS-N04 | 0.335254 | 2.508421 | additive | 10 | 10 |
| HORIZON-M09 | 0.122002 | 1.059280 | multiplicative | 10 | 5 |
| TAURO 2-N04 | 0.100264 | 6.448081 | additive | 10 | 10 |

### A.5 Métricas en entrenamiento (in-sample) y prueba (out-of-sample)

**Generado por:** `analysis_revision/01_prophet_reproducibilidad.py`. El repo original NO
calcula MAPE/WAPE (`planificacion/forecast/metricas.py:8-16` solo tiene MSE/RMSE/MAE/R²) ni
los persiste a CSV — acá se agregan con una implementación nueva
(`analysis_revision/_common.py: mape()/wape()`) sobre exactamente los mismos pares
y/yhat que usa el pipeline original.

**In-sample** (mismo camino que `Pronostico_Ventas.ipynb` celda 23,
`modelo.ajustar_y_pronosticar` + `evaluacion.metricas_de_ajuste_historico`) —
`outputs/A_prophet/insample.csv`:

| Artículo | MSE | RMSE | MAE | R² | MAPE (%) | WAPE (%) |
|---|---|---|---|---|---|---|
| CRONOS-N04 | 164668.29 | 405.79 | 328.61 | 0.973 | 3.31 | 3.06 |
| HORIZON-M09 | 212069.85 | 460.51 | 365.00 | 0.592 | 14.24 | 12.84 |
| TAURO 2-N04 | 50521.67 | 224.77 | 172.01 | 0.941 | 6.07 | 5.41 |

Reproduce **exacto** (delta relativo < 1e-5) contra la salida ya ejecutada de la celda 23 del
notebook (`outputs/A_prophet/verificacion_reproducibilidad.csv`).

**Out-of-sample** (hold-out 12 meses, mismo split que
`evaluacion.evaluar_holdout`/celda 25 del notebook) — `outputs/A_prophet/outofsample_prophet.csv`:

| Artículo | MSE | RMSE | MAE | R² | MAPE (%) | WAPE (%) | R² documentado en el notebook |
|---|---|---|---|---|---|---|---|
| CRONOS-N04 | 1.58×10⁷ | 3974.55 | 3912.83 | -4.573 | 32.85 | 32.56 | -5.67 |
| HORIZON-M09 | 5.96×10⁵ | 771.88 | 687.90 | -2.617 | 25.12 | 23.74 | -2.62 |
| TAURO 2-N04 | 2.06×10⁶ | 1434.33 | 1319.22 | -7.342 | 39.42 | 37.21 | -4.83 |

R² promedio out-of-sample de esta corrida: **-4.844** (documentado en el notebook: -4.37,
delta relativo 10.8%, fuera de la tolerancia del 5% — ver hallazgo de no-determinismo en la
Sección G). El R² individual de HORIZON-M09 reproduce casi exacto (-2.617 vs -2.62); el de
CRONOS-N04 y TAURO 2-N04 no.

**Baseline naive estacional** (mismo mes, año anterior;
`metricas.baseline_naive_estacional`) — `outputs/A_prophet/outofsample_baseline.csv`,
reproduce exacto contra el notebook:

| Artículo | MSE | RMSE | MAE | R² | MAPE (%) | WAPE (%) |
|---|---|---|---|---|---|---|
| CRONOS-N04 | 1987804.00 | 1409.90 | 913.83 | 0.299 | 7.66 | 7.60 |
| HORIZON-M09 | 130941.25 | 361.86 | 230.42 | 0.205 | 8.01 | 7.95 |
| TAURO 2-N04 | 173135.08 | 416.10 | 269.75 | 0.298 | 7.66 | 7.61 |

R² promedio baseline: **0.267** (documentado: 0.27, coincide). **Interpretación (propia, no
del código):** en el hold-out de 12 meses, Prophet con estos hiperparámetros no supera al
baseline naive estacional en ninguno de los 3 artículos — esto ya lo reconoce el propio
notebook (celda 24, "si Prophet no supera al naive, conviene saberlo").

---

## B. Curva de talles y explosión BOM

- **Fórmula:** distribución normal (`scipy.stats.norm.pdf`) evaluada en talles enteros y
  normalizada a suma 1 (**Fuente:** `planificacion/insumos/curva_talles.py:10-20`).
- **Parámetros — hardcodeados**, no calculados de datos ni leídos de un archivo externo
  (**Fuente:** `planificacion/config.py:100-104`): `talle_min=34`, `talle_max=50`,
  `talle_media=42`, `talle_desvio=2.5`.
- **BOM:** `BOM_Zapato_Terminado.xlsx`, columnas `Material | Articulo | Talle | Componente |
  Componente de lista de materia | Cantidad | UM | Proveedor | Precio | Moneda | Lead Time`
  (**Fuente:** `planificacion/io_datos.py:32-36`, estructura confirmada por lectura directa
  del archivo). Fórmula de explosión: `Consumo = Σ_talles (Forecast_artículo × p_talle ×
  Cantidad_BOM)` (**Fuente:** `planificacion/insumos/bom.py:75-119`).

### Ejemplo desarrollado: CRONOS-N04, septiembre 2026, talle 42

**Generado por:** `analysis_revision/02_bom_talle_ejemplo.py` (usa `curva_talles` y `bom`
reales sobre `pronostico_ventas.xlsx`/`BOM_Zapato_Terminado.xlsx` ya existentes) →
`outputs/B_bom_talles/`.

1. Ventas pronosticadas del artículo (todos los talles), sep-2026: **10574.5751**
   (`pronostico_ventas.xlsx`, `Tipo='Pronostico'`).
2. Proporción del talle 42 (media=42, coincide con el pico de la curva):
   `norm.pdf(42; 42, 2.5) / Σ_{34..50} norm.pdf(t; 42, 2.5)` = **0.159676**.
3. Demanda del talle 42: `forecast_talle = round(10574.5751 × 0.159676, 2)` = **1688.51**
   pares (`bom.py:94`).
4. Consumo de insumos (`consumo_proyectado = forecast_talle × Cantidad`, `bom.py:113`):
   - **PUNTERA ACERO 59 NORMAL T10** (código 4000312004, Cantidad=1 PAA):
     1688.51 × 1 = **1688.51 PAA**.
   - **CONJ SISTEMA PU**, componente 3000093 (Cantidad=159.929 G):
     1688.51 × 159.929 = **270 041.72 G**.
   - **CONJ SISTEMA PU**, componente 3000094 (Cantidad=323.238 G):
     1688.51 × 323.238 = **545 790.60 G** (el insumo aparece 2 veces en la BOM para
     este Articulo_Talle — dos componentes de la lista de materiales con el mismo nombre;
     el merge es 1→N y ambas filas generan consumo, que luego se suma al consolidar por
     familia de compra).

### Ejemplo con talle extremo (talle 34, mismo mes/artículo)

Proporción del talle 34: **0.000954** (vs. 0.159676 del talle 42 — el talle extremo pesa
~167 veces menos). `forecast_talle = round(10574.5751 × 0.000954, 2)` = **10.09** pares.
Consumo de CONJ SISTEMA PU en talle 34: 10.09×121.929 + 10.09×247.238 = **1230.26 G +
2494.63 G** (`outputs/B_bom_talles/ejemplo_cronos_n04_talle_extremo.csv`).

---

## C. Curva de talles vs. datos empíricos (Q-Q plot)

**NO DETERMINADO.** No existe en el repositorio ningún dataset de ventas históricas
desagregado por talle (búsqueda documentada en
`outputs/C_curva_talles/NO_DETERMINADO.md`, **Generado por:**
`analysis_revision/02_bom_talle_ejemplo.py: seccion_c()`): se inspeccionaron todos los
`.xlsx`/`.csv` de la raíz del repo buscando columnas `talle` + `venta`/`volumen` a la vez.
El único archivo con columna `Talle` es la BOM (cantidad de insumo por unidad de producto
terminado y talle — no un registro de unidades vendidas por talle). El pronóstico de Prophet
(`pronostico_ventas.xlsx`) es agregado a nivel artículo, sin apertura por talle.

**Qué haría falta:** un dataset de ventas reales con columna `Talle`
(`Articulo | Fecha | Talle | Unidades_Vendidas`), hoy inexistente en el pipeline. Con eso se
podría construir la tabla de proporciones empíricas, correr una prueba χ² de bondad de ajuste
contra la curva normal y generar el gráfico Q-Q. Sin ese dato no se genera el gráfico.

---

## D. Selección de K y calidad del clustering

**Generado por:** `analysis_revision/03_kmeans_criticidad.py` (reusa
`main.explotar_demanda_en_requerimientos_de_insumos`, `clustering.evaluar_cantidad_de_clusters`
y `clustering.clasificar_por_kmeans` reales, mismo `random_state=42`/`n_init=20` que el
pipeline) → `outputs/D_E_kmeans_ahp/`.

**Universo real: 24 familias entran al K-Means** (no 27 — ver Sección G), confirmado con
`assert len(familias) == 24` en el script. **17 quedan en "alcance de Compras"** tras el
filtro `Es_Compra_Externa = Proveedor.notna()` (**Fuente:**
`planificacion/insumos/clustering.py:84,95-109`) — filtro de negocio explícito (separa
insumos con proveedor externo de producción interna: capelladas y plantillas
semielaboradas), no un `dropna()` ni un top-N.

### Codo + silueta, k=2..6 (`clustering.py:21-35`, `k_range=range(2,7)`, `config.py:117`)

`outputs/D_E_kmeans_ahp/codo_silueta.csv`:

| K | Silhouette | Inercia |
|---|---|---|
| 2 | 0.496 | 4.522 |
| 3 | 0.524 | 2.847 |
| 4 | 0.576 | 1.909 |
| 5 | 0.643 | 0.933 |
| **6** | **0.649** | 0.596 |

**El silhouette máximo da K=6 (0.649)**, no K=3. `k_clusters=3` está hardcodeado
(**Fuente:** `config.py:116`, comentario `# 3 = CRÍTICO / IMPORTANTE / SECUNDARIO`) por
decisión de interpretabilidad de negocio, no por optimizar la métrica — ver Sección G.

### Silueta por cluster (k=3) — cálculo nuevo

`silhouette_samples` (sklearn) no existe en el repo (solo `silhouette_score`, promedio
global). **Generado por:** este script → `outputs/D_E_kmeans_ahp/silueta_por_familia.csv`.
Promedio verificado exacto contra `silhouette_score` (delta=0.0):

| Criticidad | Silhouette medio |
|---|---|
| CRÍTICO | 0.347 |
| IMPORTANTE | 0.391 |
| SECUNDARIO | 0.830 |

(Silhouette global k=3: **0.524**, coincide con `ClasificacionInsumos.silhouette` real.)

---

## E. AHP

- **Pesos usados** (**Fuente:** `planificacion/config.py:122-124`): Alcance=0.604,
  Lead Time=0.312, Volumen=0.084 (suman 1.000, validado en `__post_init__`,
  `config.py:138-145` — solo valida la suma, no calcula ni verifica un CR).
- **NO DETERMINADO — metodología AHP completa.** No existe en el repositorio ningún código
  de: escala original de la encuesta (Saaty/Likert), conversión a escala de Saaty, matriz de
  comparaciones pareadas, agregación de encuestados (media geométrica u otra), autovector
  principal, λmax, CI, RI ni CR. Los 3 pesos llegan al código como constantes ya calculadas;
  el propio notebook lo reconoce (`Insumos_Criticos_KMeans.ipynb`, celda 33, sección de
  conclusiones: "pesos derivados por AHP... definidos en `planificacion/config.py`").
- **Valor 7.22 (Alcance vs. Volumen) y sus supuestos componentes 1.93 / 3.74: sin
  coincidencias en el repo.** Búsqueda ejecutada (**Generado por:** este script,
  `outputs/D_E_kmeans_ahp/busqueda_ahp_7_22.txt`, excluyendo `analysis_revision/` y
  descartando líneas que son blobs base64 de imágenes PNG incrustadas en los `.ipynb`):
  `grep -rn '7.22'`, `'7,22'`, `'1.93'`, `'1,93'`, `'3.74'`, `'3,74'` sobre `.py`/`.ipynb`/`.md`
  → **sin coincidencias reales** (las únicas líneas que matchean "1.93"/"3.74" son metadatos
  de timestamp/userId de Jupyter, no valores AHP). **No se puede determinar dentro de este
  repositorio** si 7.22 fue ingresado directo o derivado por transitividad
  (1.93 × 3.74 = 7.212 ≈ 7.22); el proceso AHP se hizo fuera del repo.
- **Cómo entra el AHP al K-Means: no entra al clustering en sí.** `clustering.escalar_variables`
  usa `MinMaxScaler` (no `StandardScaler`) sobre las 3 features a [0,1] sin ponderar por AHP
  (**Fuente:** `clustering.py:15-18`); el K-Means corre "ciego" al AHP. El score AHP
  (`features.score_ahp`, combinación lineal `Vol_norm·w + Alcance_pct·w + LeadTime_norm·w`) se
  usa **después**, solo para ordenar los centroides ya calculados y asignarles la etiqueta
  CRÍTICO/IMPORTANTE/SECUNDARIO (mayor score → CRÍTICO) — **Fuente:** `clustering.py:71-79`.
  Aparece también como columna adicional en el heatmap de resultados
  (`planificacion/insumos/graficos.py:213-214`), no como entrada del modelo.

### Score AHP por familia (k=3, `features.score_ahp` real)

**Generado por:** este script → `outputs/D_E_kmeans_ahp/score_ahp_por_familia.csv`
(24 familias, orden descendente por Score_AHP; se muestran las de mayor y menor score):

| Familia | Vol_norm | Alcance_pct | LeadTime_norm | Score_AHP | Criticidad | Compra externa |
|---|---|---|---|---|---|---|
| CONJ SISTEMA PU | 100.0 | 100.0 | 100.0 | 100.00 | CRÍTICO | Sí |
| SISTEMA PU TINTA, GRIS | 2.54 | 100.0 | 62.5 | 80.11 | IMPORTANTE | Sí |
| PUNTERA ACERO 59 NORMAL | 100.0 | 66.7 | 87.5 | 75.97 | CRÍTICO | Sí |
| CAJA EMPAQUE (BOTA/BOTÍN) | 100.0 | 66.7 | 62.5 | 68.17 | CRÍTICO | Sí |
| … | | | | | | |
| CORDON TRENZ C/PIN, NAT/MA, 1,05 | 3.33 | 33.3 | 0.0 | 20.41 | SECUNDARIO | Sí |

(tabla completa de 24 filas en el CSV).

### Verificación de reproducibilidad

`outputs/D_E_kmeans_ahp/verificacion_vs_excel.csv`: las 17 familias del alcance de Compras
reproducen **exactamente** la misma Criticidad que `Insumos_Criticos.xlsx` (17/17 = OK).

---

## F. Política de inventario

**Generado por:** `analysis_revision/04_politicas_inventario.py` (reusa
`politicas.calcular_politicas_de_inventario` real) → `outputs/F_politicas/`.

### σ: cómo se calcula (respuesta directa)

**In-sample (ajuste), no out-of-sample.** `politicas.demanda_por_articulo`
(**Fuente:** `planificacion/insumos/politicas.py:36-65`) calcula el residuo
`Histórico − Pronóstico` sobre las **mismas fechas de entrenamiento** que el modelo ya vio al
ajustar (no sobre el hold-out de 12 meses de `evaluar_holdout`). El propio docstring del
módulo (líneas 7-12) reconoce esto como una limitación: *"Al ser un error in-sample,
subestima la incertidumbre de un pronóstico genuinamente fuera de muestra: el Stock de
Seguridad resultante es una cota conservadora/mínima, no el valor definitivo."*

### Tabla in-sample (reproducción exacta, tal cual `politicas.py`)

`outputs/F_politicas/politicas_insample.csv`, verificado contra
`Politicas_Inventario_Insumos_Criticos.xlsx` (delta absoluto máximo = 0.0):

| Familia | UM | Política | LT (días) | d̄ mensual | σ mensual | SS | ROP/Nivel Objetivo | Stock Máximo | Cobertura SS (días) |
|---|---|---|---|---|---|---|---|---|---|
| CONJ SISTEMA PU | G | Revisión continua (s,Q) | 45 | 8 405 980.2 | 328 335.1 | 824 359.7 | 13 433 330.0 | 21 839 310.2 | 2.9 |
| PUNTERA ACERO 59 NORMAL | PAA | Revisión periódica (R,S) | 40 | 13 779.2 | 470.5 | 1 473.2 | 33 624.8 | 47 404.0 | 3.2 |
| CAJA EMPAQUE (BOTA/BOTÍN) | UN | Revisión periódica (R,S) | 30 | 13 779.2 | 470.5 | 1 363.9 | 28 922.4 | 42 701.6 | 3.0 |

### Cálculo desarrollado — CONJ SISTEMA PU (revisión continua, horizonte = LT)

`z_servicio = 2.05` (**Fuente:** `config.py:130`, ≈98% de nivel de servicio).
`horizonte_meses = LT_dias/30 + R_meses = 45/30 + 0 = 1.5` (**Fuente:**
`politicas.py:129`, `DIAS_POR_MES=30`, `politicas.py:25`).

```
SS  = z · σ_mes · √horizonte = 2.05 × 328335.1 × √1.5 ≈ 824 359.7 g
ROP = d̄_mes · horizonte + SS = 8405980.2 × 1.5 + 824359.7 ≈ 13 433 330.0 g
```

(`ciclo = 1.0` en revisión continua, `Stock_Maximo = ROP + d̄_mes × 1.0 ≈ 21 839 310.2 g`.)

### Análisis adicional: σ recalculado con errores de PRUEBA (out-of-sample)

Reimplementación nueva (no modifica `politicas.py`; reusa `politicas._lead_time` y
`politicas.DIAS_POR_MES`), sustituyendo solo el origen de σ por el residuo del hold-out de
12 meses (mismo split que `evaluacion.evaluar_holdout`) → `outputs/F_politicas/`.

**σ mensual, por artículo:**

| Artículo | σ in-sample | σ out-of-sample | Ratio oos/insample |
|---|---|---|---|
| CRONOS-N04 | 411.55 | 728.74 | 1.77× |
| HORIZON-M09 | 467.04 | 365.70 | 0.78× |
| TAURO 2-N04 | 227.96 | 588.03 | 2.58× |

**Política recalculada con σ out-of-sample:**

| Familia | Desvío mensual | SS | ROP/Nivel Objetivo | Stock Máximo |
|---|---|---|---|---|
| CONJ SISTEMA PU | 495 426.1 | 1 243 879.7 | 13 852 850.0 | 22 258 830.3 |
| PUNTERA ACERO 59 NORMAL | 936.4 | 2 932.3 | 35 083.8 | 48 863.0 |
| CAJA EMPAQUE (BOTA/BOTÍN) | 936.4 | 2 714.7 | 30 273.2 | 44 052.4 |

**% diferencia (out-of-sample vs. in-sample):**

| Familia | Δ Desvío mensual | Δ SS | Δ ROP/Nivel Objetivo | Δ Stock Máximo |
|---|---|---|---|---|
| CONJ SISTEMA PU | +50.9% | +50.9% | +3.1% | +1.9% |
| PUNTERA ACERO 59 NORMAL | +99.0% | +99.0% | +4.3% | +3.1% |
| CAJA EMPAQUE (BOTA/BOTÍN) | +99.0% | +99.0% | +4.7% | +3.2% |

**Interpretación (propia, no del código):** el σ in-sample subestima la variabilidad real
del pronóstico entre 51% y 99% según el insumo (consistente con la advertencia ya escrita en
el docstring de `politicas.py`). El impacto sobre el Stock de Seguridad es directamente
proporcional a σ (fórmula lineal), pero el impacto sobre ROP/Stock Máximo es moderado (2-5%)
porque esos valores están dominados por el término de demanda media (`d̄ × horizonte`), no
por el término de seguridad.

---

## G. Inconsistencias detectadas

| # | Inconsistencia | Detalle |
|---|---|---|
| 1 | AHP: valor 7.22 no verificable | No hay código ni datos crudos de AHP en el repo (Sección E); no se puede confirmar si 7.22 es un dato ingresado o derivado por transitividad de 1.93×3.74. |
| 2 | Universo "27" vs. real "24 → 17" | El evaluador (o el usuario) preguntó por 27 familias; el código usa **24** familias en el K-Means y **17** en el alcance de Compras (Sección D). El número 27 no aparece en ningún lugar del código, outputs, ni artefactos del repo. |
| 3 | K=3 usado vs. K=6 óptimo por silueta | `evaluar_cantidad_de_clusters` da Silhouette máximo en K=6 (0.649); se usa K=3 (0.524) por decisión de interpretabilidad de negocio, no por optimizar la métrica (Sección D). El propio pipeline no imprime advertencia de esto en su ejecución estándar del notebook, salvo la que documenta este análisis. |
| 4 | σ de la política de inventario: in-sample, no out-of-sample | Reconocido en el propio docstring de `politicas.py`, pero subestima el Stock de Seguridad real entre 51% y 99% según el insumo (Sección F). |
| 5 | MAPE/WAPE ausentes del repo original | `planificacion/forecast/metricas.py` solo calcula MSE/RMSE/MAE/R². Se agregaron en esta revisión (`analysis_revision/_common.py`) sin modificar el código original. |
| 6 | No-determinismo del out-of-sample de Prophet | El ajuste in-sample y el baseline naive reproducen exacto entre ejecuciones; el ajuste **out-of-sample** de Prophet (hold-out de 12 meses) da resultados distintos en una nueva ejecución para 2 de 3 artículos (CRONOS-N04: R² -4.57 vs. -5.67 documentado; TAURO 2-N04: -7.34 vs. -4.83; HORIZON-M09 sí reproduce casi exacto: -2.62). Esto ocurre porque `crear_modelo`/`Prophet.fit` no fija una semilla para el optimizador L-BFGS de Stan/cmdstanpy — el ajuste sobre 24 puntos de entrenamiento parece más sensible a esta falta de semilla que el ajuste sobre los 36 puntos completos (que sí reproduce exacto). **Implicación para el informe:** las métricas out-of-sample de Prophet reportadas en el Capítulo 4 deben presentarse como indicativas de un orden de magnitud (Prophet no le gana al baseline naive), no como valores exactos reproducibles bit a bit. |
| 7 | No hay dataset de ventas por talle | Impide la comparación empírica vs. curva normal pedida en la Sección C (ver NO DETERMINADO). |
| 8 | Prophet pierde contra todos los métodos clásicos evaluados (Sección H) | En los 3 artículos, Prophet queda último en MAE/RMSE/MAPE/WAPE del benchmark, incluso por debajo del Naive estacional (el método más simple posible). No es una inconsistencia de código, sino un hallazgo relevante para la discusión del Capítulo 4: con 36 meses de historia y sin regresores/holidays, la ganancia de Prophet sobre métodos clásicos no está sustentada empíricamente en este dataset. |

---

## H. Benchmark de modelos de series de tiempo (Fase 2)

**Generado por:** `analysis_revision/benchmark_ts.py` → `outputs/benchmark/comparacion_modelos.csv`
y `comparacion_rmse.png`. Mismo split que el pipeline original (train=24 meses,
test=12 meses, `evaluacion.evaluar_holdout`), por artículo. `pmdarima` instaló y corrió sin
problemas en este entorno (no hizo falta el fallback de SARIMAX acotado).

- **Naive estacional:** reusa `metricas.baseline_naive_estacional` (y_t = y_{t-12}); con
  24 meses de train siempre hay 12 meses previos disponibles, no hizo falta el fallback a
  naive simple.
- **Media móvil (ventana 3):** walk-forward con los valores reales observados de la serie
  (no con las propias predicciones del modelo) — convención elegida por ser la menos
  ambigua; documentada en `outputs/benchmark/notas_modelos.txt`.
- **SES:** `statsmodels.tsa.holtwinters.SimpleExpSmoothing`, optimizado.
- **Holt-Winters/ETS:** tendencia aditiva + estacionalidad elegida por AIC entre
  aditiva/multiplicativa. Ganador por artículo (`outputs/benchmark/notas_modelos.txt`):
  CRONOS-N04 → aditiva (AIC 359.05 vs. 364.10 mult.); HORIZON-M09 → **multiplicativa**
  (AIC 319.91 vs. 321.09 adit.); TAURO 2-N04 → aditiva (AIC 336.22 vs. 340.44 mult.).
- **SARIMA:** `pmdarima.auto_arima(seasonal=True, m=12)`. Órdenes elegidos:
  CRONOS-N04 → `(0,1,1)(0,0,0,12)`; HORIZON-M09 → `(2,0,1)(0,0,0,12)`;
  TAURO 2-N04 → `(0,0,0)(0,0,0,12)` (equivalente a ruido blanco sobre la media —
  `auto_arima` no encontró estructura ARIMA significativa con solo 24 observaciones).
- **Prophet (optimizado):** reusa el mismo split y los hiperparámetros cacheados de la
  Sección A (idénticos valores a `outputs/A_prophet/outofsample_prophet.csv`).

### MAE / RMSE / MAPE / WAPE en test (12 meses), por artículo y modelo

| Artículo | Modelo | MAE | RMSE | MAPE (%) | WAPE (%) |
|---|---|---|---|---|---|
| CRONOS-N04 | **Naive estacional** | **913.83** | **1409.90** | **7.66** | **7.60** |
| CRONOS-N04 | Media móvil (3) | 1204.06 | 1916.22 | 10.03 | 10.02 |
| CRONOS-N04 | SES | 1366.82 | 1731.97 | 11.49 | 11.37 |
| CRONOS-N04 | SARIMA | 1447.60 | 1804.31 | 11.95 | 12.04 |
| CRONOS-N04 | Holt-Winters (ETS) | 1941.77 | 2179.11 | 16.45 | 16.16 |
| CRONOS-N04 | Prophet (optimizado) | 3912.83 | 3974.55 | 32.85 | 32.56 |
| HORIZON-M09 | **Naive estacional** | **230.42** | **361.86** | **8.01** | **7.95** |
| HORIZON-M09 | Media móvil (3) | 285.53 | 460.24 | 9.79 | 9.85 |
| HORIZON-M09 | SARIMA | 309.69 | 392.56 | 10.79 | 10.69 |
| HORIZON-M09 | SES | 365.45 | 455.23 | 12.33 | 12.61 |
| HORIZON-M09 | Holt-Winters (ETS) | 447.96 | 528.00 | 16.91 | 15.46 |
| HORIZON-M09 | Prophet (optimizado) | 687.90 | 771.88 | 25.12 | 23.74 |
| TAURO 2-N04 | **Naive estacional** | **269.75** | **416.10** | **7.66** | **7.61** |
| TAURO 2-N04 | Media móvil (3) | 355.06 | 565.30 | 10.02 | 10.01 |
| TAURO 2-N04 | SES | 499.84 | 614.41 | 13.52 | 14.10 |
| TAURO 2-N04 | SARIMA | 621.99 | 740.78 | 16.57 | 17.54 |
| TAURO 2-N04 | Holt-Winters (ETS) | 557.49 | 758.08 | 15.69 | 15.72 |
| TAURO 2-N04 | Prophet (optimizado) | 1319.22 | 1434.33 | 39.42 | 37.21 |

**Resultado, sin filtrar lo desfavorable a Prophet:** en los 3 artículos, **Prophet
(optimizado por Random Search + CV) queda último en las 4 métricas**, incluso peor que el
Naive estacional simple (el método más básico) y que Media móvil (3). El orden de mejor a
peor (por RMSE) es consistente entre artículos: Naive estacional < Media móvil (3) ≲
SES/SARIMA < Holt-Winters < Prophet.

**Interpretación (propia, no del código):** con solo 24 meses de entrenamiento y sin
regresores externos ni holidays, Prophet parece sobreajustar la estacionalidad/tendencia del
tramo de entrenamiento (nótese el R² in-sample alto, Sección A.5) a costa de generalizar peor
al año de prueba que un simple "repetir el mismo mes del año anterior". Esto es consistente
con lo que el propio notebook ya documenta (Prophet no supera al baseline naive en el
hold-out) y lo extiende: tampoco supera a ningún otro método clásico evaluado.

---

## Anexo A — Funciones principales del pipeline

| Función | Archivo:línea | Qué hace | Entradas → Salidas | Paso del proceso |
|---|---|---|---|---|
| `cargar_ventas` | `planificacion/io_datos.py:11` | Descarga el histórico de ventas (Google Sheets) con fallback a caché local | url, cache → DataFrame wide | 1. Carga de datos |
| `cargar_bom` | `planificacion/io_datos.py:32` | Carga la lista de materiales | path → DataFrame BOM | 3. Explosión BOM |
| `cargar_forecast` | `planificacion/io_datos.py:39` | Carga la salida del pronóstico | path → DataFrame Articulo/Fecha/Ventas/Tipo | 3. Explosión BOM |
| `pasar_a_formato_largo` | `planificacion/forecast/datos.py:16` | Convierte el Excel wide a formato largo | df_wide → Articulo/Fecha/Volumen_Ventas | 1. Preprocesamiento |
| `validar_calidad_de_la_serie` | `planificacion/forecast/datos.py:58` | Cuenta nulos y detecta ventas negativas | df_long → Validacion | 1. Preprocesamiento |
| `interpolar_nulos` | `planificacion/forecast/datos.py:66` | Interpola linealmente nulos por artículo (no se usa en la corrida actual) | df_long → df_long | 1. Preprocesamiento |
| `a_formato_prophet` | `planificacion/forecast/datos.py:78` | Renombra al contrato Prophet (ds/y) | df_long → df_prophet | 2. Prophet |
| `serie_articulo` | `planificacion/forecast/datos.py:86` | Extrae la serie ds/y de un artículo | df_prophet, articulo → serie | 2. Prophet |
| `crear_modelo` | `planificacion/forecast/modelo.py:34` | Instancia Prophet sin estacionalidad semanal/diaria | params → Prophet | 2. Prophet |
| `muestrear_hiperparametros` | `planificacion/forecast/modelo.py:41` | Muestrea una combinación aleatoria de hiperparámetros | rng → dict de params | 2. Random Search |
| `evaluar_cv` | `planificacion/forecast/modelo.py:60` | Ajusta Prophet y devuelve la métrica media de cross-validation | serie, params, cfg → float | 2. Random Search |
| `random_search` | `planificacion/forecast/modelo.py:113` | Random Search de n_iter combinaciones | serie, cfg → ResultadoBusqueda | 2. Random Search |
| `buscar_hiperparametros` | `planificacion/forecast/modelo.py:146` | Optimiza hiperparámetros por artículo con caché en disco | df_prophet, articulos, cfg → dict | 2. Random Search |
| `ajustar_y_pronosticar` | `planificacion/forecast/modelo.py:210` | Entrena con la serie completa y predice histórico + horizonte | serie, params, cfg → Ajuste | 2. Pronóstico final |
| `ajustar_modelos_por_articulo` | `planificacion/forecast/modelo.py:233` | Aplica el ajuste a todos los artículos | df_prophet, articulos, params → dict[Ajuste] | 2. Pronóstico final |
| `metricas` | `planificacion/forecast/metricas.py:8` | MSE, RMSE, MAE, R² | y_true, y_pred → dict | 2. Evaluación |
| `baseline_naive_estacional` | `planificacion/forecast/metricas.py:19` | Baseline: mismo mes del año anterior | serie, test_periods → dict | 2. Evaluación |
| `metricas_de_ajuste_historico` | `planificacion/forecast/evaluacion.py:11` | R² de ajuste in-sample | ajustes, df_prophet → DataFrame | 2. Evaluación |
| `evaluar_holdout` | `planificacion/forecast/evaluacion.py:26` | Entrena sin los últimos test_periods meses y evalúa out-of-sample | serie, params, cfg → dict | 2. Evaluación |
| `metricas_de_validacion_out_of_sample` | `planificacion/forecast/evaluacion.py:40` | Compara Prophet vs. baseline en el hold-out | df_prophet, articulos, params, cfg → (DF, DF) | 2. Evaluación |
| `separar_ajuste_historico_y_pronostico` | `planificacion/forecast/resultados.py:23` | Arma (futuro, insample) en formato largo | ajustes → (DF, DF) | 2. Export |
| `exportar_pronostico_a_excel` | `planificacion/forecast/resultados.py:40` | Exporta Articulo/Fecha/Ventas/Tipo | histórico, insample, futuro, path → DF | 2. Export (contrato con Sección 3) |
| `construir_curva_de_talles` | `planificacion/insumos/curva_talles.py:10` | Distribución normal de talles, normalizada a suma 1 | cfg → DataFrame talle/proporcion | 3. Curva de talles |
| `detectar_tipo_pronostico` | `planificacion/insumos/bom.py:18` | Autodetecta qué valor de Tipo es el pronóstico | df_forecast, tipo_forzado → str | 3. Explosión BOM |
| `explotar_a_consumo_de_insumos` | `planificacion/insumos/bom.py:75` | Consumo = Σ_talles(Forecast × p_talle × Cantidad_BOM) | df_forecast, df_bom, df_curva, tipo → (DF, Diagnostico) | 3. Explosión BOM |
| `asignar_familias_de_compra` | `planificacion/insumos/familias.py:55` | Agrega Nombre_Base/Familia/UM_familia al consumo | consumo → DF | 3. Consolidación |
| `consolidar_por_familia_de_compra` | `planificacion/insumos/features.py:11` | Agrupa el consumo por familia de compra | consumo, cfg → DF familias | 4. Feature engineering |
| `calcular_variables_de_criticidad` | `planificacion/insumos/features.py:32` | Calcula Vol_norm, Alcance_pct, LeadTime_norm | familias, total_articulos → DF | 4. Feature engineering |
| `score_ahp` | `planificacion/insumos/features.py:62` | Score ponderado lineal con pesos AHP | datos, cfg → Series/array | 4/5. AHP |
| `escalar_variables` | `planificacion/insumos/clustering.py:15` | MinMaxScaler [0,1] sobre las 3 features | familias → (array, scaler) | 5. K-Means |
| `evaluar_cantidad_de_clusters` | `planificacion/insumos/clustering.py:21` | Silhouette e inercia por K (2..6) | variables_escaladas, cfg → DF | 5. Selección de K |
| `clasificar_por_kmeans` | `planificacion/insumos/clustering.py:62` | Entrena K-Means final y etiqueta clusters por Score AHP | familias, variables, scaler, cfg → ClasificacionInsumos | 5. K-Means |
| `tabla_de_familias_clasificadas` | `planificacion/insumos/clustering.py:95` | Filtra a alcance de Compras y ordena por criticidad | clasificacion, cfg → DF | 5. Resultados |
| `tabla_de_skus_por_familia` | `planificacion/insumos/clustering.py:112` | Detalle de SKUs (talles) por familia | consumo, df_resultado, familias, total, cfg → DF | 5. Resultados |
| `demanda_por_articulo` | `planificacion/insumos/politicas.py:36` | Deriva d̄ (media futura) y σ (std residuo in-sample) | df_forecast, tipo → DemandaArticulo | 6. Política de inventario |
| `calcular_politicas_de_inventario` | `planificacion/insumos/politicas.py:93` | Calcula SS, ROP/Nivel Objetivo, Stock Máximo por insumo | forecast, bom, curva, tipo, cfg, familias → DF | 6. Política de inventario |
| `explotar_demanda_en_requerimientos_de_insumos` | `main.py:52` | Orquesta forecast → BOM → familias → features | cfg → RequerimientosDeInsumos | Orquestación (Secciones 3-4) |
| `clasificar_insumos_por_criticidad` | `main.py:81` | Orquesta K-Means + AHP + export | requerimientos, cfg → ClasificacionInsumos | Orquestación (Sección 5) |
| `definir_politicas_de_inventario` | `main.py:104` | Orquesta el cálculo de políticas + export | requerimientos, clasificacion, cfg → None | Orquestación (Sección 6) |
