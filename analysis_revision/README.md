# analysis_revision/

Extracción de evidencia para ampliar el Capítulo 4 del informe, a pedido de un evaluador.
Rama `revision-cap4`. No modifica nada de `planificacion/`, los notebooks, ni los
`.xlsx`/`.json`/`.png` ya versionados en la raíz del repo — todo el código y las salidas
nuevas viven acá.

**Informe principal: [`REVISION_CAP4.md`](REVISION_CAP4.md)** (secciones A–H, con el origen
de cada dato y los puntos `NO DETERMINADO`).

## Cómo correr todo, en orden

```bash
pip install -r ../requirements.txt        # pandas, numpy, scipy, sklearn, prophet, etc.
                                            # (prophet compila cmdstan, puede tardar minutos)
python 00_verificar_entorno.py             # gate: aborta si pytest ../tests/ falla
python 01_prophet_reproducibilidad.py      # Sección A
python 02_bom_talle_ejemplo.py             # Secciones B y C
python 03_kmeans_criticidad.py             # Secciones D y E
python 04_politicas_inventario.py          # Sección F

pip install statsmodels pmdarima           # solo para la Fase 2 (benchmark)
python benchmark_ts.py                     # Sección H
```

Cada script es ejecutable de punta a punta por sí solo (recalcula lo que necesita, no
depende de CSVs intermedios de otro script) y es determinístico con las semillas del repo
original (`random_state=42`, hiperparámetros cacheados en `../hiperparametros_prophet.json`),
**salvo el ajuste out-of-sample de Prophet**, que no es bit-a-bit reproducible entre
ejecuciones — ver Sección G, ítem 6 de `REVISION_CAP4.md`.

## Reglas de seguridad (por qué no se llama a ciertas funciones del pipeline)

- Nunca se llama `planificacion.io_datos.cargar_ventas` (puede sobreescribir
  `ventas_historicas_cache.xlsx` si la descarga a Google Sheets tiene éxito) — se lee el
  cache local directo.
- Nunca se llama `modelo.buscar_hiperparametros` (puede reescribir
  `hiperparametros_prophet.json`) — se lee el JSON directo.
- Nunca se llama `graficos.*`, `reporte_excel.*`, ni ningún `.to_excel(cfg.*_output_path)`
  (escriben los artefactos ya versionados en la raíz del repo) — solo se usan las
  funciones de cálculo puras de `planificacion/`, y toda salida propia se escribe en
  `outputs/` con rutas explícitas.

## Estructura

```
_common.py                    helpers compartidos (MAPE/WAPE, lectura de cache/JSON)
00_verificar_entorno.py       gate de reproducibilidad (deps + pytest ../tests/)
01_prophet_reproducibilidad.py   Sección A
02_bom_talle_ejemplo.py          Secciones B y C
03_kmeans_criticidad.py          Secciones D y E
04_politicas_inventario.py       Sección F
benchmark_ts.py                  Sección H (Fase 2)
outputs/{A_prophet,B_bom_talles,C_curva_talles,D_E_kmeans_ahp,F_politicas,benchmark}/
                               CSVs, PNG y notas que respaldan cada sección del informe
REVISION_CAP4.md              informe final
```
