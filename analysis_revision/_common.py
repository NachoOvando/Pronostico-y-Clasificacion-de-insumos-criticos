"""Helpers compartidos por los scripts de analysis_revision/.

Todo lo que hay acá es código NUEVO (no existe en `planificacion/`), escrito
para esta revisión del Capítulo 4. No modifica ni reemplaza nada del pipeline
original: los scripts que lo usan importan las funciones reales de
`planificacion/` para el cálculo, y usan estos helpers solo para no repetir
lectura de archivos / fórmulas de MAPE-WAPE en cada script.

Reglas de seguridad (ver REVISION_CAP4.md, sección 0):
- Nunca se llama a `planificacion.io_datos.cargar_ventas` (puede sobreescribir
  `ventas_historicas_cache.xlsx` si la descarga a Google Sheets tiene éxito).
  Acá se lee el cache local directo.
- Nunca se llama a `planificacion.forecast.modelo.buscar_hiperparametros`
  (puede reescribir `hiperparametros_prophet.json`). Acá se lee el JSON
  directo con `json.loads`.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from planificacion.config import ForecastConfig
from planificacion.forecast.metricas import metricas as metricas_base

DIR_ANALISIS = Path(__file__).resolve().parent
DIR_REPO = DIR_ANALISIS.parent


def cargar_ventas_cache(cfg: ForecastConfig) -> pd.DataFrame:
    """Lee `ventas_historicas_cache.xlsx` directo, sin pasar por la URL de Google Sheets.

    Equivalente de solo-lectura a `planificacion.io_datos.cargar_ventas(cfg.excel_url,
    cfg.cache_ventas)` en el caso (real, en este entorno) en que la descarga por red
    falla y se usa el cache: evita depender de la red y evita el side-effect de esa
    función (sobreescribir el cache si la descarga sí tuviera éxito).
    """
    return pd.read_excel(cfg.cache_ventas)


def cargar_params_cacheados(cfg: ForecastConfig) -> dict[str, dict]:
    """Lee `hiperparametros_prophet.json` y arma {articulo: params}.

    No recalcula la clave SHA-256 de `_clave_cache` (planificacion/forecast/modelo.py:88-100):
    alcanza con indexar por el campo `articulo` que ya guarda cada entrada del cache.
    """
    data = json.loads(Path(cfg.cache_hiperparametros).read_text(encoding='utf-8'))
    return {entry['articulo']: entry['params'] for entry in data.values()}


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error, en %.

    No existe en `planificacion/forecast/metricas.py` (implementación nueva para esta
    revisión). Los meses con y_true == 0 se excluyen del promedio (no hay "% de error"
    bien definido sobre una base 0); si eso pasa se documenta en el CSV de salida.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float('nan')
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def wape(y_true, y_pred) -> float:
    """Weighted Absolute Percentage Error, en %: sum(|error|) / sum(|y_true|).

    No existe en `planificacion/forecast/metricas.py` (implementación nueva). A
    diferencia de MAPE, no tiene problema de división por cero salvo que toda la
    serie sea 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float('nan')
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def metricas_extendidas(y_true, y_pred) -> dict[str, float]:
    """MSE/RMSE/MAE/R2 (planificacion.forecast.metricas.metricas, sin modificar)
    + MAPE/WAPE (nuevos, ver arriba). Punto único de verdad para que todos los
    scripts de analysis_revision/ calculen MAPE/WAPE de la misma forma."""
    out = dict(metricas_base(y_true, y_pred))
    out['MAPE'] = mape(y_true, y_pred)
    out['WAPE'] = wape(y_true, y_pred)
    return out


def cita(archivo: str, lineas: str = '') -> str:
    """Formatea una referencia de origen tipo 'planificacion/insumos/clustering.py:71-79'."""
    return f"{archivo}:{lineas}" if lineas else archivo
