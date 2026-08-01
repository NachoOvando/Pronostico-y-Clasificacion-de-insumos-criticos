"""Entrada/salida de archivos Excel del pipeline."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def cargar_ventas(url: str, cache: Path | None = None) -> pd.DataFrame:
    """Carga el histórico de ventas en formato *wide* (`Articulo` + una columna por mes).

    Lee desde `url` (Google Drive) y guarda una copia local en `cache`. Si la
    descarga falla y existe la copia, se usa esa: el pipeline queda reproducible
    sin conexión.
    """
    try:
        df = pd.read_excel(url)
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de red o de formato
        if cache is None or not Path(cache).exists():
            raise
        logger.warning("No se pudo leer %s (%s). Se usa la copia local %s.",
                       url, exc, cache)
        return pd.read_excel(cache)

    if cache is not None:
        df.to_excel(cache, index=False)
    return df


def cargar_bom(path: Path) -> pd.DataFrame:
    """Carga la lista de materiales y normaliza los nombres de columna."""
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    return df


def cargar_forecast(path: Path) -> pd.DataFrame:
    """Carga la salida del notebook de pronóstico (`Articulo | Fecha | Ventas | Tipo`)."""
    df = pd.read_excel(path)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    return df
