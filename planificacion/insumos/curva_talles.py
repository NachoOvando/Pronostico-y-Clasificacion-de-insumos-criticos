"""Curva de talles: desagrega el forecast por artículo en forecast por talle."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from planificacion.config import InsumosConfig


def construir_curva_de_talles(cfg: InsumosConfig) -> pd.DataFrame:
    """Distribución normal de talles normalizada a suma 1.

    Devuelve `talle | proporcion`. El forecast llega a nivel artículo (sin
    discriminar talle) y esta curva es la que lo abre para poder cruzarlo con la
    lista de materiales, que sí está por talle.
    """
    talles = np.arange(cfg.talle_min, cfg.talle_max + 1)
    proporcion = norm.pdf(talles, loc=cfg.talle_media, scale=cfg.talle_desvio)
    proporcion = proporcion / proporcion.sum()
    return pd.DataFrame({'talle': talles, 'proporcion': proporcion})
