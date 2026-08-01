"""Métricas de error usadas en todo el notebook de pronóstico."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metricas(y_true, y_pred) -> dict[str, float]:
    """Calcula MSE, RMSE, MAE y R²."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
    }


def baseline_naive_estacional(serie: pd.DataFrame, test_periods: int) -> dict[str, float]:
    """Predice los últimos `test_periods` meses con el mismo mes del año previo.

    Es la referencia mínima honesta contra la que comparar cualquier modelo de
    pronóstico: si Prophet no le gana, conviene saberlo.
    """
    if len(serie) < 2 * test_periods:
        raise ValueError(
            f"El baseline naive estacional necesita al menos {2 * test_periods} "
            f"observaciones ({test_periods} de prueba + {test_periods} del año "
            f"previo); la serie tiene {len(serie)}."
        )
    y = serie['y'].values
    return metricas(y[-test_periods:], y[-2 * test_periods:-test_periods])
