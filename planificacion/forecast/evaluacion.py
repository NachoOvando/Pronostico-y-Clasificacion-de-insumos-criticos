"""Evaluación del modelo: ajuste in-sample y validación out-of-sample."""

import pandas as pd

from planificacion.config import ForecastConfig
from planificacion.forecast.datos import serie_articulo
from planificacion.forecast.metricas import metricas
from planificacion.forecast.modelo import Ajuste, crear_modelo


def metricas_de_ajuste_historico(ajustes: dict[str, Ajuste],
                                 df_prophet: pd.DataFrame) -> pd.DataFrame:
    """R² de ajuste (in-sample): cuánta varianza del histórico explica el modelo.

    Se compara contra el `yhat` sin recortar a 0, que es el valor que el modelo
    realmente predice.
    """
    filas = []
    for articulo, ajuste in ajustes.items():
        serie = serie_articulo(df_prophet, articulo)
        filas.append({'Articulo': articulo,
                      **metricas(serie['y'].values, ajuste.insample['yhat'].values)})
    return pd.DataFrame(filas)


def predicciones_holdout(serie: pd.DataFrame, params: dict,
                         cfg: ForecastConfig) -> pd.DataFrame:
    """Entrena con todo menos los últimos `test_periods` meses y predice esos meses.

    Devuelve `ds | yhat | y` (pronóstico y valor real de cada mes del hold-out).
    """
    train = serie.iloc[:-cfg.test_periods]
    test = serie.iloc[-cfg.test_periods:]

    modelo = crear_modelo(params)
    modelo.fit(train)
    future = modelo.make_future_dataframe(periods=cfg.test_periods, freq='MS')
    return modelo.predict(future)[['ds', 'yhat']].merge(test, on='ds', how='inner')


def evaluar_holdout(serie: pd.DataFrame, params: dict,
                    cfg: ForecastConfig) -> dict[str, float]:
    """Métricas de error del pronóstico sobre los últimos `test_periods` meses."""
    pred = predicciones_holdout(serie, params, cfg)
    return metricas(pred['y'].values, pred['yhat'].values)


def metricas_de_validacion_out_of_sample(df_prophet: pd.DataFrame, articulos: list[str],
                         params_por_articulo: dict[str, dict],
                         cfg: ForecastConfig) -> pd.DataFrame:
    """Métricas de Prophet por artículo sobre el hold-out (out-of-sample)."""
    return pd.DataFrame([
        {'Articulo': articulo,
         **evaluar_holdout(serie_articulo(df_prophet, articulo),
                           params_por_articulo[articulo], cfg)}
        for articulo in articulos
    ])
