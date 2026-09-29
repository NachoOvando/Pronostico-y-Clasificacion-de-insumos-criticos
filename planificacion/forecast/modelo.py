"""Modelo Prophet: búsqueda de hiperparámetros, ajuste y pronóstico."""

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from planificacion.config import ForecastConfig
from planificacion.forecast.datos import serie_articulo

logger = logging.getLogger(__name__)


def silenciar_logs_prophet() -> None:
    """Apaga los logs de Prophet/cmdstanpy sin ocultar los warnings de pandas."""
    for nombre in ('prophet', 'cmdstanpy'):
        lg = logging.getLogger(nombre)
        lg.setLevel(logging.ERROR)
        lg.propagate = False
        lg.handlers = [logging.NullHandler()]


def crear_modelo(params: dict) -> Prophet:
    """Instancia Prophet con la config base del proyecto (datos mensuales:
    sin estacionalidad semanal ni diaria) y los hiperparámetros dados."""
    return Prophet(weekly_seasonality=False, daily_seasonality=False, **params)


# ── Espacio de búsqueda (Random Search) ───────────────────────────────────────
def muestrear_hiperparametros(rng: random.Random) -> dict:
    """Muestrea una combinación aleatoria de hiperparámetros de Prophet.

    El orden de las llamadas a `rng` es parte del contrato: cambiarlo cambia la
    secuencia muestreada y, con ella, los resultados reproducibles con semilla.
    """
    return {
        # log-uniforme: muestreamos el exponente y elevamos 10**x
        'changepoint_prior_scale': float(10 ** rng.uniform(np.log10(0.01), np.log10(0.5))),
        # El EDA muestra estacionalidad anual clara: acotamos el prior a [1, 10] para que la
        # búsqueda no la "apague" (un prior ~0.01 ignora la estacionalidad y empeora el ajuste).
        'seasonality_prior_scale': float(10 ** rng.uniform(np.log10(1.0), np.log10(10.0))),
        'seasonality_mode': rng.choice(['additive', 'multiplicative']),
        'n_changepoints': rng.choice([5, 10, 15, 20, 25]),
        # Órdenes de Fourier moderados/altos para capturar bien el patrón mensual anual.
        'yearly_seasonality': rng.choice([True, 5, 8, 10]),
    }


def evaluar_cv(serie: pd.DataFrame, params: dict, cfg: ForecastConfig) -> float:
    """Entrena Prophet con `params` y devuelve la métrica promedio por cross-validation."""
    modelo = crear_modelo(params)
    modelo.fit(serie)
    df_cv = cross_validation(
        modelo,
        initial=cfg.cv_initial,
        period=cfg.cv_period,
        horizon=cfg.cv_horizon,
        parallel=cfg.cv_parallel,
        disable_tqdm=True,
    )
    df_perf = performance_metrics(df_cv, metrics=[cfg.metrica])
    return float(df_perf[cfg.metrica].mean())


# ── Búsqueda aleatoria ────────────────────────────────────────────────────────
@dataclass
class ResultadoBusqueda:
    """Mejor combinación encontrada para un artículo."""

    articulo: str
    params: dict
    score: float
    desde_cache: bool = False


def _clave_cache(articulo: str, serie: pd.DataFrame, cfg: ForecastConfig) -> str:
    """Huella de (serie, semilla, grilla de búsqueda): si cambia algo, se re-busca."""
    payload = {
        'articulo': articulo,
        'serie': [[d.strftime('%Y-%m-%d'), float(v)]
                  for d, v in zip(serie['ds'], serie['y'])],
        'n_iter': cfg.n_iter,
        'seed': cfg.random_seed,
        'cv': [cfg.cv_initial, cfg.cv_period, cfg.cv_horizon],
        'metrica': cfg.metrica,
    }
    bruto = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(bruto.encode('utf-8')).hexdigest()[:16]


def _leer_cache(path: Path) -> dict:
    if not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Cache de hiperparámetros ilegible (%s); se rehace la búsqueda.", exc)
        return {}


def random_search(serie: pd.DataFrame, cfg: ForecastConfig,
                  articulo: str = '') -> ResultadoBusqueda:
    """Random Search de hiperparámetros de Prophet sobre una serie.

    Muestrea `cfg.n_iter` combinaciones y devuelve la de mejor métrica de
    cross-validation. Las combinaciones que no convergen se descartan con un warning.
    """
    rng = random.Random(cfg.random_seed)
    mejor_params: dict | None = None
    mejor_score = np.inf

    for _ in range(cfg.n_iter):
        params = muestrear_hiperparametros(rng)
        try:
            score = evaluar_cv(serie, params, cfg)
        except Exception as exc:  # noqa: BLE001 - combinación inválida o sin convergencia
            logger.warning("[%s] combinación descartada (%s): %s", articulo, exc, params)
            continue
        if score < mejor_score:
            mejor_score, mejor_params = score, params

    if mejor_params is None:
        raise RuntimeError(
            f"[{articulo}] ninguna de las {cfg.n_iter} combinaciones de "
            f"hiperparámetros convergió. Revisar la serie de entrada."
        )
    return ResultadoBusqueda(articulo, mejor_params, mejor_score)


def buscar_hiperparametros(df_prophet: pd.DataFrame, articulos: list[str],
                           cfg: ForecastConfig,
                           usar_cache: bool = True) -> dict[str, ResultadoBusqueda]:
    """Optimiza los hiperparámetros por separado para cada artículo.

    Las series tienen escalas y dinámicas distintas (ver EDA), así que un único
    set global sería subóptimo. El resultado se cachea en disco: re-correr el
    notebook sin cambiar datos ni semilla no repite los ~300 ajustes de Prophet.
    """
    cache = _leer_cache(cfg.cache_hiperparametros) if usar_cache else {}
    resultados: dict[str, ResultadoBusqueda] = {}
    hubo_busqueda = False

    for articulo in articulos:
        serie = serie_articulo(df_prophet, articulo)
        clave = _clave_cache(articulo, serie, cfg)
        guardado = cache.get(clave)
        if guardado is not None:
            resultados[articulo] = ResultadoBusqueda(
                articulo, guardado['params'], guardado['score'], desde_cache=True)
            continue

        resultado = random_search(serie, cfg, articulo=articulo)
        resultados[articulo] = resultado
        cache[clave] = {'articulo': articulo,
                        'params': resultado.params,
                        'score': resultado.score,
                        'metrica': cfg.metrica}
        hubo_busqueda = True

    if usar_cache and hubo_busqueda:
        Path(cfg.cache_hiperparametros).write_text(
            json.dumps(cache, indent=2, ensure_ascii=False), encoding='utf-8')

    return resultados


def tabla_hiperparametros(resultados: dict[str, ResultadoBusqueda],
                          metrica: str) -> pd.DataFrame:
    """Resumen de la búsqueda: una fila por artículo con sus mejores parámetros."""
    return pd.DataFrame([
        {'Articulo': articulo, **r.params, f'{metrica}_cv': round(r.score, 2)}
        for articulo, r in resultados.items()
    ])


# ── Ajuste final y pronóstico ─────────────────────────────────────────────────
@dataclass
class Ajuste:
    """Modelo entrenado sobre la serie completa, con su predicción ya partida.

    `insample` y `futuro` traen el `yhat` **sin recortar**: el recorte a 0 es una
    decisión de presentación (las ventas no pueden ser negativas) y se aplica en
    el export, no en las métricas de ajuste.
    """

    articulo: str
    modelo: Prophet
    insample: pd.DataFrame   # ds | yhat, sobre las fechas observadas
    futuro: pd.DataFrame     # ds | yhat, posteriores a `ultima_fecha`


def ajustar_y_pronosticar(serie: pd.DataFrame, params: dict, cfg: ForecastConfig,
                          ultima_fecha: pd.Timestamp,
                          articulo: str = '') -> Ajuste:
    """Entrena con la serie completa y predice histórico + horizonte en una pasada."""
    modelo = crear_modelo(params)
    modelo.fit(serie)

    future = modelo.make_future_dataframe(periods=cfg.forecast_horizon, freq='MS')
    pred = modelo.predict(future)[['ds', 'yhat']]

    return Ajuste(
        articulo=articulo,
        modelo=modelo,
        insample=pred[pred['ds'] <= ultima_fecha].reset_index(drop=True),
        futuro=pred[pred['ds'] > ultima_fecha].reset_index(drop=True),
    )


def ajustar_modelos_por_articulo(df_prophet: pd.DataFrame, articulos: list[str],
                  params_por_articulo: dict[str, dict],
                  cfg: ForecastConfig) -> dict[str, Ajuste]:
    """Aplica `ajustar_y_pronosticar` a cada artículo con sus mejores parámetros."""
    ultima_fecha = df_prophet['ds'].max()
    return {
        articulo: ajustar_y_pronosticar(
            serie_articulo(df_prophet, articulo), params_por_articulo[articulo],
            cfg, ultima_fecha, articulo=articulo)
        for articulo in articulos
    }
