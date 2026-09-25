"""Fase 2 — benchmark de Prophet contra métodos clásicos de series de tiempo.

Mismo split train/test que el pipeline original (`evaluacion.evaluar_holdout`:
train = serie.iloc[:-12], test = serie.iloc[-12:], `planificacion/config.py:49`),
mismo horizonte (12 meses), por artículo. No descarta resultados desfavorables
a Prophet (requisito explícito del usuario).

Modelos:
  - Naive estacional: reusa `metricas.baseline_naive_estacional` tal cual.
  - Media móvil (ventana 3): walk-forward con los valores REALES observados de
    la serie completa (no con las propias predicciones del modelo) — es decir,
    para cada mes de test se promedian los 3 meses anteriores tal como
    ocurrieron. Convención elegida por ser la más simple y menos ambigua de
    documentar; se dice explícitamente acá porque el enunciado no lo precisa.
  - SES (suavizado exponencial simple): `statsmodels.tsa.holtwinters.
    SimpleExpSmoothing`, optimizado.
  - Holt-Winters / ETS: aditivo y multiplicativo, se elige por AIC más bajo.
  - SARIMA: `pmdarima.auto_arima` si está disponible; si no (no instala o
    falla el import), fallback a una búsqueda acotada de SARIMAX por AIC.
  - Prophet ya optimizado: reusa los hiperparámetros cacheados
    (`hiperparametros_prophet.json`), mismo split, cero reimplementación del
    ajuste de Prophet en sí.
"""

import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import DIR_ANALISIS, cargar_params_cacheados, cargar_ventas_cache, metricas_extendidas
from planificacion.config import ForecastConfig
from planificacion.forecast import datos, modelo
from planificacion.forecast.metricas import baseline_naive_estacional

warnings.filterwarnings('ignore')

OUT = DIR_ANALISIS / 'outputs' / 'benchmark'
OUT.mkdir(parents=True, exist_ok=True)

try:
    import pmdarima as pm
    TIENE_PMDARIMA = True
except ImportError:
    TIENE_PMDARIMA = False

from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def modelo_naive_estacional(train_y, test_y):
    """y_t = y_{t-12}. Requiere >=12 meses previos al test; con 24 en train, siempre alcanza."""
    if len(train_y) < 12:
        # Fallback documentado por el enunciado: naive simple (último valor observado).
        pred = np.full(len(test_y), train_y[-1])
        return pred, 'naive_simple (fallback: <12 meses previos)'
    pred = train_y[-12:][:len(test_y)]
    return pred, 'naive_estacional (y_t-12)'


def modelo_media_movil_3(serie_completa_y, idx_inicio_test, n_test):
    """Media móvil ventana 3, walk-forward con valores reales observados (no con
    las propias predicciones): pred[i] = mean(y[idx_inicio_test+i-3 : idx_inicio_test+i])."""
    pred = []
    for i in range(n_test):
        ventana = serie_completa_y[idx_inicio_test + i - 3: idx_inicio_test + i]
        pred.append(np.mean(ventana))
    return np.array(pred)


def modelo_ses(train_y, n_test):
    m = SimpleExpSmoothing(train_y, initialization_method='estimated').fit(optimized=True)
    return m.forecast(n_test)


def modelo_holt_winters(train_y, n_test, periodos_estacionales=12):
    """Aditivo vs. multiplicativo, se elige por AIC. Con 24 obs (=2 ciclos) es el mínimo
    para ajustar una estacionalidad de período 12; puede fallar/no converger bien."""
    candidatos = {}
    for seasonal in ('add', 'mul'):
        try:
            m = ExponentialSmoothing(
                train_y, trend='add', seasonal=seasonal,
                seasonal_periods=periodos_estacionales,
                initialization_method='estimated',
            ).fit(optimized=True)
            candidatos[seasonal] = m
        except Exception as exc:  # noqa: BLE001
            print(f"    Holt-Winters {seasonal}: no ajustó ({exc})")
    if not candidatos:
        raise RuntimeError("Ningún Holt-Winters (aditivo/multiplicativo) pudo ajustarse.")
    mejor_tipo = min(candidatos, key=lambda k: candidatos[k].aic)
    mejor = candidatos[mejor_tipo]
    aics = {k: round(v.aic, 2) for k, v in candidatos.items()}
    return mejor.forecast(n_test), f"ETS seasonal={mejor_tipo} (AIC: {aics})"


def modelo_sarima(train_y, n_test):
    if TIENE_PMDARIMA:
        try:
            m = pm.auto_arima(train_y, seasonal=True, m=12, suppress_warnings=True,
                              error_action='ignore', stepwise=True)
            pred = m.predict(n_periods=n_test)
            return np.asarray(pred), f"auto_arima order={m.order} seasonal_order={m.seasonal_order}"
        except Exception as exc:  # noqa: BLE001
            print(f"    pmdarima.auto_arima falló ({exc}), usando fallback SARIMAX acotado.")

    # Fallback: búsqueda acotada de SARIMAX por AIC.
    mejor_aic, mejor_orden, mejor_modelo = np.inf, None, None
    ordenes_pdq = list(itertools.product([0, 1], [0, 1], [0, 1]))
    ordenes_PDQ = list(itertools.product([0, 1], [0, 1], [0, 1]))
    for p, d, q in ordenes_pdq:
        for P, D, Q in ordenes_PDQ:
            try:
                m = SARIMAX(train_y, order=(p, d, q), seasonal_order=(P, D, Q, 12),
                           enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                if m.aic < mejor_aic:
                    mejor_aic, mejor_orden, mejor_modelo = m.aic, (p, d, q, P, D, Q), m
            except Exception:  # noqa: BLE001
                continue
    if mejor_modelo is None:
        raise RuntimeError("Ninguna combinación SARIMAX de la búsqueda acotada convergió.")
    pred = mejor_modelo.forecast(n_test)
    p, d, q, P, D, Q = mejor_orden
    return np.asarray(pred), f"SARIMAX fallback order=({p},{d},{q}) seasonal=({P},{D},{Q},12) AIC={mejor_aic:.1f}"


def modelo_prophet_optimizado(serie, params, cfg):
    train = serie.iloc[:-cfg.test_periods]
    test = serie.iloc[-cfg.test_periods:]
    m = modelo.crear_modelo(params)
    m.fit(train)
    future = m.make_future_dataframe(periods=cfg.test_periods, freq='MS')
    pred = m.predict(future)[['ds', 'yhat']].merge(test, on='ds', how='inner')
    return pred['y'].values, pred['yhat'].values


def main() -> None:
    modelo.silenciar_logs_prophet()
    cfg = ForecastConfig()
    print(f"pmdarima disponible: {TIENE_PMDARIMA}")

    df_wide = cargar_ventas_cache(cfg)
    df_long = datos.pasar_a_formato_largo(df_wide)
    df_prophet = datos.a_formato_prophet(df_long)
    articulos = sorted(df_prophet['Articulo'].unique())
    params_por_articulo = cargar_params_cacheados(cfg)

    filas = []
    notas = []

    for articulo in articulos:
        print(f"\n=== {articulo} ===")
        serie = datos.serie_articulo(df_prophet, articulo)
        y = serie['y'].values
        n_test = cfg.test_periods
        train_y, test_y = y[:-n_test], y[-n_test:]

        # Naive estacional
        pred, nota = modelo_naive_estacional(train_y, test_y)
        filas.append({'Articulo': articulo, 'Modelo': 'Naive estacional',
                      **metricas_extendidas(test_y, pred)})
        notas.append(f"{articulo} | Naive estacional | {nota}")

        # Media móvil ventana 3 (walk-forward con valores reales)
        pred_mm3 = modelo_media_movil_3(y, len(train_y), n_test)
        filas.append({'Articulo': articulo, 'Modelo': 'Media movil (3)',
                      **metricas_extendidas(test_y, pred_mm3)})

        # SES
        try:
            pred_ses = modelo_ses(train_y, n_test)
            filas.append({'Articulo': articulo, 'Modelo': 'SES',
                          **metricas_extendidas(test_y, pred_ses)})
        except Exception as exc:  # noqa: BLE001
            print(f"  SES no se pudo ajustar: {exc}")
            notas.append(f"{articulo} | SES | NO SE PUDO AJUSTAR: {exc}")

        # Holt-Winters / ETS
        try:
            pred_hw, nota_hw = modelo_holt_winters(train_y, n_test)
            filas.append({'Articulo': articulo, 'Modelo': 'Holt-Winters (ETS)',
                          **metricas_extendidas(test_y, pred_hw)})
            notas.append(f"{articulo} | Holt-Winters | {nota_hw}")
        except Exception as exc:  # noqa: BLE001
            print(f"  Holt-Winters no se pudo ajustar: {exc}")
            notas.append(f"{articulo} | Holt-Winters | NO SE PUDO AJUSTAR: {exc}")

        # SARIMA
        try:
            pred_sarima, nota_sarima = modelo_sarima(train_y, n_test)
            filas.append({'Articulo': articulo, 'Modelo': 'SARIMA',
                          **metricas_extendidas(test_y, pred_sarima)})
            notas.append(f"{articulo} | SARIMA | {nota_sarima}")
            print(f"  SARIMA: {nota_sarima}")
        except Exception as exc:  # noqa: BLE001
            print(f"  SARIMA no se pudo ajustar: {exc}")
            notas.append(f"{articulo} | SARIMA | NO SE PUDO AJUSTAR: {exc}")

        # Prophet ya optimizado
        y_true_p, y_pred_p = modelo_prophet_optimizado(serie, params_por_articulo[articulo], cfg)
        filas.append({'Articulo': articulo, 'Modelo': 'Prophet (optimizado)',
                      **metricas_extendidas(y_true_p, y_pred_p)})

    df_resultado = pd.DataFrame(filas)
    df_resultado.to_csv(OUT / 'comparacion_modelos.csv', index=False)
    (OUT / 'notas_modelos.txt').write_text('\n'.join(notas) + '\n', encoding='utf-8')

    print("\n=== Comparación de modelos — MAE/RMSE/MAPE/WAPE en test (12 meses) ===")
    cols = ['Articulo', 'Modelo', 'MAE', 'RMSE', 'MAPE', 'WAPE']
    print(df_resultado[cols].to_string(index=False))

    # Gráfico de barras agrupadas: RMSE por modelo y artículo
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        pivot = df_resultado.pivot(index='Modelo', columns='Articulo', values='RMSE')
        ax = pivot.plot(kind='bar', figsize=(10, 6))
        ax.set_ylabel('RMSE (test, 12 meses)')
        ax.set_title('Benchmark de modelos de pronóstico — RMSE por artículo')
        plt.tight_layout()
        plt.savefig(OUT / 'comparacion_rmse.png', dpi=150)
        print(f"\nGráfico guardado en {OUT / 'comparacion_rmse.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"No se pudo generar el gráfico: {exc}")

    print(f"\nCSV escrito en {OUT / 'comparacion_modelos.csv'}")


if __name__ == '__main__':
    main()
