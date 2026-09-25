"""Sección A — Prophet: reproducibilidad, hiperparámetros y métricas train/test.

Reusa las funciones reales de `planificacion.forecast.*` (nunca las reescribe).
No llama a `modelo.buscar_hiperparametros` (leería/podría reescribir el cache):
toma los hiperparámetros ya guardados en `hiperparametros_prophet.json`.

Para cada artículo (CRONOS-N04, HORIZON-M09, TAURO 2-N04):
  - In-sample: usa `modelo.ajustar_y_pronosticar` (mismo camino que usa el
    notebook, `Pronostico_Ventas.ipynb` celda ~21) y cruza el resultado con
    `evaluacion.metricas_de_ajuste_historico` (MSE/RMSE/MAE/R2 oficiales).
    Agrega MAPE/WAPE (no existen en el repo) sobre los mismos pares y/yhat.
  - Out-of-sample: replica el split de `evaluacion.evaluar_holdout`
    (train=serie.iloc[:-12], test=serie.iloc[-12:], mismo `crear_modelo`) para
    poder extraer los pares y_true/y_pred y calcular MAPE/WAPE, y cruza las 4
    métricas base contra una llamada directa a `evaluacion.evaluar_holdout`
    (Prophet no fija semilla en el optimizador de Stan: un segundo fit puede
    dar un resultado *ligeramente* distinto — se reporta el delta, no se
    asume igualdad exacta).
  - Baseline: `metricas.baseline_naive_estacional` + MAPE/WAPE sobre los
    mismos y[-12:] / y[-24:-12] que usa esa función.

Valores ya documentados en el notebook (in-sample R2~0.97/0.59/0.94;
out-of-sample R2 promedio Prophet~-4.37 vs baseline~0.27) se usan solo como
referencia de comparación, con tolerancia relativa del 5% dado el
no-determinismo del optimizador de Stan.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import DIR_ANALISIS, cargar_params_cacheados, cargar_ventas_cache, metricas_extendidas
from planificacion.config import ForecastConfig
from planificacion.forecast import datos, evaluacion, modelo

OUT = DIR_ANALISIS / 'outputs' / 'A_prophet'
OUT.mkdir(parents=True, exist_ok=True)

TOL_REL = 0.05  # tolerancia relativa para comparar contra valores ya documentados

# Valores tal cual figuran en la salida ya ejecutada de Pronostico_Ventas.ipynb
# (celda 23 para in-sample, celda 25 para out-of-sample/baseline; leídos directo
# del .ipynb, no del resumen de un agente). Sirven de referencia independiente
# para la verificación de reproducibilidad.
DOCUMENTADO_INSAMPLE = {
    'CRONOS-N04':  {'MSE': 164668.29, 'RMSE': 405.79, 'MAE': 328.61, 'R2': 0.97},
    'HORIZON-M09': {'MSE': 212069.85, 'RMSE': 460.51, 'MAE': 365.00, 'R2': 0.59},
    'TAURO 2-N04': {'MSE': 50521.67,  'RMSE': 224.77, 'MAE': 172.01, 'R2': 0.94},
}
DOCUMENTADO_OOS_PROPHET = {
    'CRONOS-N04':  {'MSE': 18901034.84, 'RMSE': 4347.53, 'MAE': 4284.16, 'R2': -5.67},
    'HORIZON-M09': {'MSE': 595802.16,   'RMSE': 771.88,  'MAE': 687.90,  'R2': -2.62},
    'TAURO 2-N04': {'MSE': 1437493.80,  'RMSE': 1198.96, 'MAE': 1078.38, 'R2': -4.83},
}
DOCUMENTADO_OOS_BASELINE = {
    'CRONOS-N04':  {'MSE': 1987804.00, 'RMSE': 1409.90, 'MAE': 913.83, 'R2': 0.30},
    'HORIZON-M09': {'MSE': 130941.25,  'RMSE': 361.86,  'MAE': 230.42, 'R2': 0.21},
    'TAURO 2-N04': {'MSE': 173135.08,  'RMSE': 416.10,  'MAE': 269.75, 'R2': 0.30},
}


def _delta_rel(a: float, b: float) -> float:
    if b == 0:
        return float('nan')
    return abs(a - b) / abs(b)


def main() -> None:
    modelo.silenciar_logs_prophet()
    cfg = ForecastConfig()

    df_wide = cargar_ventas_cache(cfg)
    df_long = datos.pasar_a_formato_largo(df_wide)
    df_prophet = datos.a_formato_prophet(df_long)
    articulos = sorted(df_prophet['Articulo'].unique())
    params_por_articulo = cargar_params_cacheados(cfg)

    ultima_fecha = df_prophet['ds'].max()

    filas_insample, filas_oos_prophet, filas_oos_baseline, filas_verif = [], [], [], []

    for articulo in articulos:
        serie = datos.serie_articulo(df_prophet, articulo)
        params = params_por_articulo[articulo]

        # ── In-sample: mismo camino que el notebook (modelo.ajustar_y_pronosticar) ──
        ajuste = modelo.ajustar_y_pronosticar(serie, params, cfg, ultima_fecha, articulo)
        ajustes = {articulo: ajuste}
        oficial_insample = evaluacion.metricas_de_ajuste_historico(ajustes, df_prophet)
        oficial_insample = oficial_insample.set_index('Articulo').loc[articulo].to_dict()

        propio_insample = metricas_extendidas(serie['y'].values, ajuste.insample['yhat'].values)
        doc_in = DOCUMENTADO_INSAMPLE[articulo]
        for met in ('MSE', 'RMSE', 'MAE', 'R2'):
            filas_verif.append({
                'Seccion': 'A.insample', 'Articulo': articulo, 'Metrica': met,
                'valor_propio': propio_insample[met], 'valor_documentado_notebook': doc_in[met],
                'delta_relativo': _delta_rel(propio_insample[met], doc_in[met]),
                'coincide_con_documentado': _delta_rel(propio_insample[met], doc_in[met]) < TOL_REL,
            })
        filas_insample.append({'Articulo': articulo, **propio_insample,
                               'MSE_oficial_evaluacion.py': oficial_insample['MSE'],
                               'R2_oficial_evaluacion.py': oficial_insample['R2']})

        # ── Out-of-sample: replica evaluacion.evaluar_holdout para tener y_true/y_pred ──
        train = serie.iloc[:-cfg.test_periods]
        test = serie.iloc[-cfg.test_periods:]
        m_oos = modelo.crear_modelo(params)
        m_oos.fit(train)
        future = m_oos.make_future_dataframe(periods=cfg.test_periods, freq='MS')
        pred = m_oos.predict(future)[['ds', 'yhat']].merge(test, on='ds', how='inner')
        propio_oos = metricas_extendidas(pred['y'].values, pred['yhat'].values)

        oficial_oos = evaluacion.evaluar_holdout(serie, params, cfg)
        doc_oos = DOCUMENTADO_OOS_PROPHET[articulo]
        for met in ('MSE', 'RMSE', 'MAE', 'R2'):
            delta = _delta_rel(propio_oos[met], doc_oos[met])
            filas_verif.append({
                'Seccion': 'A.outofsample', 'Articulo': articulo, 'Metrica': met,
                'valor_propio': propio_oos[met], 'valor_documentado_notebook': doc_oos[met],
                'delta_relativo': delta, 'coincide_con_documentado': delta < TOL_REL,
            })
        filas_oos_prophet.append({'Articulo': articulo, **propio_oos,
                                  'R2_oficial_evaluar_holdout_mismo_run': oficial_oos['R2'],
                                  'R2_documentado_notebook': doc_oos['R2']})

        # ── Baseline naive estacional ──
        y = serie['y'].values
        baseline_extendido = metricas_extendidas(y[-cfg.test_periods:],
                                                 y[-2 * cfg.test_periods:-cfg.test_periods])
        doc_base = DOCUMENTADO_OOS_BASELINE[articulo]
        for met in ('MSE', 'RMSE', 'MAE', 'R2'):
            delta = _delta_rel(baseline_extendido[met], doc_base[met])
            filas_verif.append({
                'Seccion': 'A.baseline', 'Articulo': articulo, 'Metrica': met,
                'valor_propio': baseline_extendido[met], 'valor_documentado_notebook': doc_base[met],
                'delta_relativo': delta, 'coincide_con_documentado': delta < TOL_REL,
            })
        filas_oos_baseline.append({'Articulo': articulo, **baseline_extendido})

    df_insample = pd.DataFrame(filas_insample)
    df_oos_prophet = pd.DataFrame(filas_oos_prophet)
    df_oos_baseline = pd.DataFrame(filas_oos_baseline)
    df_verif = pd.DataFrame(filas_verif)

    # Promedios para comparar contra lo documentado en el notebook (celda 25)
    r2_prophet_prom = df_oos_prophet['R2'].mean()
    r2_baseline_prom = df_oos_baseline['R2'].mean()
    filas_verif.append({
        'Seccion': 'A.promedios', 'Articulo': 'TODOS', 'Metrica': 'R2_promedio_Prophet_oos',
        'valor_propio': r2_prophet_prom, 'valor_documentado_notebook': -4.37,
        'delta_relativo': _delta_rel(r2_prophet_prom, -4.37),
        'coincide_con_documentado': _delta_rel(r2_prophet_prom, -4.37) < TOL_REL,
    })
    filas_verif.append({
        'Seccion': 'A.promedios', 'Articulo': 'TODOS', 'Metrica': 'R2_promedio_baseline_oos',
        'valor_propio': r2_baseline_prom, 'valor_documentado_notebook': 0.27,
        'delta_relativo': _delta_rel(r2_baseline_prom, 0.27),
        'coincide_con_documentado': _delta_rel(r2_baseline_prom, 0.27) < TOL_REL,
    })
    df_verif = pd.DataFrame(filas_verif)

    tabla_params = pd.DataFrame([
        {'Articulo': a, **params_por_articulo[a]} for a in articulos
    ])

    df_insample.to_csv(OUT / 'insample.csv', index=False)
    df_oos_prophet.to_csv(OUT / 'outofsample_prophet.csv', index=False)
    df_oos_baseline.to_csv(OUT / 'outofsample_baseline.csv', index=False)
    df_verif.to_csv(OUT / 'verificacion_reproducibilidad.csv', index=False)
    tabla_params.to_csv(OUT / 'hiperparametros_por_articulo.csv', index=False)

    print("=== Hiperparámetros cacheados (hiperparametros_prophet.json) ===")
    print(tabla_params.to_string(index=False))
    print("\n=== In-sample (MSE/RMSE/MAE/R2/MAPE/WAPE) ===")
    print(df_insample.to_string(index=False))
    print("\n=== Out-of-sample Prophet (hold-out 12 meses) ===")
    print(df_oos_prophet.to_string(index=False))
    print("\n=== Out-of-sample baseline naive estacional ===")
    print(df_oos_baseline.to_string(index=False))
    print(f"\nR2 promedio Prophet oos: {r2_prophet_prom:.3f} (documentado: -4.37)")
    print(f"R2 promedio baseline oos: {r2_baseline_prom:.3f} (documentado: 0.27)")
    print("\n=== Verificación de reproducibilidad ===")
    print(df_verif.to_string(index=False))
    print(f"\nCSVs escritos en {OUT}")


if __name__ == '__main__':
    main()
