"""Sección F — política de inventario: reproducción in-sample + análisis out-of-sample.

1. Reproduce `politicas.calcular_politicas_de_inventario` tal cual (sin tocar
   `planificacion/insumos/politicas.py`) para los 3 insumos de
   `POLITICAS_POR_DEFECTO` y verifica contra
   `Politicas_Inventario_Insumos_Criticos.xlsx` (ya existente, solo lectura).

2. Análisis ADICIONAL pedido por el usuario (no reemplaza nada): recalcula σ
   con los residuos del hold-out de 12 meses (out-of-sample) en vez del
   residuo in-sample que usa `politicas.demanda_por_articulo`
   (`planificacion/insumos/politicas.py:36-65`), y reporta el %diff en
   Stock_Seguridad / ROP / Stock_Maximo. Reimplementa el tramo de agregación
   por familia de `calcular_politicas_de_inventario` (líneas 105-148)
   sustituyendo únicamente `demanda.sigma` por `sigma_oos`, porque esa
   función no acepta un σ externo como parámetro; reusa `politicas._lead_time`
   (función "privada" del módulo, pero de solo lectura) para no duplicar esa
   lógica.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import DIR_ANALISIS, cargar_params_cacheados, cargar_ventas_cache
from main import explotar_demanda_en_requerimientos_de_insumos
from planificacion.config import ForecastConfig, InsumosConfig
from planificacion.forecast import datos, modelo
from planificacion.insumos import clustering, politicas

OUT = DIR_ANALISIS / 'outputs' / 'F_politicas'
OUT.mkdir(parents=True, exist_ok=True)


def calcular_sigma_out_of_sample(cfg_forecast: ForecastConfig) -> pd.Series:
    """σ mensual por artículo, estimado con el error del hold-out de 12 meses
    (out-of-sample), en vez del residuo in-sample de `politicas.demanda_por_articulo`.

    Replica el split de `evaluacion.evaluar_holdout` (train=serie.iloc[:-12],
    test=serie.iloc[-12:]) con los hiperparámetros ya cacheados.
    """
    modelo.silenciar_logs_prophet()
    df_wide = cargar_ventas_cache(cfg_forecast)
    df_long = datos.pasar_a_formato_largo(df_wide)
    df_prophet = datos.a_formato_prophet(df_long)
    params_por_articulo = cargar_params_cacheados(cfg_forecast)

    sigmas = {}
    for articulo in sorted(df_prophet['Articulo'].unique()):
        serie = datos.serie_articulo(df_prophet, articulo)
        train = serie.iloc[:-cfg_forecast.test_periods]
        test = serie.iloc[-cfg_forecast.test_periods:]
        m = modelo.crear_modelo(params_por_articulo[articulo])
        m.fit(train)
        future = m.make_future_dataframe(periods=cfg_forecast.test_periods, freq='MS')
        pred = m.predict(future)[['ds', 'yhat']].merge(test, on='ds', how='inner')
        residuo = pred['y'] - pred['yhat']
        sigmas[articulo] = residuo.std(ddof=1)  # mismo criterio que politicas.py:54 (.std())

    return pd.Series(sigmas, name='sigma_oos')


def calcular_politicas_con_sigma_custom(df_forecast: pd.DataFrame, df_bom: pd.DataFrame,
                                        df_curva: pd.DataFrame, tipo_pronostico: str,
                                        cfg: InsumosConfig, familias: pd.DataFrame,
                                        media_mensual: pd.Series, sigma: pd.Series) -> pd.DataFrame:
    """Reimplementación literal de `politicas.calcular_politicas_de_inventario`
    (planificacion/insumos/politicas.py:93-150), sustituyendo únicamente el
    origen de `sigma` (esa función siempre usa el residuo in-sample; acá se
    inyecta el que se le pase). El resto de la fórmula es idéntico."""
    filas = []
    for politica in cfg.politicas:
        sub = df_bom[df_bom['Componente de lista de materia']
                     .str.contains(politica.patron, case=False, na=False)].copy()
        sub = sub.merge(df_curva, left_on='Talle', right_on='talle', how='left')
        sub['Cantidad'] = pd.to_numeric(sub['Cantidad'], errors='coerce').fillna(0)

        consumo_unitario = (sub['proporcion'] * sub['Cantidad']).groupby(sub['Articulo']).sum()
        consumo_unitario = consumo_unitario[consumo_unitario.index.isin(media_mensual.index)]

        d_mes = float((consumo_unitario * media_mensual.reindex(consumo_unitario.index)).sum())
        sigma_mes = float(np.sqrt(
            ((consumo_unitario * sigma.reindex(consumo_unitario.index)) ** 2).sum()))

        lt_dias = politicas._lead_time(politica, sub, familias, cfg)
        horizonte_meses = lt_dias / politicas.DIAS_POR_MES + politica.R_meses

        stock_seguridad = cfg.z_servicio * sigma_mes * np.sqrt(horizonte_meses)
        rop_o_nivel = d_mes * horizonte_meses + stock_seguridad
        ciclo = 1.0 if politica.continua else politica.R_meses
        stock_maximo = rop_o_nivel + d_mes * ciclo

        filas.append({
            'Familia': politica.familia, 'UM': sub['UM'].iloc[0],
            'Política': politica.nombre_politica, 'Lead_Time_dias': lt_dias,
            'Demanda_media_mensual': round(d_mes, 1), 'Desvio_mensual': round(sigma_mes, 1),
            'Stock_Seguridad': round(stock_seguridad, 1),
            'ROP_o_Nivel_Objetivo': round(rop_o_nivel, 1),
            'Stock_Maximo': round(stock_maximo, 1),
        })
    return pd.DataFrame(filas)


def main() -> None:
    cfg_insumos = InsumosConfig()
    cfg_forecast = ForecastConfig()

    requerimientos = explotar_demanda_en_requerimientos_de_insumos(cfg_insumos)
    variables_escaladas, scaler = clustering.escalar_variables(requerimientos.familias_de_compra)
    clasificacion = clustering.clasificar_por_kmeans(
        requerimientos.familias_de_compra, variables_escaladas, scaler, cfg_insumos)

    # ── 1. Reproducción in-sample (tal cual politicas.py) ──
    df_insample = politicas.calcular_politicas_de_inventario(
        requerimientos.ventas_pronosticadas, requerimientos.lista_de_materiales,
        requerimientos.curva_de_talles, requerimientos.tipo_pronostico, cfg_insumos,
        familias=clasificacion.familias)
    print("=== Política in-sample (politicas.calcular_politicas_de_inventario, real) ===")
    print(df_insample.to_string(index=False))

    path_excel = cfg_insumos.politicas_output_path
    if path_excel.exists():
        df_excel = pd.read_excel(path_excel)
        merge = df_insample.merge(df_excel, on='Familia', suffixes=('_recalc', '_excel'))
        cols_check = ['Stock_Seguridad', 'ROP_o_Nivel_Objetivo', 'Stock_Maximo']
        for c in cols_check:
            merge[f'{c}_delta_abs'] = (merge[f'{c}_recalc'] - merge[f'{c}_excel']).abs()
        print(f"\nVerificación vs. {path_excel.name} (ya existente):")
        print(merge[['Familia'] + [f'{c}_recalc' for c in cols_check]
                    + [f'{c}_excel' for c in cols_check]
                    + [f'{c}_delta_abs' for c in cols_check]].to_string(index=False))
        merge.to_csv(OUT / 'insample_vs_excel.csv', index=False)
        max_delta = merge[[f'{c}_delta_abs' for c in cols_check]].values.max()
        print(f"Delta absoluto máximo: {max_delta:.4f} "
             f"({'reproduce exacto (dentro de redondeo)' if max_delta < 1.0 else 'DIVERGE'})")
    else:
        print(f"\n{path_excel.name} no existe: no se puede verificar contra un resultado previo.")

    # ── 2. Análisis adicional: σ out-of-sample (hold-out 12 meses) ──
    demanda_insample = politicas.demanda_por_articulo(
        requerimientos.ventas_pronosticadas, requerimientos.tipo_pronostico)
    sigma_oos = calcular_sigma_out_of_sample(cfg_forecast)
    print("\n=== σ mensual: in-sample vs. out-of-sample, por artículo ===")
    comp_sigma = pd.DataFrame({
        'sigma_insample': demanda_insample.sigma,
        'sigma_outofsample': sigma_oos.reindex(demanda_insample.sigma.index),
    })
    comp_sigma['ratio_oos_sobre_insample'] = (
        comp_sigma['sigma_outofsample'] / comp_sigma['sigma_insample'])
    print(comp_sigma.to_string())
    comp_sigma.to_csv(OUT / 'sigma_insample_vs_outofsample.csv')

    df_oos = calcular_politicas_con_sigma_custom(
        requerimientos.ventas_pronosticadas, requerimientos.lista_de_materiales,
        requerimientos.curva_de_talles, requerimientos.tipo_pronostico, cfg_insumos,
        familias=clasificacion.familias, media_mensual=demanda_insample.media_mensual,
        sigma=sigma_oos)
    print("\n=== Política recalculada con σ OUT-OF-SAMPLE (análisis adicional) ===")
    print(df_oos.to_string(index=False))

    comparacion = df_insample.merge(df_oos, on='Familia', suffixes=('_insample', '_outofsample'))
    for c in ['Desvio_mensual', 'Stock_Seguridad', 'ROP_o_Nivel_Objetivo', 'Stock_Maximo']:
        comparacion[f'{c}_pct_diff'] = (
            (comparacion[f'{c}_outofsample'] - comparacion[f'{c}_insample'])
            / comparacion[f'{c}_insample'] * 100)
    print("\n=== % diferencia (out-of-sample vs. in-sample) ===")
    cols_pct = ['Familia'] + [f'{c}_pct_diff' for c in
                              ('Desvio_mensual', 'Stock_Seguridad', 'ROP_o_Nivel_Objetivo', 'Stock_Maximo')]
    print(comparacion[cols_pct].to_string(index=False))

    df_insample.to_csv(OUT / 'politicas_insample.csv', index=False)
    df_oos.to_csv(OUT / 'politicas_outofsample.csv', index=False)
    comparacion.to_csv(OUT / 'comparacion_insample_vs_outofsample.csv', index=False)
    print(f"\nCSVs escritos en {OUT}")


if __name__ == '__main__':
    main()
