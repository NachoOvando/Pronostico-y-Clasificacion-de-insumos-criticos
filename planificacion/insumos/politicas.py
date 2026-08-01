"""Políticas de inventario (s,Q) y (R,S) para los insumos críticos seleccionados.

La demanda media y su variabilidad salen del propio modelo de pronóstico:

- d̄  = pronóstico Prophet de los meses futuros, propagado por la curva de talles
       y la BOM (el mismo mecanismo de explosión de la Sección 3).
- σ_d = error **in-sample** del modelo (venta histórica real − valor que el modelo
       predijo para esos mismos meses), propagado por la BOM.

⚠️ Al ser un error in-sample, subestima la incertidumbre de un pronóstico
genuinamente fuera de muestra: el Stock de Seguridad resultante es una cota
conservadora/mínima, no el valor definitivo.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from planificacion.config import InsumosConfig, PoliticaInsumo

logger = logging.getLogger(__name__)

DIAS_POR_MES = 30


@dataclass(frozen=True)
class DemandaArticulo:
    """Demanda mensual futura y variabilidad in-sample, por artículo terminado."""

    media_mensual: pd.Series   # índice: Articulo
    sigma: pd.Series           # índice: Articulo


def demanda_por_articulo(df_forecast: pd.DataFrame, tipo_pronostico: str) -> DemandaArticulo:
    """Separa el histórico del pronóstico y deriva d̄ y σ_d por artículo."""
    piv = df_forecast.pivot_table(index=['Articulo', 'Fecha'],
                                  columns='Tipo', values='Ventas')

    otras = [c for c in piv.columns if c != tipo_pronostico]
    if len(otras) != 1:
        raise ValueError(
            f"Se esperaba que la columna 'Tipo' tuviera exactamente dos valores "
            f"(histórico y pronóstico); se encontró {list(piv.columns)}. "
            f"Con '{tipo_pronostico}' como pronóstico quedan {len(otras)} candidatos "
            f"a histórico: {otras}."
        )
    col_hist = otras[0]

    # σ_d: desvío del residuo en las fechas donde conviven histórico y pronóstico.
    residuos = (piv.dropna()
                .assign(residuo=lambda d: d[col_hist] - d[tipo_pronostico]))
    sigma = residuos.groupby('Articulo')['residuo'].std()

    # d̄: promedio del pronóstico en los meses sin dato histórico real.
    futuro = piv[piv[col_hist].isna()][tipo_pronostico].reset_index()
    media = futuro.groupby('Articulo')[tipo_pronostico].mean()

    if media.empty:
        raise ValueError(
            "No hay meses futuros en el archivo de pronóstico (todos tienen dato "
            "histórico). Revisar el contrato de datos con Pronostico_Ventas.ipynb."
        )
    return DemandaArticulo(media_mensual=media, sigma=sigma)


def _lead_time(politica: PoliticaInsumo, sub: pd.DataFrame,
               familias: pd.DataFrame | None, cfg: InsumosConfig):
    """Lead time de la familia, tomado de `familias` para no tener dos fuentes.

    Si difiere del que sale de la BOM cruda (puede pasar cuando la familia
    incluye artículos sin forecast, que no llegan a `familias`), se avisa.
    """
    lt_bom = sub[cfg.leadtime_col].max()
    if familias is None:
        return lt_bom

    fila = familias[familias['Familia'] == politica.familia]
    if fila.empty:
        logger.warning("La familia '%s' no aparece en el resultado del clustering; "
                       "se usa el Lead Time de la BOM (%s días).", politica.familia, lt_bom)
        return lt_bom

    lt_fam = fila['Lead_Time_dias'].iloc[0]
    if abs(float(lt_fam) - float(lt_bom)) > 1e-9:
        logger.warning("Lead Time distinto para '%s': %.0f días según el clustering "
                       "vs %.0f según la BOM cruda. Se usa el del clustering.",
                       politica.familia, lt_fam, lt_bom)
    return lt_fam


def calcular_politicas_de_inventario(df_forecast: pd.DataFrame, df_bom: pd.DataFrame,
             df_curva: pd.DataFrame,
             tipo_pronostico: str, cfg: InsumosConfig,
             familias: pd.DataFrame | None = None) -> pd.DataFrame:
    """Calcula SS, ROP / Nivel Objetivo y Stock Máximo de cada insumo con política propia.

    - Revisión continua (s,Q): se repone al caer al Punto de Pedido; horizonte = LT.
    - Revisión periódica (R,S): se pide hasta el Nivel Objetivo cada R; horizonte = LT + R.
    """
    demanda = demanda_por_articulo(df_forecast, tipo_pronostico)
    filas = []

    for politica in cfg.politicas:
        sub = df_bom[df_bom['Componente de lista de materia']
                     .str.contains(politica.patron, case=False, na=False)].copy()
        if sub.empty:
            raise ValueError(
                f"Ningún componente de la BOM matchea el patrón "
                f"'{politica.patron}' de la familia '{politica.familia}'."
            )
        sub = sub.merge(df_curva, left_on='Talle', right_on='talle', how='left')
        sub['Cantidad'] = pd.to_numeric(sub['Cantidad'], errors='coerce').fillna(0)

        # Consumo del insumo por unidad vendida de cada artículo padre,
        # ponderado por la curva de talles.
        consumo_unitario = (sub['proporcion'] * sub['Cantidad']).groupby(sub['Articulo']).sum()
        consumo_unitario = consumo_unitario[
            consumo_unitario.index.isin(demanda.media_mensual.index)]

        d_mes = float((consumo_unitario
                       * demanda.media_mensual.reindex(consumo_unitario.index)).sum())
        # Los artículos son independientes entre sí: los desvíos se suman en cuadratura.
        sigma_mes = float(np.sqrt(
            ((consumo_unitario * demanda.sigma.reindex(consumo_unitario.index)) ** 2).sum()))

        lt_dias = _lead_time(politica, sub, familias, cfg)
        horizonte_meses = lt_dias / DIAS_POR_MES + politica.R_meses

        stock_seguridad = cfg.z_servicio * sigma_mes * np.sqrt(horizonte_meses)
        rop_o_nivel = d_mes * horizonte_meses + stock_seguridad
        ciclo = 1.0 if politica.continua else politica.R_meses
        stock_maximo = rop_o_nivel + d_mes * ciclo

        filas.append({
            'Familia': politica.familia,
            'UM': sub['UM'].iloc[0],
            'Política': politica.nombre_politica,
            'Lead_Time_dias': lt_dias,
            'Demanda_media_mensual': round(d_mes, 1),
            'Desvio_mensual': round(sigma_mes, 1),
            'Stock_Seguridad': round(stock_seguridad, 1),
            'ROP_o_Nivel_Objetivo': round(rop_o_nivel, 1),
            'Stock_Maximo': round(stock_maximo, 1),
            'Cobertura_SS_dias': round((stock_seguridad / d_mes) * DIAS_POR_MES, 1)
                                 if d_mes > 0 else np.nan,
        })

    return pd.DataFrame(filas)
