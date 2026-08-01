"""Armado del pronóstico final y export al contrato que consume el K-Means."""

from pathlib import Path

import pandas as pd

from planificacion.forecast.modelo import Ajuste

COLUMNAS = ['Articulo', 'Fecha', 'Volumen_Ventas']


def _a_largo(ajuste: pd.DataFrame, articulo: str) -> pd.DataFrame:
    """`ds`/`yhat` -> `Articulo`/`Fecha`/`Volumen_Ventas`, recortado a 0.

    El recorte es una regla de negocio: las ventas no pueden ser negativas.
    """
    df = ajuste.rename(columns={'ds': 'Fecha', 'yhat': 'Volumen_Ventas'}).copy()
    df['Volumen_Ventas'] = df['Volumen_Ventas'].clip(lower=0)
    df['Articulo'] = articulo
    return df[COLUMNAS]


def separar_ajuste_historico_y_pronostico(
        ajustes: dict[str, Ajuste]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve `(futuro, insample)` en formato largo, listos para graficar y exportar."""
    futuro = [_a_largo(a.futuro, articulo) for articulo, a in ajustes.items()]
    insample = [_a_largo(a.insample, articulo) for articulo, a in ajustes.items()]
    return (pd.concat(futuro, ignore_index=True),
            pd.concat(insample, ignore_index=True))


def combinar_historico_y_pronostico(df_historico: pd.DataFrame,
                                    df_futuro: pd.DataFrame) -> pd.DataFrame:
    """Serie continua histórico + pronóstico, para el gráfico de la Sección 10."""
    return (pd.concat([df_historico, df_futuro], ignore_index=True)
            .sort_values(by=['Articulo', 'Fecha'])
            .reset_index(drop=True))


def exportar_pronostico_a_excel(df_historico: pd.DataFrame, df_insample: pd.DataFrame,
             df_futuro: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Exporta `Articulo | Fecha | Ventas | Tipo` al Excel que consume el K-Means.

    Contrato de datos con `Insumos_Criticos_KMeans.ipynb`: el `yhat` in-sample se
    exporta a propósito, junto con el tramo futuro, bajo `Tipo = Pronostico`. El
    notebook de insumos críticos estima la variabilidad de la demanda (σ_d) como
    `Historico − Pronostico` sobre las mismas fechas; si se exportara solo el
    tramo futuro, ese cálculo quedaría sin datos.
    """
    hist = df_historico.copy()
    hist['Tipo'] = 'Historico'

    pron = pd.concat([df_insample, df_futuro], ignore_index=True)
    pron['Tipo'] = 'Pronostico'

    df_final = (pd.concat([hist, pron], ignore_index=True)
                .rename(columns={'Volumen_Ventas': 'Ventas'})
                [['Articulo', 'Fecha', 'Ventas', 'Tipo']]
                .sort_values(by=['Articulo', 'Fecha'])
                .reset_index(drop=True))

    df_final.to_excel(path, index=False)
    return df_final
