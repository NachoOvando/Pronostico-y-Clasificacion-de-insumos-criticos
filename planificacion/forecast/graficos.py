"""Gráficos del notebook de pronóstico. No calculan nada: reciben el DataFrame ya armado."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def aplicar_estilo() -> None:
    """Estilo común a todos los gráficos del notebook."""
    sns.set_style('whitegrid')


def series_historicas(df_long: pd.DataFrame) -> plt.Figure:
    """Ventas mensuales por artículo — sirve para leer tendencia y estacionalidad."""
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.lineplot(data=df_long, x='Fecha', y='Volumen_Ventas',
                 hue='Articulo', marker='o', ax=ax)
    ax.set_title('Volumen de Ventas Mensuales por Artículo')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Volumen de Ventas')
    ax.legend(title='Artículo')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return fig


def historico_y_pronostico(df_combinado: pd.DataFrame,
                           ultima_fecha: pd.Timestamp,
                           horizonte: int) -> plt.Figure:
    """Serie continua histórico + pronóstico, con el corte marcado."""
    fig, ax = plt.subplots(figsize=(18, 9))
    sns.lineplot(data=df_combinado, x='Fecha', y='Volumen_Ventas',
                 hue='Articulo', marker='o', alpha=0.8, ax=ax)
    ax.axvline(x=ultima_fecha, color='grey', linestyle='--',
               label='Fin de datos históricos')
    ax.set_title(f'Ventas Históricas y Pronóstico a {horizonte} Meses por Artículo '
                 '(Prophet optimizado)')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Volumen de Ventas')
    ax.legend(title='Artículo', loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return fig
