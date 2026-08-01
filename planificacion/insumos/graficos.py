"""Gráficos del notebook de insumos críticos.

Ninguna de estas funciones calcula variables del modelo: reciben los DataFrames
ya construidos y solo los dibujan. Los colores salen de `planificacion.estilos`.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from planificacion import estilos
from planificacion.config import InsumosConfig
from planificacion.insumos.clustering import ClasificacionInsumos
from planificacion.insumos.features import FEATURES, score_ahp

SIZE_MIN, SIZE_MAX = 90, 520


def curva_talles(df_curva: pd.DataFrame) -> plt.Figure:
    """Distribución de talles usada para desagregar el forecast."""
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(df_curva['talle'], df_curva['proporcion'] * 100,
           color=estilos.AZUL_MEDIO, alpha=0.8, edgecolor='white')
    ax.set_xlabel('Talle')
    ax.set_ylabel('Proporción (%)')
    ax.set_title('Curva de Talles — Distribución Normal')
    ax.set_xticks(df_curva['talle'])
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig


def seleccion_k(metricas_por_k: pd.DataFrame, cfg: InsumosConfig) -> plt.Figure:
    """Silhouette y método del codo, con el K elegido marcado."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Selección del K óptimo', fontsize=12, fontweight='bold')

    ax1.plot(metricas_por_k['K'], metricas_por_k['Silhouette'], 'o-', color=estilos.AZUL_MEDIO,
             linewidth=2, markersize=8)
    for k, s in zip(metricas_por_k['K'], metricas_por_k['Silhouette']):
        ax1.annotate(f'{s:.3f}', (k, s), textcoords='offset points',
                     xytext=(0, 9), ha='center', fontsize=9)
    ax1.set_title('Silhouette Score (↑ mejor)')
    ax1.set_ylabel('Silhouette Score')

    ax2.plot(metricas_por_k['K'], metricas_por_k['Inercia'], 'o-', color=estilos.NARANJA,
             linewidth=2, markersize=8)
    ax2.set_title('Inercia — Método del Codo (↓ mejor)')
    ax2.set_ylabel('Inercia (WCSS)')

    for ax in (ax1, ax2):
        ax.axvline(x=cfg.k_clusters, color='#C00000', linestyle='--', linewidth=1.5,
                   label=f'K elegido = {cfg.k_clusters}')
        ax.set_xlabel('K')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def _tamanos_por_leadtime(df_resultado: pd.DataFrame) -> pd.Series:
    """Tamaño de cada punto proporcional a su Lead Time."""
    lt_min, lt_max = df_resultado['Lead_Time_dias'].min(), df_resultado['Lead_Time_dias'].max()
    if lt_max > lt_min:
        return SIZE_MIN + (df_resultado['LeadTime_norm'] / 100) * (SIZE_MAX - SIZE_MIN)
    return pd.Series((SIZE_MIN + SIZE_MAX) / 2, index=df_resultado.index)


def _recuadro_zoom(df_resultado: pd.DataFrame) -> tuple[float, float, float, float]:
    """Bounding box de la zona densa (familias de bajo volumen y alcance)."""
    mascara = (df_resultado['Vol_norm'] <= 35) & (df_resultado['Alcance_pct'] <= 80)
    if mascara.sum() < 3:
        return -3, 35, 20, 75
    return (max(df_resultado.loc[mascara, 'Vol_norm'].min() - 4, -3),
            df_resultado.loc[mascara, 'Vol_norm'].max() + 4,
            max(df_resultado.loc[mascara, 'Alcance_pct'].min() - 6, 0),
            df_resultado.loc[mascara, 'Alcance_pct'].max() + 6)


def dispersion(df_resultado: pd.DataFrame, clasificacion: ClasificacionInsumos,
               cfg: InsumosConfig, path: Path | None = None) -> plt.Figure:
    """Panorama general + zoom de la zona densa.

    Cada punto lleva el número de su fila en la Tabla de clasificación: escribir
    el nombre completo de cada familia era ilegible al acumularse los puntos.
    """
    tam = _tamanos_por_leadtime(df_resultado)
    zx0, zx1, zy0, zy1 = _recuadro_zoom(df_resultado)
    lt_min = df_resultado['Lead_Time_dias'].min()
    lt_max = df_resultado['Lead_Time_dias'].max()

    def dibujar(ax):
        ax.set_facecolor(estilos.FONDO)
        for idx, fila in df_resultado.iterrows():
            ax.scatter(fila['Vol_norm'], fila['Alcance_pct'],
                       c=estilos.COLOR_CRIT[fila['Criticidad']],
                       marker=estilos.UM_MARKER.get(fila['UM'], 'o'),
                       s=tam[idx], edgecolors='#333333', linewidth=0.8,
                       zorder=3, alpha=0.88)
            ax.annotate(str(idx), (fila['Vol_norm'], fila['Alcance_pct']),
                        ha='center', va='center', fontsize=7.3, fontweight='bold',
                        color='white' if fila['Criticidad'] == 'CRÍTICO' else '#222222',
                        zorder=4)
        for etiqueta in cfg.etiquetas:
            cx, cy, _ = clasificacion.centroide(etiqueta)
            ax.scatter(cx, cy, c=estilos.COLOR_CRIT[etiqueta], marker='*', s=380,
                       edgecolors='white', linewidth=1.1, zorder=5)
        ax.grid(True, alpha=0.22, linestyle='--')

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15.5, 7.2),
                                     gridspec_kw={'width_ratios': [1.05, 1]})
    fig.patch.set_facecolor(estilos.FONDO)

    dibujar(ax_a)
    ax_a.set_xlim(-5, 108)
    ax_a.set_ylim(15, 112)
    ax_a.set_xlabel('Volumen relativo dentro de la UM  (0–100)', fontsize=9.5)
    ax_a.set_ylabel('Alcance productivo  (% de artículos padre)', fontsize=9.5)
    ax_a.set_title('(a) Panorama general', fontsize=11, fontweight='bold')
    ax_a.add_patch(mpatches.Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False,
                                      edgecolor='#444444', linestyle='--',
                                      linewidth=1.4, zorder=6))
    ax_a.annotate('zoom →', (zx1, zy1), fontsize=8, color='#444444',
                  xytext=(4, 4), textcoords='offset points', fontweight='bold')

    dibujar(ax_b)
    ax_b.set_xlim(zx0, zx1)
    ax_b.set_ylim(zy0, zy1)
    ax_b.set_xlabel('Volumen relativo dentro de la UM  (0–100)', fontsize=9.5)
    ax_b.set_title('(b) Zoom — zona de baja prioridad', fontsize=11, fontweight='bold')

    fig.suptitle(
        f'Identificación de Insumos Críticos por Familia de Compra — '
        f'K-Means (K={cfg.k_clusters})\n'
        'Números = ID de la Tabla de clasificación  ·  Tamaño = Lead Time  ·  ★ = centroide',
        fontsize=12.5, fontweight='bold', color=estilos.AZUL, y=1.04)

    leyenda_crit = ax_a.legend(
        handles=[mpatches.Patch(color=estilos.COLOR_CRIT[e], label=e) for e in cfg.etiquetas],
        title='Criticidad', loc='lower right', fontsize=8.5, title_fontsize=9)
    ax_a.add_artist(leyenda_crit)
    ax_a.legend(
        handles=[plt.Line2D([0], [0], marker=estilos.UM_MARKER[um], color='w',
                            markerfacecolor='#555', markersize=8,
                            label=estilos.UM_LABEL[um]) for um in ('G', 'PAA', 'UN')],
        title='Unidad de medida', loc='upper left', fontsize=8, title_fontsize=8.5)
    ax_b.legend(
        handles=[plt.scatter([], [], s=SIZE_MIN, c='#999', edgecolors='#333',
                             label=f'LT mín. ({lt_min:.0f} d)'),
                 plt.scatter([], [], s=SIZE_MAX, c='#999', edgecolors='#333',
                             label=f'LT máx. ({lt_max:.0f} d)')],
        title='Tamaño = Lead Time', loc='lower right', fontsize=8, title_fontsize=8.5)

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig


def coordenadas_paralelas(df_resultado: pd.DataFrame, clasificacion: ClasificacionInsumos,
                          cfg: InsumosConfig, path: Path | None = None) -> plt.Figure:
    """Perfil completo de las 3 variables por familia.

    El gráfico de dispersión proyecta 3 variables en un plano y compacta
    información; acá se ve que cada nivel de criticidad es un patrón, no un punto.
    """
    etiquetas_eje = ['Volumen\nrelativo', 'Alcance\nproductivo', 'Lead Time\n(normalizado)']
    x = list(range(len(FEATURES)))

    fig, ax = plt.subplots(figsize=(8.5, 6))
    fig.patch.set_facecolor(estilos.FONDO)
    ax.set_facecolor(estilos.FONDO)

    for _, fila in df_resultado.iterrows():
        ax.plot(x, [fila[f] for f in FEATURES],
                color=estilos.COLOR_CRIT[fila['Criticidad']], alpha=0.55,
                linewidth=1.4, marker='o', markersize=4, zorder=2)

    for etiqueta in cfg.etiquetas:
        ax.plot(x, clasificacion.centroide(etiqueta), color=estilos.COLOR_CRIT[etiqueta],
                linewidth=4.2, marker='D', markersize=9,
                label=f'Centroide {etiqueta}', zorder=5, solid_capstyle='round')

    for xi in x:
        ax.axvline(xi, color='#999999', linewidth=0.8, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas_eje, fontsize=10)
    ax.set_ylabel('Valor normalizado (0–100)', fontsize=10)
    ax.set_ylim(-5, 108)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.set_title('Perfil de las 3 variables de clasificación por familia\n'
                 'Coordenadas Paralelas — color = Criticidad asignada por K-Means',
                 fontsize=12, fontweight='bold', color=estilos.AZUL)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3,
              fontsize=9, frameon=False)

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig


def matriz_decision(df_resultado: pd.DataFrame, cfg: InsumosConfig,
                    path: Path | None = None) -> plt.Figure:
    """Heatmap familias × variables + Score AHP, ordenado por score descendente.

    Permite ver, variable por variable, *por qué* cada familia quedó donde quedó.
    """
    df_heat = (df_resultado
               .assign(Score_AHP=lambda d: score_ahp(d, cfg))
               .sort_values('Score_AHP', ascending=False))

    columnas = FEATURES + ['Score_AHP']
    etiquetas_col = [f'Volumen\nrelativo\n(W={cfg.w_volumen:.2f})',
                     f'Alcance\nproductivo\n(W={cfg.w_alcance:.2f})',
                     f'Lead Time\n(norm.)\n(W={cfg.w_leadtime:.2f})',
                     'Score\nAHP']
    datos = df_heat[columnas].values
    n_fam = len(df_heat)

    fig, ax = plt.subplots(figsize=(9.5, 0.42 * n_fam + 2.2))
    fig.patch.set_facecolor(estilos.FONDO)
    im = ax.imshow(datos, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(range(len(columnas)))
    ax.set_xticklabels(etiquetas_col, fontsize=9)
    ax.set_yticks(range(n_fam))
    ax.set_yticklabels([f'{idx}. {fam[:32]}'
                        for idx, fam in zip(df_heat.index, df_heat['Familia'])],
                       fontsize=8.2)
    ax.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False)

    for i in range(datos.shape[0]):
        for j in range(datos.shape[1]):
            valor = datos[i, j]
            ax.text(j, i, f'{valor:.0f}', ha='center', va='center', fontsize=7.6,
                    color='white' if valor > 55 else '#222222', fontweight='bold')

    # Franja de criticidad a la izquierda + separadores entre grupos
    anterior = None
    for i, criticidad in enumerate(df_heat['Criticidad']):
        ax.add_patch(mpatches.Rectangle((-1.0, i - 0.5), 0.35, 1.0,
                                        color=estilos.COLOR_CRIT[criticidad],
                                        clip_on=False))
        if anterior is not None and criticidad != anterior:
            ax.axhline(i - 0.5, color='black', linewidth=1.8, zorder=5)
        anterior = criticidad

    ax.set_xlim(-1.3, len(columnas) - 0.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label('Valor normalizado (0–100)', fontsize=9)

    ax.legend(handles=[mpatches.Patch(color=estilos.COLOR_CRIT[e], label=e)
                       for e in cfg.etiquetas],
              title='Criticidad (franja izq.)', loc='upper center',
              bbox_to_anchor=(0.5, -0.04 - 0.012 * n_fam), ncol=3,
              fontsize=8.5, title_fontsize=9, frameon=False)
    ax.set_title('Matriz de decisión multicriterio — Familias × Variables normalizadas\n'
                 'Ordenado por Score AHP descendente',
                 fontsize=12, fontweight='bold', color=estilos.AZUL, pad=14)

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig
