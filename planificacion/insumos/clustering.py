"""K-Means sobre las familias de compra: selección de K, entrenamiento y etiquetado."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from planificacion.config import InsumosConfig
from planificacion.insumos.features import FEATURES, score_ahp


def escalar_variables(familias: pd.DataFrame) -> tuple[np.ndarray, MinMaxScaler]:
    """Lleva las tres features a [0, 1] para que pesen igual en la distancia euclidiana."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(familias[FEATURES].values), scaler


def evaluar_cantidad_de_clusters(variables_escaladas: np.ndarray,
                                 cfg: InsumosConfig) -> pd.DataFrame:
    """Silhouette e inercia para cada K del rango, para justificar el K elegido.

    El Silhouette mide qué tan separado está cada punto de su cluster vecino
    (−1 malo, +1 perfecto); la inercia es la suma de distancias intra-cluster.
    """
    filas = []
    for k in cfg.k_range:
        km = KMeans(n_clusters=k, random_state=cfg.random_state, n_init=cfg.n_init)
        etiquetas = km.fit_predict(variables_escaladas)
        filas.append({'K': k,
                      'Silhouette': silhouette_score(variables_escaladas, etiquetas),
                      'Inercia': km.inertia_})
    return pd.DataFrame(filas)


@dataclass
class ClasificacionInsumos:
    """Resultado del K-Means, con los clusters ya traducidos a niveles de criticidad."""

    familias: pd.DataFrame          # con Cluster_raw, Criticidad y Es_Compra_Externa
    centroides: np.ndarray        # escala original, indexados por cluster crudo
    orden: np.ndarray             # clusters ordenados por score AHP descendente
    mapa_etiquetas: dict[int, str]
    silhouette: float

    def centroide(self, etiqueta: str) -> np.ndarray:
        """Centroide del cluster que quedó etiquetado como `etiqueta`."""
        crudo = next(c for c, e in self.mapa_etiquetas.items() if e == etiqueta)
        return self.centroides[crudo]

    def resumen_centroides(self) -> str:
        lineas = []
        for etiqueta in self.mapa_etiquetas.values():
            vol, alcance, lead = self.centroide(etiqueta)
            lineas.append(f"  [{etiqueta}]  Vol_norm={vol:>6.1f}  "
                          f"Alcance_pct={alcance:>5.1f}  LeadTime_norm={lead:>6.1f}")
        return '\n'.join(lineas)


def clasificar_por_kmeans(familias: pd.DataFrame, variables_escaladas: np.ndarray,
             scaler: MinMaxScaler, cfg: InsumosConfig) -> ClasificacionInsumos:
    """Entrena el K-Means final y traduce cada cluster a un nivel de criticidad.

    El cluster con mayor score AHP de su centroide es el CRÍTICO. El clustering
    corre sobre la población completa —incluida la producción interna— para que
    la normalización por UM y el rango del Lead Time no cambien; el recorte al
    alcance de Compras se hace después, con `Es_Compra_Externa`.
    """
    km = KMeans(n_clusters=cfg.k_clusters, random_state=cfg.random_state,
                n_init=cfg.n_init)

    df = familias.copy()
    df['Cluster_raw'] = km.fit_predict(variables_escaladas)

    centroides = scaler.inverse_transform(km.cluster_centers_)
    orden = np.argsort(score_ahp(centroides, cfg))[::-1]   # mayor score primero
    mapa_etiquetas = {int(orden[i]): cfg.etiquetas[i] for i in range(cfg.k_clusters)}

    df['Criticidad'] = df['Cluster_raw'].map(mapa_etiquetas)
    # Las familias sin proveedor son producción interna (capelladas y plantillas
    # semielaboradas): quedan fuera del alcance de Compras.
    df['Es_Compra_Externa'] = df['Proveedor'].notna()

    return ClasificacionInsumos(
        familias=df,
        centroides=centroides,
        orden=orden,
        mapa_etiquetas=mapa_etiquetas,
        silhouette=float(silhouette_score(variables_escaladas, df['Cluster_raw'])),
    )


def tabla_de_familias_clasificadas(clasificacion: ClasificacionInsumos,
                                   cfg: InsumosConfig) -> pd.DataFrame:
    """Familias dentro del alcance de Compras, ordenadas por criticidad y volumen.

    El índice arranca en 1 porque es el número con el que cada familia aparece
    en los gráficos de la Sección 7.
    """
    orden_crit = {e: i for i, e in enumerate(cfg.etiquetas)}
    df = (clasificacion.familias[clasificacion.familias['Es_Compra_Externa']]
          .assign(_o=lambda d: d['Criticidad'].map(orden_crit))
          .sort_values(['_o', 'Vol_norm'], ascending=[True, False])
          .drop(columns=['_o', 'Cluster_raw', 'Es_Compra_Externa'])
          .reset_index(drop=True))
    df.index += 1
    return df


def tabla_de_skus_por_familia(consumo: pd.DataFrame, df_resultado: pd.DataFrame,
               familias: pd.DataFrame, total_articulos: int,
               cfg: InsumosConfig) -> pd.DataFrame:
    """Variantes de talle que componen cada familia, para emitir órdenes de compra."""
    orden_crit = {e: i for i, e in enumerate(cfg.etiquetas)}
    df_sku = (
        consumo[consumo['Familia'].isin(df_resultado['Familia'])]
        .groupby(['Familia', 'UM_familia', 'Nombre_Base'])
        .agg(Consumo_SKU=('consumo_proyectado', 'sum'),
             N_Art=('Articulo_Padre', 'nunique'))
        .reset_index()
        .rename(columns={'UM_familia': 'UM'})
    )
    df_sku = df_sku[df_sku['Consumo_SKU'] > 0]
    df_sku['Alcance_pct'] = df_sku['N_Art'] / total_articulos * 100
    df_sku = df_sku.merge(familias[['Familia', 'Criticidad']], on='Familia', how='left')
    df_sku = (df_sku
              .assign(_o=lambda d: d['Criticidad'].map(orden_crit))
              .sort_values(['_o', 'Familia', 'Consumo_SKU'],
                           ascending=[True, True, False])
              .drop(columns='_o')
              .reset_index(drop=True))
    df_sku.index += 1
    return df_sku
