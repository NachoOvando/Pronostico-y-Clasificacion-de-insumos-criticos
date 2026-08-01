"""Variables de entrada al K-Means, a nivel familia de compra."""

import pandas as pd

from planificacion.config import InsumosConfig

# El orden importa: es el mismo de las columnas de los centroides y el de los pesos.
FEATURES: list[str] = ['Vol_norm', 'Alcance_pct', 'LeadTime_norm']


def consolidar_por_familia_de_compra(consumo: pd.DataFrame, cfg: InsumosConfig) -> pd.DataFrame:
    """Consolida el consumo proyectado por familia de compra."""
    familias = (
        consumo.groupby(['Familia', 'UM_familia'])
        .agg(
            Consumo_Total=('consumo_proyectado', 'sum'),
            N_Articulos=('Articulo_Padre', 'nunique'),
            N_SKUs_hijo=('Nombre_Base', 'nunique'),
            Lead_Time_dias=(cfg.leadtime_col, cfg.leadtime_agg),
            Proveedor=('Proveedor', 'first'),   # referencia, no entra al modelo
            Precio_unit=('Precio', 'max'),      # referencia, no entra al modelo
        )
        .reset_index()
        .rename(columns={'UM_familia': 'UM'})
    )
    familias = familias[familias['Consumo_Total'] > 0].copy()
    familias['Lead_Time_dias'] = pd.to_numeric(
        familias['Lead_Time_dias'], errors='coerce').fillna(0)
    return familias.reset_index(drop=True)


def calcular_variables_de_criticidad(familias: pd.DataFrame, total_articulos: int) -> pd.DataFrame:
    """Agrega las tres variables normalizadas 0–100 que consume el K-Means.

    | Variable        | Interpretación                                          |
    |-----------------|---------------------------------------------------------|
    | `Vol_norm`      | Peso del insumo dentro de su unidad de medida           |
    | `Alcance_pct`   | % de artículos padre que dependen del insumo            |
    | `LeadTime_norm` | Riesgo de abastecimiento: a mayor LT, menos reacción    |

    Normalizar es condición necesaria para que ninguna variable domine
    arbitrariamente la distancia euclidiana del K-Means.
    """
    df = familias.copy()

    df['Alcance_pct'] = df['N_Articulos'] / total_articulos * 100

    # Volumen relativo DENTRO de cada UM: gramos, pares y unidades no son comparables entre sí.
    max_por_um = df.groupby('UM')['Consumo_Total'].transform('max')
    df['Vol_norm'] = df['Consumo_Total'] / max_por_um * 100

    lt_min, lt_max = df['Lead_Time_dias'].min(), df['Lead_Time_dias'].max()
    if lt_max > lt_min:
        df['LeadTime_norm'] = (df['Lead_Time_dias'] - lt_min) / (lt_max - lt_min) * 100
    else:
        # Todas las familias con el mismo Lead Time: la variable no aporta varianza.
        df['LeadTime_norm'] = 0.0

    return df.reset_index(drop=True)


def score_ahp(datos, cfg: InsumosConfig):
    """Score multicriterio con los pesos AHP de `cfg`.

    Acepta un DataFrame con las columnas de `FEATURES` o un array/matriz cuyas
    columnas estén en ese mismo orden. Es la única implementación del score: se
    usa tanto para etiquetar los centroides como para ordenar el heatmap.
    """
    pesos = [cfg.pesos[f] for f in FEATURES]
    if isinstance(datos, pd.DataFrame):
        return sum(datos[f] * p for f, p in zip(FEATURES, pesos))
    return datos[:, 0] * pesos[0] + datos[:, 1] * pesos[1] + datos[:, 2] * pesos[2]
