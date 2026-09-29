"""Explosión de la lista de materiales: forecast por artículo -> consumo por insumo."""

import unicodedata
from dataclasses import dataclass, field

import pandas as pd

# Variantes aceptadas al autodetectar qué valor de 'Tipo' es el pronóstico.
CANDIDATOS_TIPO = ('pronostico', 'forecast', 'proyectado', 'prediccion', 'prevision')


def normalizar(texto: str) -> str:
    """minúsculas, sin tildes, sin espacios extra — para comparar sin depender del formato."""
    texto = str(texto).strip().lower()
    return unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('ascii')


def detectar_tipo_pronostico(df_forecast: pd.DataFrame,
                             tipo_forzado: str | None = None) -> str:
    """Devuelve el valor de la columna `Tipo` que representa el pronóstico.

    Si `tipo_forzado` viene dado, se usa tal cual; si no, se busca una variante
    de 'pronostico'/'forecast'/'proyectado' sin distinguir tildes ni mayúsculas.
    """
    if tipo_forzado is not None:
        return tipo_forzado

    valores = list(df_forecast['Tipo'].unique())
    coincidencias = [v for v in valores
                     if any(c in normalizar(v) for c in CANDIDATOS_TIPO)]
    if not coincidencias:
        raise ValueError(
            f"No se pudo detectar automáticamente qué valor de 'Tipo' corresponde al "
            f"pronóstico.\nValores disponibles en tu archivo: {valores}\n"
            f"Solución: fijá `tipo_pronostico='<valor exacto>'` en InsumosConfig."
        )
    return coincidencias[0]


@dataclass
class Diagnostico:
    """Avisos del cruce Forecast × BOM, para revisar antes de dar por buenos los números."""

    articulos_sin_bom: list[str] = field(default_factory=list)
    articulos_sin_forecast: list[str] = field(default_factory=list)
    claves_sin_match: list[str] = field(default_factory=list)
    filas_sin_componente: int = 0
    filas_totales: int = 0

    def __str__(self) -> str:
        lineas = []
        if self.articulos_sin_bom:
            lineas.append(f"⚠️  Artículos en Forecast SIN BOM asociado: {self.articulos_sin_bom}")
        if self.articulos_sin_forecast:
            lineas.append("⚠️  Artículos en BOM SIN forecast asociado (se ignoran): "
                          f"{self.articulos_sin_forecast}")
        if self.claves_sin_match:
            lineas.append(
                f"⚠️  {len(self.claves_sin_match)} combinaciones Articulo_Talle del forecast "
                f"no tienen fila en la BOM (quedarán con consumo 0). Ejemplos: "
                f"{self.claves_sin_match[:5]}")
        if self.filas_sin_componente:
            lineas.append(f"⚠️  {self.filas_sin_componente:,} de {self.filas_totales:,} filas "
                          "no encontraron componente BOM (Cantidad nula → consumo 0)")
        if not lineas:
            lineas.append('✅ Cruce Forecast × BOM sin observaciones.')
        return '\n'.join(lineas)


def _clave(articulos: pd.Series, talles: pd.Series) -> pd.Series:
    """Clave de unión normalizada: sin espacios extra y con el talle como entero."""
    return articulos.astype(str).str.strip() + '_' + talles.astype(int).astype(str)


def explotar_a_consumo_de_insumos(df_forecast: pd.DataFrame, df_bom: pd.DataFrame,
             df_curva: pd.DataFrame,
             tipo_pronostico: str) -> tuple[pd.DataFrame, Diagnostico]:
    """Calcula el consumo proyectado de cada componente.

        Consumo = Σ_talles ( Forecast_artículo × p_talle × Cantidad_BOM )

    Trabaja sobre copias: no agrega columnas a los DataFrames recibidos.
    """
    df_f = df_forecast[df_forecast['Tipo'] == tipo_pronostico].copy()
    if df_f.empty:
        raise ValueError(
            f"El filtro Tipo == '{tipo_pronostico}' devolvió 0 filas. Revisá que el valor "
            f"coincida EXACTO (puede tener espacios extra) con alguno de: "
            f"{list(df_forecast['Tipo'].unique())}"
        )

    # Cross join forecast × curva de talles
    df_fd = pd.merge(df_f, df_curva, how='cross')
    df_fd['forecast_talle'] = (df_fd['Ventas'] * df_fd['proporcion']).round(2)
    df_fd['clave'] = _clave(df_fd['Articulo'], df_fd['talle'])

    bom = df_bom.copy()
    bom['clave'] = _clave(bom['Articulo'], bom['Talle'])

    arts_forecast = set(df_fd['Articulo'].str.strip().unique())
    arts_bom = set(bom['Articulo'].str.strip().unique())
    diag = Diagnostico(
        articulos_sin_bom=sorted(arts_forecast - arts_bom),
        articulos_sin_forecast=sorted(arts_bom - arts_forecast),
        claves_sin_match=sorted(set(df_fd['clave']) - set(bom['clave'])),
    )

    consumo = pd.merge(df_fd, bom, on='clave', how='left')
    diag.filas_totales = len(consumo)
    diag.filas_sin_componente = int(consumo['Cantidad'].isna().sum())

    consumo['Cantidad'] = pd.to_numeric(consumo['Cantidad'], errors='coerce').fillna(0)
    consumo['consumo_proyectado'] = consumo['forecast_talle'] * consumo['Cantidad']

    consumo = consumo.rename(columns={
        'Articulo_x': 'Articulo_Padre',
        'Componente de lista de materia': 'Nombre_Componente_Insumo',
    })
    return consumo, diag
