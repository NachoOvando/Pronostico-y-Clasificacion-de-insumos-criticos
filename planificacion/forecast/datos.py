"""Preparación del histórico de ventas: wide → long, fechas y validaciones."""

from dataclasses import dataclass

import pandas as pd

# Abreviaturas de mes en español -> inglés, para que `pd.to_datetime` pueda
# parsear 'ene-23' con el formato '%b-%y' independientemente del locale.
MESES_ES_EN: dict[str, str] = {
    'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr',
    'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'ago': 'Aug',
    'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec',
}


def pasar_a_formato_largo(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Convierte el Excel *wide* a `Articulo | Fecha | Volumen_Ventas`, ordenado."""
    df_long = df_wide.melt(id_vars=['Articulo'],
                           var_name='Fecha_str',
                           value_name='Volumen_Ventas')

    fecha_en = (df_long['Fecha_str'].str[:3].map(MESES_ES_EN)
                + df_long['Fecha_str'].str[3:])
    if fecha_en.isna().any():
        invalidas = df_long.loc[fecha_en.isna(), 'Fecha_str'].unique()
        raise ValueError(
            f"No se pudieron interpretar estas columnas de mes: {list(invalidas)}. "
            f"Se esperan abreviaturas en español ({', '.join(MESES_ES_EN)})."
        )
    df_long['Fecha'] = pd.to_datetime(fecha_en, format='%b-%y')

    return (df_long
            .drop(columns=['Fecha_str'])
            .sort_values(by=['Articulo', 'Fecha'])
            .reset_index(drop=True))


@dataclass(frozen=True)
class Validacion:
    """Resultado del control de calidad del histórico."""

    nulos: int
    negativos: pd.DataFrame

    @property
    def ok(self) -> bool:
        return self.nulos == 0 and self.negativos.empty

    def __str__(self) -> str:
        lineas = [f"Valores nulos en 'Volumen_Ventas': {self.nulos}"]
        if self.negativos.empty:
            lineas.append('Validación OK: no se encontraron volúmenes de ventas negativos.')
        else:
            lineas.append(f'Advertencia: {len(self.negativos)} volúmenes negativos encontrados.')
        return '\n'.join(lineas)


def validar_calidad_de_la_serie(df_long: pd.DataFrame) -> Validacion:
    """Cuenta nulos y detecta ventas negativas (imposibles por definición)."""
    return Validacion(
        nulos=int(df_long['Volumen_Ventas'].isnull().sum()),
        negativos=df_long[df_long['Volumen_Ventas'] < 0],
    )


def interpolar_nulos(df_long: pd.DataFrame) -> pd.DataFrame:
    """Rellena huecos por interpolación lineal dentro de cada artículo.

    Estrategia recomendada para series temporales. Hoy el dataset no tiene
    nulos, pero queda disponible si el histórico se amplía.
    """
    df = df_long.copy()
    df['Volumen_Ventas'] = (df.groupby('Articulo')['Volumen_Ventas']
                              .transform(lambda x: x.interpolate(method='linear')))
    return df


def a_formato_prophet(df_long: pd.DataFrame) -> pd.DataFrame:
    """Renombra al contrato de Prophet: `ds` (fecha) e `y` (valor)."""
    return (df_long
            .rename(columns={'Fecha': 'ds', 'Volumen_Ventas': 'y'})
            [['Articulo', 'ds', 'y']]
            .copy())


def serie_articulo(df_prophet: pd.DataFrame, articulo: str) -> pd.DataFrame:
    """Devuelve la serie `ds`/`y` ordenada de un artículo."""
    return (df_prophet[df_prophet['Articulo'] == articulo][['ds', 'y']]
            .sort_values('ds')
            .reset_index(drop=True))
