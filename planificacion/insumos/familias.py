"""Agrupamiento de insumos en familias de compra.

Los insumos que solo varían por talle se consolidan en una **familia de compra**,
que es la unidad real de decisión del área de Compras.
"""

import re

import pandas as pd

# Reglas explícitas para agrupar familias con nomenclaturas distintas que
# corresponden al mismo artículo de compra. (patrón_regex, familia, UM).
# Se aplican en orden: la primera que matchea gana.
REGLAS_FAMILIA: tuple[tuple[str, str, str], ...] = (
    (r'PUNTERA ACERO',             'PUNTERA ACERO 59 NORMAL',        'PAA'),
    (r'PUNTERA ALUMINIO',          'PUNTERA ALUMINIO 459 NORMAL',    'PAA'),
    (r'PLANT ARM STROB UL',        'PLANTILLA ARM STROB UL',         'PAA'),
    (r'PLANT ARM STROB VORAN',     'PLANTILLA ARM STROB VORAN',      'PAA'),
    (r'PLANT VIS CONF',            'PLANTILLA VIS CONF MA',          'PAA'),
    (r'PLANT VIS PLANA 4mm FUNC',  'PLANTILLA VIS PLANA FUNC MA',    'PAA'),
    (r'PLANT VIS PLANA 4mm VORAN', 'PLANTILLA VIS PLANA VORAN NEGRO', 'PAA'),
    (r'INSERTO B/PU UL DEL',       'INSERTO B/PU UL DELANTERO',      'PAA'),
    (r'INSERTO B/PU UL TRA',       'INSERTO B/PU UL TRASERO',        'PAA'),
    # Mismo proveedor (PAPEL PACK ENVASES SRL), mismo Lead Time, mismos artículos:
    # es una sola decisión de compra con dos estilos de caja, no dos familias distintas.
    (r'CAJA EMPAQUE',              'CAJA EMPAQUE (BOTA/BOTÍN)',      'UN'),
)

_RE_TALLE_COMA = re.compile(r',?\s*T\.\s*\d+\s*$')                   # "..., T. 42"
_RE_TALLE_SUFIJO = re.compile(r'\s+[Tt][\.] ?\s*\d+[\.,]?\d*\s*$')   # "... T.42" / "... T.45,5"


def nombre_base(nombre: str) -> str:
    """Quita el sufijo de talle al final del nombre del componente.

    Solo reconoce el talle escrito con punto (`T. 42`, `T.45,5`). Los nombres que
    lo escriben pegado (`... H450/3 T45,5`) quedan intactos y se consolidan más
    adelante por las reglas de `REGLAS_FAMILIA`, que es lo que agrupa la familia
    de compra real.
    """
    limpio = str(nombre).strip()
    limpio = _RE_TALLE_COMA.sub('', limpio)
    limpio = _RE_TALLE_SUFIJO.sub('', limpio)
    return limpio.strip()


def asignar_familia(base: str, um: str) -> tuple[str, str]:
    """Devuelve `(familia, UM)`. Sin regla que matchee, el nombre base pasa intacto."""
    for patron, familia, um_familia in REGLAS_FAMILIA:
        if re.search(patron, base, re.IGNORECASE):
            return familia, um_familia
    return base, um


def asignar_familias_de_compra(consumo: pd.DataFrame) -> pd.DataFrame:
    """Agrega `Nombre_Base`, `Familia` y `UM_familia` al consumo proyectado.

    Las expresiones regulares se evalúan una sola vez por valor distinto y el
    resultado se propaga con un diccionario: aplicarlas fila por fila significaba
    miles de evaluaciones para unas pocas decenas de nombres reales.
    """
    if consumo.empty:
        raise ValueError(
            'El consumo proyectado está vacío (0 filas). El problema está en el cruce '
            'Forecast × BOM, no acá: revisar el valor de `Tipo` usado como pronóstico.'
        )

    df = consumo.copy()
    mapa_base = {n: nombre_base(n) for n in df['Nombre_Componente_Insumo'].unique()}
    df['Nombre_Base'] = df['Nombre_Componente_Insumo'].map(mapa_base)

    claves = list(zip(df['Nombre_Base'], df['UM']))
    mapa_familia = {clave: asignar_familia(*clave) for clave in set(claves)}
    df['Familia'] = [mapa_familia[c][0] for c in claves]
    df['UM_familia'] = [mapa_familia[c][1] for c in claves]
    return df
