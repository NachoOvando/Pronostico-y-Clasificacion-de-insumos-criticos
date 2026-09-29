"""Paleta visual del informe (gráficos, tablas del notebook y Excel): única fuente de verdad."""

# ── Colores institucionales del informe ───────────────────────────────────────
AZUL = '#1B3A6B'        # títulos
AZUL_MEDIO = '#2E75B6'  # subtítulos, barras
NARANJA = '#ED7D31'
FONDO = '#FAFAFA'

# ── Criticidad ────────────────────────────────────────────────────────────────
COLOR_CRIT: dict[str, str] = {
    'CRÍTICO': '#C00000',
    'IMPORTANTE': '#ED7D31',
    'SECUNDARIO': '#FFC000',
}

# Color de texto legible sobre cada color de criticidad.
COLOR_TEXTO_CRIT: dict[str, str] = {
    'CRÍTICO': '#FFFFFF',
    'IMPORTANTE': '#FFFFFF',
    'SECUNDARIO': '#000000',
}

# Tonos pastel para pintar filas completas sin tapar el texto (detalle de SKUs).
COLOR_FILA_CRIT: dict[str, str] = {
    'CRÍTICO': '#FADBD8',
    'IMPORTANTE': '#FDEBD0',
    'SECUNDARIO': '#FEF9E7',
}

# ── Unidades de medida (gráfico de dispersión) ────────────────────────────────
UM_MARKER: dict[str, str] = {'G': 'D', 'PAA': 'o', 'UN': 's'}
UM_LABEL: dict[str, str] = {
    'G': 'Gramos (G)',
    'PAA': 'Pares (PAA)',
    'UN': 'Unidades (UN)',
}


def hex_openpyxl(color: str) -> str:
    """openpyxl espera el color sin '#' (p. ej. 'C00000')."""
    return color.lstrip('#').upper()


def estilo_celda_criticidad(valor: str) -> str:
    """Estilo CSS para la columna Criticidad de un `Styler` de pandas."""
    fondo = COLOR_CRIT.get(valor, '#FFFFFF')
    texto = COLOR_TEXTO_CRIT.get(valor, '#000000')
    return f'background-color: {fondo}; color: {texto}; font-weight: bold'


def estilo_fila_criticidad(criticidad: str, n_columnas: int) -> list[str]:
    """Estilo CSS pastel para una fila entera, según su criticidad."""
    fondo = COLOR_FILA_CRIT.get(criticidad, '#FFFFFF')
    return [f'background-color: {fondo}'] * n_columnas
