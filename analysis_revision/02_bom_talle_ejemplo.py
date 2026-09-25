"""Secciones B y C — curva de talles, explosión BOM y ejemplo numérico.

Usa `curva_talles.construir_curva_de_talles` y `bom.explotar_a_consumo_de_insumos`
reales, sobre los artefactos ya existentes en el repo (`pronostico_ventas.xlsx`,
`BOM_Zapato_Terminado.xlsx`) leídos con `io_datos.cargar_forecast`/`cargar_bom`
(ambas de solo lectura). No se llama nada que escriba archivos.

Sección C: busca en el repo cualquier dataset de ventas históricas por talle
(necesario para comparar la curva normal supuesta contra una distribución
empírica y hacer el Q-Q plot pedido). Si no aparece, documenta la búsqueda y
declara el punto NO DETERMINADO — no se inventa el gráfico.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import DIR_ANALISIS, DIR_REPO
from planificacion import io_datos
from planificacion.config import InsumosConfig
from planificacion.insumos import bom, curva_talles

OUT_B = DIR_ANALISIS / 'outputs' / 'B_bom_talles'
OUT_C = DIR_ANALISIS / 'outputs' / 'C_curva_talles'
OUT_B.mkdir(parents=True, exist_ok=True)
OUT_C.mkdir(parents=True, exist_ok=True)

ARTICULO_EJEMPLO = 'CRONOS-N04'
TALLE_EJEMPLO = 42
FECHA_EJEMPLO = '2026-09-01'


def seccion_b(cfg: InsumosConfig) -> None:
    df_forecast = io_datos.cargar_forecast(cfg.forecast_path)
    df_bom = io_datos.cargar_bom(cfg.bom_path)
    df_curva = curva_talles.construir_curva_de_talles(cfg)
    df_curva.to_csv(OUT_B / 'curva_talles.csv', index=False)

    tipo_pron = bom.detectar_tipo_pronostico(df_forecast, cfg.tipo_pronostico)
    consumo, diag = bom.explotar_a_consumo_de_insumos(df_forecast, df_bom, df_curva, tipo_pron)
    print("Diagnóstico del cruce Forecast x BOM:")
    print(diag)

    fila_talle42 = df_curva[df_curva['talle'] == TALLE_EJEMPLO].iloc[0]
    print(f"\nProporción talle {TALLE_EJEMPLO} (media={cfg.talle_media}, "
          f"desvio={cfg.talle_desvio}): {fila_talle42['proporcion']:.6f}")

    talle_extremo = int(df_curva['talle'].min())  # talle mínimo del rango, 34
    fila_extremo = df_curva[df_curva['talle'] == talle_extremo].iloc[0]
    print(f"Proporción talle extremo {talle_extremo}: {fila_extremo['proporcion']:.6f}")

    ejemplo = consumo[
        (consumo['Articulo_Padre'] == ARTICULO_EJEMPLO)
        & (consumo['talle'] == TALLE_EJEMPLO)
        & (consumo['Fecha'] == FECHA_EJEMPLO)
    ].copy()
    if ejemplo.empty:
        raise ValueError(f"No hay filas de consumo para {ARTICULO_EJEMPLO}/T{TALLE_EJEMPLO}/"
                         f"{FECHA_EJEMPLO}. Revisar fechas disponibles en el forecast.")

    forecast_articulo = ejemplo['Ventas'].iloc[0]
    forecast_talle = ejemplo['forecast_talle'].iloc[0]
    print(f"\n=== Ejemplo: {ARTICULO_EJEMPLO}, talle {TALLE_EJEMPLO}, {FECHA_EJEMPLO} ===")
    print(f"Ventas pronosticadas (artículo, todos los talles): {forecast_articulo:.4f}")
    print(f"Proporción talle {TALLE_EJEMPLO}: {fila_talle42['proporcion']:.6f}")
    print(f"forecast_talle = round(Ventas * proporcion, 2) = {forecast_talle}")

    cols = ['Componente', 'Nombre_Componente_Insumo', 'Cantidad', 'UM',
           'forecast_talle', 'consumo_proyectado']
    print("\nComponentes BOM y consumo proyectado:")
    print(ejemplo[cols].to_string(index=False))

    # Ejemplo desarrollado para talle extremo, mismo mes/artículo (para completar el punto C.3)
    ejemplo_extremo = consumo[
        (consumo['Articulo_Padre'] == ARTICULO_EJEMPLO)
        & (consumo['talle'] == talle_extremo)
        & (consumo['Fecha'] == FECHA_EJEMPLO)
    ].copy()

    df_curva.to_csv(OUT_B / 'curva_talles.csv', index=False)
    ejemplo[cols + ['Articulo_Padre', 'talle', 'Fecha', 'Ventas']].to_csv(
        OUT_B / 'ejemplo_cronos_n04_talle42.csv', index=False)
    if not ejemplo_extremo.empty:
        ejemplo_extremo[cols + ['Articulo_Padre', 'talle', 'Fecha', 'Ventas']].to_csv(
            OUT_B / 'ejemplo_cronos_n04_talle_extremo.csv', index=False)

    resumen = pd.DataFrame([{
        'Articulo': ARTICULO_EJEMPLO, 'Talle': TALLE_EJEMPLO, 'Fecha': FECHA_EJEMPLO,
        'Ventas_pronosticadas_articulo': forecast_articulo,
        'Proporcion_talle': fila_talle42['proporcion'],
        'Forecast_talle': forecast_talle,
        'Talle_extremo': talle_extremo,
        'Proporcion_talle_extremo': fila_extremo['proporcion'],
    }])
    resumen.to_csv(OUT_B / 'resumen_ejemplo.csv', index=False)
    print(f"\nCSVs escritos en {OUT_B}")


def seccion_c() -> None:
    """Busca en el repo cualquier dataset de ventas por talle (no BOM)."""
    candidatos = list(DIR_REPO.glob('*.xlsx')) + list(DIR_REPO.glob('*.csv'))
    hallazgos = []
    for archivo in candidatos:
        if archivo.name in ('BOM_Zapato_Terminado.xlsx',):
            continue  # ya se sabe que tiene columna Talle, pero es cantidad por unidad, no ventas
        try:
            if archivo.suffix == '.xlsx':
                cols = pd.read_excel(archivo, nrows=0).columns.tolist()
            else:
                cols = pd.read_csv(archivo, nrows=0).columns.tolist()
        except Exception as exc:  # noqa: BLE001
            hallazgos.append((archivo.name, f'no se pudo leer: {exc}'))
            continue
        tiene_talle = any('talle' in str(c).lower() for c in cols)
        tiene_venta = any(('venta' in str(c).lower()) or ('volumen' in str(c).lower())
                          for c in cols)
        hallazgos.append((archivo.name, f"columnas={cols}, tiene_talle={tiene_talle}, "
                                        f"tiene_venta_o_volumen={tiene_venta}"))

    texto = ["# Sección C — Comparación curva normal vs. datos empíricos de venta por talle\n",
            "**NO DETERMINADO.**\n",
            "No existe en el repositorio ningún dataset de ventas históricas desagregado "
            "por talle. Búsqueda realizada por `analysis_revision/02_bom_talle_ejemplo.py` "
            "(función `seccion_c`): se listaron todos los `.xlsx`/`.csv` de la raíz del "
            "repo y se inspeccionaron sus columnas buscando 'talle' + 'venta'/'volumen' "
            "simultáneamente. Resultado:\n"]
    for nombre, detalle in hallazgos:
        texto.append(f"- `{nombre}`: {detalle}\n")
    texto.append(
        "\nEl único archivo con columna `Talle` es `BOM_Zapato_Terminado.xlsx` "
        "(planificacion/insumos/bom.py), pero esa columna es la lista de materiales "
        "(cantidad de insumo por unidad de producto terminado y talle), no un registro "
        "de unidades vendidas por talle.\n\n"
        "**Qué haría falta para responder este punto:** un dataset de ventas reales "
        "con apertura por talle (por ejemplo, `Articulo | Fecha | Talle | Unidades_Vendidas`), "
        "que hoy no se releva ni se registra en ningún punto del pipeline. Con ese dato se "
        "podría comparar la proporción empírica por talle contra la curva normal "
        "(`talle_media=42, talle_desvio=2.5`, `planificacion/config.py:100-104`) con una "
        "prueba χ² de bondad de ajuste y un gráfico Q-Q. Sin él, no se genera el gráfico "
        "(no se inventan datos ni distribuciones).\n"
    )
    (OUT_C / 'NO_DETERMINADO.md').write_text(''.join(texto), encoding='utf-8')
    print(f"Sección C: NO DETERMINADO, ver {OUT_C / 'NO_DETERMINADO.md'}")


if __name__ == '__main__':
    cfg = InsumosConfig()
    seccion_b(cfg)
    print()
    seccion_c()
