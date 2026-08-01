from dataclasses import dataclass

import matplotlib
import pandas as pd

matplotlib.use('Agg')   

from planificacion import io_datos  # Carga de datos y exportación de resultados
from planificacion.config import ForecastConfig, InsumosConfig      #Configuración de forecast e insumos
from planificacion.forecast import datos, modelo, resultados    # Forecast con Prophet
from planificacion.insumos import (bom, clustering, curva_talles, familias, features,
                                   graficos, politicas, reporte_excel)  # Procesamiento de insumos
from planificacion.insumos.clustering import ClasificacionInsumos # Clase para la clasificación de insumos por criticidad

## Proyeccion de 12 meses de ventas de articulos con Prophet

def pronosticar_demanda_mensual_de_articulos(cfg: ForecastConfig) -> None:
    modelo.silenciar_logs_prophet()

    ventas_en_columnas = io_datos.cargar_ventas(cfg.excel_url, cfg.cache_ventas)
    ventas_mensuales = datos.pasar_a_formato_largo(ventas_en_columnas)


    ventas_historicas = ventas_mensuales[['Articulo', 'Fecha', 'Volumen_Ventas']].copy()
    serie_prophet = datos.a_formato_prophet(ventas_mensuales)
    articulos = sorted(serie_prophet['Articulo'].unique())

    busqueda = modelo.buscar_hiperparametros(serie_prophet, articulos, cfg)
    mejores_hiperparametros = {articulo: r.params for articulo, r in busqueda.items()}
    modelos_ajustados = modelo.ajustar_modelos_por_articulo(
        serie_prophet, articulos, mejores_hiperparametros, cfg)

    pronostico, ajuste_historico = resultados.separar_ajuste_historico_y_pronostico(
        modelos_ajustados)
    pronostico_exportado = resultados.exportar_pronostico_a_excel(
        ventas_historicas, ajuste_historico, pronostico, cfg.output_xlsx)


@dataclass(frozen=True)
class RequerimientosDeInsumos:
    ventas_pronosticadas: pd.DataFrame
    lista_de_materiales: pd.DataFrame
    curva_de_talles: pd.DataFrame
    tipo_pronostico: str
    consumo_por_insumo: pd.DataFrame
    familias_de_compra: pd.DataFrame
    total_articulos: int


# Desagregacion del Pronostico y relacion con la BOM para obtener el consumo proyectado por insumo, y asociacion por familia de compra

def explotar_demanda_en_requerimientos_de_insumos(cfg: InsumosConfig) -> RequerimientosDeInsumos:

    ventas_pronosticadas = io_datos.cargar_forecast(cfg.forecast_path)
    lista_de_materiales = io_datos.cargar_bom(cfg.bom_path)
    curva_de_talles = curva_talles.construir_curva_de_talles(cfg)

    tipo_pronostico = bom.detectar_tipo_pronostico(ventas_pronosticadas, cfg.tipo_pronostico)
    consumo_por_insumo, diagnostico = bom.explotar_a_consumo_de_insumos(
        ventas_pronosticadas, lista_de_materiales, curva_de_talles, tipo_pronostico)


    consumo_por_insumo = familias.asignar_familias_de_compra(consumo_por_insumo)
    total_articulos = consumo_por_insumo['Articulo_Padre'].nunique()
    familias_de_compra = features.calcular_variables_de_criticidad(
        features.consolidar_por_familia_de_compra(consumo_por_insumo, cfg), total_articulos)


    return RequerimientosDeInsumos(
        ventas_pronosticadas=ventas_pronosticadas,
        lista_de_materiales=lista_de_materiales,
        curva_de_talles=curva_de_talles,
        tipo_pronostico=tipo_pronostico,
        consumo_por_insumo=consumo_por_insumo,
        familias_de_compra=familias_de_compra,
        total_articulos=total_articulos,
    )

# Clasificacion de insumos por criticidad (K-Means + AHP) y exportacion de resultados a Excel y graficos

def clasificar_insumos_por_criticidad(requerimientos: RequerimientosDeInsumos,
                                      cfg: InsumosConfig) -> ClasificacionInsumos:

    variables_escaladas, escalador = clustering.escalar_variables(
        requerimientos.familias_de_compra)
    clasificacion = clustering.clasificar_por_kmeans(
        requerimientos.familias_de_compra, variables_escaladas, escalador, cfg)

    familias_clasificadas = clustering.tabla_de_familias_clasificadas(clasificacion, cfg)
    skus_por_familia = clustering.tabla_de_skus_por_familia(
        requerimientos.consumo_por_insumo, familias_clasificadas, clasificacion.familias,
        requerimientos.total_articulos, cfg)

    graficos.dispersion(familias_clasificadas, clasificacion, cfg, cfg.grafico_dispersion)
    graficos.coordenadas_paralelas(familias_clasificadas, clasificacion, cfg,
                                   cfg.grafico_paralelas)
    graficos.matriz_decision(familias_clasificadas, cfg, cfg.grafico_heatmap)
    reporte_excel.exportar_informe_de_criticidad(familias_clasificadas, skus_por_familia, cfg)

    return clasificacion

# Definicion de politicas de inventario por insumo critico y exportacion a Excel

def definir_politicas_de_inventario(requerimientos: RequerimientosDeInsumos,
                                    clasificacion: ClasificacionInsumos,
                                    cfg: InsumosConfig) -> None:
    
    politicas_de_inventario = politicas.calcular_politicas_de_inventario(
        requerimientos.ventas_pronosticadas, requerimientos.lista_de_materiales,
        requerimientos.curva_de_talles, requerimientos.tipo_pronostico, cfg,
        familias=clasificacion.familias)
    politicas_de_inventario.to_excel(cfg.politicas_output_path, index=False)



def main() -> None:
    pronosticar_demanda_mensual_de_articulos(ForecastConfig())

    cfg = InsumosConfig()
    requerimientos = explotar_demanda_en_requerimientos_de_insumos(cfg)
    clasificacion = clasificar_insumos_por_criticidad(requerimientos, cfg)
    definir_politicas_de_inventario(requerimientos, clasificacion, cfg)


if __name__ == '__main__':
    main()
