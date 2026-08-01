"""Lógica del pipeline de planificación de Maincal S.A.

El paquete concentra el cálculo de los dos notebooks de la carpeta, que quedan
reducidos a narrativa del informe + orquestación:

    Pronostico_Ventas.ipynb        -> planificacion.forecast
    Insumos_Criticos_KMeans.ipynb  -> planificacion.insumos

Los dos flujos no se importan entre sí: el contrato entre ambos es el archivo
`pronostico_ventas.xlsx` (columnas `Articulo | Fecha | Ventas | Tipo`).
"""

from planificacion.config import ForecastConfig, InsumosConfig

__all__ = ['ForecastConfig', 'InsumosConfig']
