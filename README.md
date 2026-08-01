# Pronóstico y Clasificación de Insumos Críticos

Sistema de planificación de demanda e inventario desarrollado para el Proyecto Final de
Ingeniería Industrial (Maincal S.A.). Pronostica la demanda mensual por artículo, la traduce
en requerimientos de insumos, clasifica esos insumos por criticidad y define políticas de
inventario para los críticos.

## Pipeline

1. **Pronóstico de demanda** — Prophet, con búsqueda de hiperparámetros (Random Search +
   cross-validation) por artículo.
2. **Explosión a requerimientos de insumos** — cruza el pronóstico con la lista de materiales
   (BOM) y la curva de distribución de talles.
3. **Clasificación por criticidad** — K-Means sobre tres variables normalizadas (volumen,
   alcance productivo, lead time), etiquetado por score AHP.
4. **Políticas de inventario** — Stock de Seguridad, Punto de Pedido y Stock Máximo, con
   revisión continua (s,Q) o periódica (R,S) según el insumo.

## Estructura

```
main.py                          orquestador: las 4 etapas del pipeline
planificacion/
├── config.py                    parámetros de cada etapa (dataclasses)
├── io_datos.py                  carga de Excel de entrada/salida
├── estilos.py                   paleta de criticidad compartida
├── forecast/                    pronóstico de ventas (Prophet)
└── insumos/                     explosión de BOM, clustering, políticas
tests/                            54 tests (pytest)
Pronostico_Ventas.ipynb          notebook narrativo — etapa 1
Insumos_Criticos_KMeans.ipynb    notebook narrativo — etapas 2 a 4
```

Los notebooks documentan el razonamiento de cada etapa para el informe; la lógica que
ejecutan vive en el paquete `planificacion/` y está cubierta por tests.

## Uso

```bash
pip install -r requirements.txt   # o instalar manualmente: pandas, prophet, scikit-learn,
                                   # scipy, openpyxl, matplotlib
python main.py
```

Genera `pronostico_ventas.xlsx`, `Insumos_Criticos.xlsx`,
`Politicas_Inventario_Insumos_Criticos.xlsx` y los gráficos de la clasificación.

## Tests

```bash
python -m pytest tests/ -q
```
