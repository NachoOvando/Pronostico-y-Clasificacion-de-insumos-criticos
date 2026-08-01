"""Parámetros de los dos flujos del pipeline.

Toda la configuración vive acá: instanciar `ForecastConfig()` o `InsumosConfig()`
con argumentos distintos alcanza para re-ejecutar el análisis con otros criterios,
sin tocar el código de cálculo.
"""

from dataclasses import dataclass, field
from pathlib import Path

# Raíz del proyecto (carpeta `Planificacion`), para que las rutas relativas no
# dependan del directorio desde el que se abra el notebook.
DIR_BASE = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ForecastConfig:
    """Configuración del pronóstico de ventas con Prophet."""

    # ── Origen de datos ───────────────────────────────────────────────────────
    # Mismo archivo de Google Drive, en formato de descarga directa
    # (/export?format=xlsx): el link de vista (/edit?usp=drive_link) no es
    # legible por pandas.
    excel_url: str = ('https://docs.google.com/spreadsheets/d/'
                      '16t7i-CEsjsIBaIKk5rSEk43lfaf2rBU9/export?format=xlsx')
    # Copia local del origen: permite re-correr el pipeline sin internet.
    cache_ventas: Path = DIR_BASE / 'ventas_historicas_cache.xlsx'

    # ── Salida (la consume Insumos_Criticos_KMeans como FORECAST_PATH) ────────
    output_xlsx: Path = DIR_BASE / 'pronostico_ventas.xlsx'

    # ── Reproducibilidad ──────────────────────────────────────────────────────
    random_seed: int = 42

    # ── Random Search + cross-validation (mensual, 36 meses por artículo) ─────
    n_iter: int = 20              # nº de combinaciones aleatorias a muestrear
    cv_initial: str = '730 days'  # ~24 meses de entrenamiento inicial
    cv_period: str = '90 days'    # ~3 meses entre cortes de validación
    cv_horizon: str = '180 days'  # ~6 meses de horizonte evaluado por corte
    metrica: str = 'rmse'         # métrica objetivo de la búsqueda
    # Paralelismo de `prophet.diagnostics.cross_validation`. 'threads' no altera
    # los resultados (cmdstanpy ajusta en subproceso); None fuerza modo serie.
    cv_parallel: str | None = 'threads'
    # Cache en disco de la búsqueda: si los datos, la semilla y la grilla no
    # cambiaron, re-correr el notebook no repite los ~300 ajustes de Prophet.
    cache_hiperparametros: Path = DIR_BASE / 'hiperparametros_prophet.json'

    # ── Evaluación y pronóstico ───────────────────────────────────────────────
    test_periods: int = 12        # meses de hold-out out-of-sample (1 ciclo)
    forecast_horizon: int = 12    # meses a pronosticar

    def resumen(self) -> str:
        return (f"N_ITER={self.n_iter} | CV: {self.cv_initial}/{self.cv_period}/"
                f"{self.cv_horizon} | métrica={self.metrica}\n"
                f"Hold-out: {self.test_periods} meses | "
                f"Horizonte de pronóstico: {self.forecast_horizon} meses")


@dataclass(frozen=True)
class PoliticaInsumo:
    """Insumo crítico con política de inventario diferenciada (Sección 9).

    `continua=True`  -> revisión continua (s, Q)
    `continua=False` -> revisión periódica (R, S) con R = `revision_meses`
    """

    familia: str
    patron: str
    continua: bool
    revision_meses: float = 1.0

    @property
    def nombre_politica(self) -> str:
        return ('Revisión continua (s,Q)' if self.continua
                else 'Revisión periódica (R,S)')

    @property
    def R_meses(self) -> float:
        """Período de revisión efectivo: 0 en revisión continua."""
        return 0.0 if self.continua else self.revision_meses


POLITICAS_POR_DEFECTO: tuple[PoliticaInsumo, ...] = (
    PoliticaInsumo('CONJ SISTEMA PU', r'CONJ SISTEMA PU', continua=True),
    PoliticaInsumo('PUNTERA ACERO 59 NORMAL', r'PUNTERA ACERO', continua=False),
    PoliticaInsumo('CAJA EMPAQUE (BOTA/BOTÍN)', r'CAJA EMPAQUE', continua=False),
)


@dataclass(frozen=True)
class InsumosConfig:
    """Configuración de la clasificación de insumos críticos con K-Means."""

    # ── Rutas ─────────────────────────────────────────────────────────────────
    forecast_path: Path = DIR_BASE / 'pronostico_ventas.xlsx'
    bom_path: Path = DIR_BASE / 'BOM_Zapato_Terminado.xlsx'
    output_path: Path = DIR_BASE / 'Insumos_Criticos.xlsx'
    politicas_output_path: Path = DIR_BASE / 'Politicas_Inventario_Insumos_Criticos.xlsx'

    # ── Curva de talles (distribución normal) ─────────────────────────────────
    talle_min: int = 34
    talle_max: int = 50       # inclusive
    talle_media: float = 42
    talle_desvio: float = 2.5

    # ── Filtro de período pronosticado ────────────────────────────────────────
    # None -> autodetectar el valor de 'Tipo' que representa el pronóstico.
    tipo_pronostico: str | None = None

    # ── Lead Time (tercera variable de clasificación) ─────────────────────────
    leadtime_col: str = 'Lead Time'   # nombre exacto de la columna en el BOM
    leadtime_unidad: str = 'días'     # informativo (títulos de gráficos/tablas)
    leadtime_agg: str = 'max'         # talle -> familia: 'max' (conservador) o 'mean'

    # ── K-Means ───────────────────────────────────────────────────────────────
    k_clusters: int = 3               # 3 = CRÍTICO / IMPORTANTE / SECUNDARIO
    k_range: range = field(default_factory=lambda: range(2, 7))
    random_state: int = 42
    n_init: int = 20

    # ── Pesos AHP del score de centroide (deben sumar 1) ──────────────────────
    w_alcance: float = 0.604   # afecta a varios artículos a la vez
    w_leadtime: float = 0.312  # riesgo temporal de reposición
    w_volumen: float = 0.084   # manejable con stock de seguridad

    # ── Etiquetas de criticidad (de mayor a menor score) ──────────────────────
    etiquetas: tuple[str, ...] = ('CRÍTICO', 'IMPORTANTE', 'SECUNDARIO')

    # ── Políticas de inventario (Sección 9) ───────────────────────────────────
    z_servicio: float = 2.05   # ≈98% de nivel de servicio
    politicas: tuple[PoliticaInsumo, ...] = POLITICAS_POR_DEFECTO

    # ── Gráficos (PNG generados en la Sección 7, incrustados en el Excel) ─────
    grafico_dispersion: Path = DIR_BASE / 'grafico_insumos_criticos.png'
    grafico_paralelas: Path = DIR_BASE / 'grafico_coordenadas_paralelas.png'
    grafico_heatmap: Path = DIR_BASE / 'grafico_matriz_decision.png'

    def __post_init__(self) -> None:
        suma = self.w_volumen + self.w_alcance + self.w_leadtime
        if abs(suma - 1.0) > 1e-6:
            raise ValueError(
                f"Los pesos AHP deben sumar 1.0, suman {suma:.4f} "
                f"(volumen={self.w_volumen}, alcance={self.w_alcance}, "
                f"leadtime={self.w_leadtime})."
            )
        if len(self.etiquetas) != self.k_clusters:
            raise ValueError(
                f"Hacen falta {self.k_clusters} etiquetas para K={self.k_clusters}, "
                f"hay {len(self.etiquetas)}: {self.etiquetas}."
            )

    @property
    def pesos(self) -> dict[str, float]:
        """Pesos AHP indexados por la columna de feature que ponderan."""
        return {
            'Vol_norm': self.w_volumen,
            'Alcance_pct': self.w_alcance,
            'LeadTime_norm': self.w_leadtime,
        }

    def resumen(self) -> str:
        return (f"K = {self.k_clusters}  |  Talles: {self.talle_min}–{self.talle_max}  |  "
                f"Archivo salida: {self.output_path.name}\n"
                f"Pesos score → Vol={self.w_volumen:.3f}  "
                f"Alcance={self.w_alcance:.3f}  LeadTime={self.w_leadtime:.3f}")
