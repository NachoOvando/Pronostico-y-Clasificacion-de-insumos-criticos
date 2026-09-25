"""Secciones D y E — selección de K, resultados del clustering y AHP.

Reusa `main.explotar_demanda_en_requerimientos_de_insumos` (solo lectura:
carga forecast+BOM y calcula, no exporta nada) y las funciones puras de
`planificacion.insumos.clustering`/`features`. No llama a `graficos.*` ni
`reporte_excel.*` (esas sí escriben archivos del repo ya versionados).

D. Selección de K: `clustering.evaluar_cantidad_de_clusters` para k=2..6
   (mismo `random_state`/`n_init` que el pipeline real), más `silhouette_samples`
   para k=3 (no existe en el repo — sklearn nativo, cálculo nuevo) para tener
   silueta por familia además del promedio global ya documentado (0.524).

E. AHP: los pesos usados (`config.py:122-124`) y evidencia de que no hay
   código AHP en el repo (grep del valor 7.22 y de 1.93/3.74). Score AHP por
   familia con `features.score_ahp` real.

También verifica que el universo real de familias es 24 (no 27) y que 17
quedan en "alcance de Compras", y que la clasificación reproduce
`Insumos_Criticos.xlsx` ya existente.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import DIR_ANALISIS, DIR_REPO
from main import explotar_demanda_en_requerimientos_de_insumos
from planificacion.config import InsumosConfig
from planificacion.insumos import clustering
from planificacion.insumos.features import FEATURES, score_ahp

OUT = DIR_ANALISIS / 'outputs' / 'D_E_kmeans_ahp'
OUT.mkdir(parents=True, exist_ok=True)


def seccion_e_ahp_busqueda_7_22() -> list[str]:
    """Grep del valor 7.22 (y sus supuestos componentes 1.93/3.74) en el código y notebooks
    originales del repo (excluye analysis_revision/ para no matchear este mismo script, y
    descarta líneas larguísimas: son blobs base64 de imágenes PNG incrustadas en los .ipynb,
    no texto/código, y pueden contener coincidencias numéricas puramente casuales)."""
    resultados = []
    for patron in ('7.22', '7,22', '1.93', '1,93', '3.74', '3,74'):
        try:
            out = subprocess.run(
                ['grep', '-rn', '--include=*.py', '--include=*.ipynb', '--include=*.md',
                '--exclude-dir=analysis_revision', patron, str(DIR_REPO)],
                capture_output=True, text=True,
            )
            lineas = [l for l in out.stdout.splitlines() if len(l) < 300]
            descartadas = len(out.stdout.splitlines()) - len(lineas)
            texto = '\n'.join(lineas) if lineas else 'SIN COINCIDENCIAS'
            if descartadas:
                texto += f"\n  ({descartadas} línea(s) descartada(s) por ser blobs base64 de imagen)"
            resultados.append(f"grep -rn '{patron}' (excl. analysis_revision/): {texto}")
        except Exception as exc:  # noqa: BLE001
            resultados.append(f"grep '{patron}' falló: {exc}")
    return resultados


def main() -> None:
    cfg = InsumosConfig()
    requerimientos = explotar_demanda_en_requerimientos_de_insumos(cfg)
    familias = requerimientos.familias_de_compra

    n_universo = len(familias)
    print(f"Familias que entran al K-Means (universo completo): {n_universo}")
    assert n_universo == 24, (
        f"Se esperaban 24 familias (planificacion/insumos/features.py: "
        f"consolidar_por_familia_de_compra + calcular_variables_de_criticidad), "
        f"se obtuvieron {n_universo}. Revisar si cambiaron los datos de entrada."
    )

    # ── D. Selección de K ──
    variables_escaladas, scaler = clustering.escalar_variables(familias)
    df_codo_silueta = clustering.evaluar_cantidad_de_clusters(variables_escaladas, cfg)
    df_codo_silueta.to_csv(OUT / 'codo_silueta.csv', index=False)
    print("\n=== Codo/silueta k=2..6 (planificacion/insumos/clustering.py:21-35) ===")
    print(df_codo_silueta.to_string(index=False))
    k_optimo_silueta = int(df_codo_silueta.loc[df_codo_silueta['Silhouette'].idxmax(), 'K'])
    print(f"K con mayor Silhouette: {k_optimo_silueta} "
         f"(vs. K configurado = {cfg.k_clusters}, config.py:116)")

    # ── Clasificación final (k=3, mismo random_state/n_init que el pipeline real) ──
    clasificacion = clustering.clasificar_por_kmeans(familias, variables_escaladas, scaler, cfg)
    print(f"\nSilhouette global k=3 (clustering.py:91): {clasificacion.silhouette:.3f}")

    # Silueta por familia (silhouette_samples no existe en el repo; sklearn nativo)
    valores_por_muestra = silhouette_samples(variables_escaladas, clasificacion.familias['Cluster_raw'])
    df_silueta_familia = clasificacion.familias[['Familia', 'Criticidad']].copy()
    df_silueta_familia['Silhouette_individual'] = valores_por_muestra
    resumen_por_cluster = df_silueta_familia.groupby('Criticidad')['Silhouette_individual'].mean()
    print("\nSilueta media por cluster (silhouette_samples, cálculo nuevo):")
    print(resumen_por_cluster.to_string())
    delta_silueta = abs(valores_por_muestra.mean() - clasificacion.silhouette)
    print(f"Chequeo: mean(silhouette_samples)={valores_por_muestra.mean():.6f} vs "
         f"silhouette_score={clasificacion.silhouette:.6f} (delta={delta_silueta:.2e})")
    df_silueta_familia.to_csv(OUT / 'silueta_por_familia.csv', index=False)

    # ── E. AHP ──
    print(f"\nPesos AHP usados (config.py:122-124): "
         f"Alcance={cfg.w_alcance}, LeadTime={cfg.w_leadtime}, Volumen={cfg.w_volumen} "
         f"(suman {cfg.w_alcance + cfg.w_leadtime + cfg.w_volumen:.3f})")
    print("Búsqueda del valor 7.22 / 1.93 / 3.74 en el repo:")
    lineas_grep = seccion_e_ahp_busqueda_7_22()
    for linea in lineas_grep:
        print(f"  {linea}")
    (OUT / 'busqueda_ahp_7_22.txt').write_text('\n'.join(lineas_grep) + '\n', encoding='utf-8')

    df_score = familias[['Familia'] + FEATURES].copy()
    df_score['Score_AHP'] = score_ahp(familias, cfg)
    df_score['Criticidad'] = clasificacion.familias['Criticidad'].values
    df_score['Es_Compra_Externa'] = clasificacion.familias['Es_Compra_Externa'].values
    df_score = df_score.sort_values('Score_AHP', ascending=False).reset_index(drop=True)
    df_score.to_csv(OUT / 'score_ahp_por_familia.csv', index=False)
    print(f"\n=== Score AHP por familia (features.score_ahp real) ===")
    print(df_score.to_string(index=False))

    # ── Universo 24 -> 17 (alcance de Compras) ──
    familias_alcance = clustering.tabla_de_familias_clasificadas(clasificacion, cfg)
    n_alcance = len(familias_alcance)
    print(f"\nFamilias en alcance de Compras (Es_Compra_Externa, clustering.py:84,95-109): "
         f"{n_alcance} de {n_universo} totales")

    # ── Verificación de reproducibilidad contra Insumos_Criticos.xlsx ya existente ──
    path_excel = cfg.output_path
    df_verif = pd.DataFrame()
    if path_excel.exists():
        # La hoja tiene título (fila 0) y subtítulo (fila 1) antes del encabezado real
        # (fila 2: '#', 'Familia de Compra', ..., 'Criticidad') — confirmado inspeccionando
        # el archivo con pandas antes de escribir este script.
        hoja_familias = pd.read_excel(path_excel, sheet_name='Familias de Compra', header=2)
        hoja_familias = hoja_familias.rename(columns={'Familia de Compra': 'Familia'})
        comunes = set(hoja_familias['Familia']) & set(familias_alcance['Familia'])
        merge = hoja_familias.set_index('Familia').loc[list(comunes), ['Criticidad']].rename(
            columns={'Criticidad': 'Criticidad_excel'})
        merge['Criticidad_recalculada'] = familias_alcance.set_index('Familia').loc[
            merge.index, 'Criticidad']
        merge['coincide'] = merge['Criticidad_excel'] == merge['Criticidad_recalculada']
        df_verif = merge.reset_index()
        n_ok = df_verif['coincide'].sum()
        print(f"\nVerificación vs. {path_excel.name} (ya existente): "
             f"{n_ok}/{len(df_verif)} familias con la misma Criticidad "
             f"({'OK' if n_ok == len(df_verif) else 'DIVERGE'})")
        df_verif.to_csv(OUT / 'verificacion_vs_excel.csv', index=False)
    else:
        print(f"\n{path_excel.name} no existe en el repo: no se puede verificar "
             "contra un resultado previamente exportado.")

    resumen = pd.DataFrame([{
        'n_universo_kmeans': n_universo,
        'n_alcance_compras': n_alcance,
        'k_configurado': cfg.k_clusters,
        'k_optimo_por_silhouette': k_optimo_silueta,
        'silhouette_k3_global': clasificacion.silhouette,
        'silhouette_max_valor': df_codo_silueta['Silhouette'].max(),
    }])
    resumen.to_csv(OUT / 'resumen_D_E.csv', index=False)
    print(f"\nCSVs escritos en {OUT}")


if __name__ == '__main__':
    main()
