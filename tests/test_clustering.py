from dataclasses import replace

import pandas as pd
import pytest

from planificacion.config import InsumosConfig
from planificacion.insumos import clustering
from planificacion.insumos.features import FEATURES

CFG = InsumosConfig()


def _df_fam():
    """Seis familias en tres grupos separados: crítico, intermedio y secundario."""
    filas = [
        # Familia, Vol_norm, Alcance_pct, LeadTime_norm, Proveedor
        ('ALTA_1', 95.0, 100.0, 90.0, 'PROV A'),
        ('ALTA_2', 90.0, 95.0, 88.0, 'PROV A'),
        ('MEDIA_1', 40.0, 60.0, 50.0, 'PROV B'),
        ('MEDIA_2', 38.0, 58.0, 48.0, 'PROV B'),
        ('BAJA_1', 5.0, 10.0, 2.0, 'PROV C'),
        ('BAJA_2', 3.0, 8.0, 1.0, None),          # producción interna
    ]
    df = pd.DataFrame(filas, columns=['Familia', *FEATURES, 'Proveedor'])
    df['UM'] = 'UN'
    df['Consumo_Total'] = [100.0, 90.0, 40.0, 38.0, 5.0, 3.0]
    df['Lead_Time_dias'] = [90, 88, 50, 48, 2, 1]
    df['N_SKUs_hijo'] = 1
    df['N_Articulos'] = 3
    return df


def _entrenar():
    familias_df = _df_fam()
    variables_escaladas, scaler = clustering.escalar_variables(familias_df)
    return familias_df, clustering.clasificar_por_kmeans(familias_df, variables_escaladas, scaler, CFG)


def test_el_grupo_de_mayor_score_ahp_queda_critico():
    _, resultado = _entrenar()

    criticidad = dict(zip(resultado.familias['Familia'], resultado.familias['Criticidad']))

    assert criticidad['ALTA_1'] == criticidad['ALTA_2'] == 'CRÍTICO'
    assert criticidad['BAJA_1'] == 'SECUNDARIO'
    assert criticidad['MEDIA_1'] == criticidad['MEDIA_2'] == 'IMPORTANTE'


def test_usa_las_tres_etiquetas_configuradas():
    _, resultado = _entrenar()

    assert set(resultado.mapa_etiquetas.values()) == set(CFG.etiquetas)
    assert 0 < resultado.silhouette <= 1


def test_el_centroide_critico_domina_al_secundario_en_las_tres_variables():
    _, resultado = _entrenar()

    critico = resultado.centroide('CRÍTICO')
    secundario = resultado.centroide('SECUNDARIO')

    assert (critico > secundario).all()


def test_las_familias_sin_proveedor_quedan_fuera_del_alcance_de_compras():
    _, resultado = _entrenar()

    df_resultado = clustering.tabla_de_familias_clasificadas(resultado, CFG)

    # BAJA_2 no tiene proveedor: participa del clustering pero no del resultado.
    assert 'BAJA_2' not in df_resultado['Familia'].tolist()
    assert len(df_resultado) == 5
    assert resultado.familias['Es_Compra_Externa'].sum() == 5


def test_la_tabla_ordena_por_criticidad_y_arranca_en_uno():
    _, resultado = _entrenar()

    df_resultado = clustering.tabla_de_familias_clasificadas(resultado, CFG)

    assert df_resultado.index[0] == 1
    assert df_resultado['Criticidad'].tolist() == [
        'CRÍTICO', 'CRÍTICO', 'IMPORTANTE', 'IMPORTANTE', 'SECUNDARIO']
    # Dentro de cada nivel, mayor volumen primero.
    assert df_resultado['Vol_norm'].tolist()[:2] == [95.0, 90.0]


def test_critico_final_excluye_al_critico_de_reposicion_rapida():
    _, resultado = _entrenar()
    # ALTA_2 sigue en el cluster CRÍTICO, pero ahora se repone en 3 días:
    # queda por debajo de la mediana del lead time (25,5 días).
    resultado.familias.loc[resultado.familias['Familia'] == 'ALTA_2', 'Lead_Time_dias'] = 3

    finales = clustering.insumos_criticos_finales(resultado, CFG)

    assert finales['Familia'].tolist() == ['ALTA_1']


def test_critico_final_excluye_la_produccion_interna():
    _, resultado = _entrenar()
    resultado.familias.loc[resultado.familias['Familia'] == 'ALTA_2', 'Proveedor'] = None
    resultado.familias['Es_Compra_Externa'] = resultado.familias['Proveedor'].notna()

    finales = clustering.insumos_criticos_finales(resultado, CFG)

    assert finales['Familia'].tolist() == ['ALTA_1']


def test_evaluar_k_recorre_todo_el_rango_configurado():
    # El silhouette necesita al menos un cluster menos que observaciones: con
    # 6 familias de prueba el rango llega hasta K=4.
    cfg = replace(CFG, k_range=range(2, 5))
    variables_escaladas, _ = clustering.escalar_variables(_df_fam())

    metricas_por_k = clustering.evaluar_cantidad_de_clusters(variables_escaladas, cfg)

    assert metricas_por_k['K'].tolist() == [2, 3, 4]
    assert metricas_por_k['Inercia'].is_monotonic_decreasing


def test_escalar_lleva_las_features_al_rango_unitario():
    variables_escaladas, _ = clustering.escalar_variables(_df_fam())

    assert variables_escaladas.min() == pytest.approx(0.0)
    assert variables_escaladas.max() == pytest.approx(1.0)
    assert variables_escaladas.shape == (6, len(FEATURES))
