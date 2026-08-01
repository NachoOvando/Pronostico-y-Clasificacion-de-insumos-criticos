from dataclasses import replace

import pandas as pd
import pytest

from planificacion.config import InsumosConfig
from planificacion.insumos import features


def _familias(consumos, ums, n_articulos, lead_times):
    return pd.DataFrame({
        'Familia': [f'F{i}' for i in range(len(consumos))],
        'UM': ums,
        'Consumo_Total': consumos,
        'N_Articulos': n_articulos,
        'Lead_Time_dias': lead_times,
    })


def test_vol_norm_se_normaliza_dentro_de_cada_um():
    # El máximo de G (1000) no debe arrastrar la escala de PAA (máximo 50).
    df = _familias([1000.0, 500.0, 50.0, 10.0],
                   ['G', 'G', 'PAA', 'PAA'], [3, 3, 3, 3], [10, 10, 10, 10])

    resultado = features.calcular_variables_de_criticidad(df, total_articulos=3)

    assert resultado['Vol_norm'].tolist() == pytest.approx([100.0, 50.0, 100.0, 20.0])


def test_alcance_pct_es_el_porcentaje_de_articulos_padre():
    df = _familias([1.0, 1.0], ['G', 'G'], [3, 1], [10, 10])

    resultado = features.calcular_variables_de_criticidad(df, total_articulos=3)

    assert resultado['Alcance_pct'].tolist() == pytest.approx([100.0, 100 / 3])


def test_leadtime_norm_es_min_max_sobre_todas_las_familias():
    df = _familias([1.0, 1.0, 1.0], ['G'] * 3, [1, 1, 1], [10, 30, 50])

    resultado = features.calcular_variables_de_criticidad(df, total_articulos=1)

    assert resultado['LeadTime_norm'].tolist() == pytest.approx([0.0, 50.0, 100.0])


def test_leadtime_norm_es_cero_si_todas_comparten_lead_time():
    # Sin varianza la variable no aporta información: no debe dividir por cero.
    df = _familias([1.0, 2.0], ['G', 'G'], [1, 1], [30, 30])

    resultado = features.calcular_variables_de_criticidad(df, total_articulos=1)

    assert resultado['LeadTime_norm'].tolist() == [0.0, 0.0]


def test_calcular_features_no_muta_el_dataframe_recibido():
    df = _familias([1.0], ['G'], [1], [30])

    features.calcular_variables_de_criticidad(df, total_articulos=1)

    assert 'Vol_norm' not in df.columns


def test_score_ahp_pondera_las_tres_features():
    cfg = replace(InsumosConfig(), w_volumen=0.1, w_alcance=0.6, w_leadtime=0.3)
    df = pd.DataFrame({'Vol_norm': [100.0], 'Alcance_pct': [50.0], 'LeadTime_norm': [0.0]})

    assert features.score_ahp(df, cfg).tolist() == pytest.approx([100 * 0.1 + 50 * 0.6])


def test_score_ahp_acepta_matriz_en_el_orden_de_features():
    import numpy as np

    cfg = replace(InsumosConfig(), w_volumen=0.1, w_alcance=0.6, w_leadtime=0.3)
    centroides = np.array([[100.0, 50.0, 0.0], [0.0, 0.0, 100.0]])

    assert features.score_ahp(centroides, cfg) == pytest.approx([40.0, 30.0])


def test_los_pesos_ahp_deben_sumar_uno():
    with pytest.raises(ValueError, match='deben sumar 1.0'):
        replace(InsumosConfig(), w_volumen=0.5, w_alcance=0.5, w_leadtime=0.5)


def test_hacen_falta_tantas_etiquetas_como_clusters():
    with pytest.raises(ValueError, match='etiquetas'):
        replace(InsumosConfig(), k_clusters=4)


def test_agregar_por_familia_descarta_las_familias_sin_consumo():
    cfg = InsumosConfig()
    consumo = pd.DataFrame({
        'Familia': ['A', 'A', 'B'],
        'UM_familia': ['G', 'G', 'G'],
        'consumo_proyectado': [10.0, 5.0, 0.0],
        'Articulo_Padre': ['X', 'Y', 'X'],
        'Nombre_Base': ['a1', 'a2', 'b1'],
        cfg.leadtime_col: [30, 45, 10],
        'Proveedor': [None, 'PROV', 'PROV'],
        'Precio': [1.0, 2.0, 3.0],
    })

    resultado = features.consolidar_por_familia_de_compra(consumo, cfg)

    assert resultado['Familia'].tolist() == ['A']
    assert resultado['Consumo_Total'].tolist() == [15.0]
    assert resultado['N_Articulos'].tolist() == [2]
    # leadtime_agg = 'max': criterio conservador al consolidar talles.
    assert resultado['Lead_Time_dias'].tolist() == [45]
    # 'first' ignora los nulos: la familia tiene proveedor externo.
    assert resultado['Proveedor'].tolist() == ['PROV']
