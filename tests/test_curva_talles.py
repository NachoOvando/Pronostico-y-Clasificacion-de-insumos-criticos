from dataclasses import replace

import pytest

from planificacion.config import InsumosConfig
from planificacion.insumos.curva_talles import construir_curva_de_talles


def test_las_proporciones_suman_uno():
    df = construir_curva_de_talles(InsumosConfig())

    assert df['proporcion'].sum() == pytest.approx(1.0)


def test_cubre_todos_los_talles_del_rango_inclusive():
    cfg = InsumosConfig()

    df = construir_curva_de_talles(cfg)

    assert df['talle'].min() == cfg.talle_min
    assert df['talle'].max() == cfg.talle_max
    assert len(df) == cfg.talle_max - cfg.talle_min + 1


def test_el_pico_esta_en_el_talle_medio():
    cfg = InsumosConfig()

    df = construir_curva_de_talles(cfg)

    assert df.loc[df['proporcion'].idxmax(), 'talle'] == cfg.talle_media


def test_es_simetrica_alrededor_del_talle_medio():
    cfg = replace(InsumosConfig(), talle_min=40, talle_max=44, talle_media=42)

    proporciones = construir_curva_de_talles(cfg)['proporcion'].tolist()

    assert proporciones[0] == pytest.approx(proporciones[4])
    assert proporciones[1] == pytest.approx(proporciones[3])


def test_un_desvio_mas_grande_aplana_la_curva():
    cfg = InsumosConfig()

    angosta = construir_curva_de_talles(cfg)['proporcion'].max()
    ancha = construir_curva_de_talles(replace(cfg, talle_desvio=6.0))['proporcion'].max()

    assert ancha < angosta
