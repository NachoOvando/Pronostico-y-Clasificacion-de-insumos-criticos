from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from planificacion.config import InsumosConfig, PoliticaInsumo
from planificacion.insumos import politicas
from planificacion.insumos.curva_talles import construir_curva_de_talles

# Un solo talle -> proporción 1.0: aísla la fórmula de la curva de talles.
CFG_BASE = replace(InsumosConfig(), talle_min=42, talle_max=42)

# Residuos histórico - pronóstico: +10 y +20  ->  desvío muestral = 7.0710678
SIGMA_ARTICULO = np.std([10.0, 20.0], ddof=1)
DEMANDA_ARTICULO = 250.0    # promedio de los meses futuros (200 y 300)
CANTIDAD_BOM = 2.0


def _forecast():
    return pd.DataFrame({
        'Articulo': ['A'] * 5,
        'Fecha': pd.to_datetime(['2025-01-01', '2025-02-01',
                                 '2025-01-01', '2025-02-01', '2025-03-01']),
        'Ventas': [100.0, 120.0,      # histórico real
                   90.0, 100.0,       # pronóstico in-sample (residuos +10 y +20)
                   200.0],            # mes futuro... (se completa abajo)
        'Tipo': ['Historico', 'Historico', 'Pronostico', 'Pronostico', 'Pronostico'],
    })


def _forecast_con_dos_meses_futuros():
    df = _forecast()
    extra = pd.DataFrame({'Articulo': ['A'], 'Fecha': pd.to_datetime(['2025-04-01']),
                          'Ventas': [300.0], 'Tipo': ['Pronostico']})
    return pd.concat([df, extra], ignore_index=True)


def _bom(lead_time=30):
    return pd.DataFrame({
        'Articulo': ['A'],
        'Talle': [42],
        'Componente de lista de materia': ['INSUMO TEST'],
        'Cantidad': [CANTIDAD_BOM],
        'UM': ['UN'],
        'Lead Time': [lead_time],
    })


def _cfg(continua: bool, lead_time: int = 30):
    return replace(CFG_BASE, politicas=(
        PoliticaInsumo('INSUMO TEST', r'INSUMO TEST', continua=continua),))


def _calcular(cfg, lead_time=30):
    return politicas.calcular_politicas_de_inventario(
        _forecast_con_dos_meses_futuros(), _bom(lead_time),
        construir_curva_de_talles(cfg), 'Pronostico', cfg)


def test_demanda_por_articulo_separa_historico_de_futuro():
    demanda = politicas.demanda_por_articulo(_forecast_con_dos_meses_futuros(),
                                             'Pronostico')

    assert demanda.media_mensual['A'] == pytest.approx(DEMANDA_ARTICULO)
    assert demanda.sigma['A'] == pytest.approx(SIGMA_ARTICULO)


def test_demanda_exige_exactamente_dos_valores_de_tipo():
    df = _forecast_con_dos_meses_futuros()
    df.loc[0, 'Tipo'] = 'Presupuesto'

    with pytest.raises(ValueError, match='exactamente dos valores'):
        politicas.demanda_por_articulo(df, 'Pronostico')


def test_demanda_falla_si_no_hay_meses_futuros():
    df = _forecast().iloc[:4]   # solo los meses con histórico

    with pytest.raises(ValueError, match='meses futuros'):
        politicas.demanda_por_articulo(df, 'Pronostico')


def test_revision_continua_usa_solo_el_lead_time_como_horizonte():
    cfg = _cfg(continua=True)

    fila = _calcular(cfg).iloc[0]

    horizonte = 30 / 30   # LT en meses, sin período de revisión
    d_mes = CANTIDAD_BOM * DEMANDA_ARTICULO
    sigma_mes = CANTIDAD_BOM * SIGMA_ARTICULO
    ss = cfg.z_servicio * sigma_mes * np.sqrt(horizonte)

    assert fila['Política'] == 'Revisión continua (s,Q)'
    assert fila['Demanda_media_mensual'] == pytest.approx(round(d_mes, 1))
    assert fila['Desvio_mensual'] == pytest.approx(round(sigma_mes, 1))
    assert fila['Stock_Seguridad'] == pytest.approx(round(ss, 1))
    assert fila['ROP_o_Nivel_Objetivo'] == pytest.approx(round(d_mes * horizonte + ss, 1))


def test_revision_periodica_suma_el_mes_de_revision_al_horizonte():
    cfg = _cfg(continua=False)

    fila = _calcular(cfg).iloc[0]

    horizonte = 30 / 30 + 1.0   # LT + R
    sigma_mes = CANTIDAD_BOM * SIGMA_ARTICULO
    ss = cfg.z_servicio * sigma_mes * np.sqrt(horizonte)

    assert fila['Política'] == 'Revisión periódica (R,S)'
    assert fila['Stock_Seguridad'] == pytest.approx(round(ss, 1))


def test_la_politica_periodica_exige_mas_stock_de_seguridad_que_la_continua():
    # A igual lead time, cubrir LT + R obliga a más stock que cubrir solo LT.
    continua = _calcular(_cfg(continua=True)).iloc[0]
    periodica = _calcular(_cfg(continua=False)).iloc[0]

    assert periodica['Stock_Seguridad'] > continua['Stock_Seguridad']


def test_la_cobertura_expresa_el_stock_de_seguridad_en_dias():
    fila = _calcular(_cfg(continua=True)).iloc[0]

    esperado = fila['Stock_Seguridad'] / fila['Demanda_media_mensual'] * 30

    assert fila['Cobertura_SS_dias'] == pytest.approx(esperado, abs=0.05)


def test_falla_si_ningun_componente_matchea_el_patron():
    cfg = replace(CFG_BASE, politicas=(
        PoliticaInsumo('NO EXISTE', r'NO EXISTE', continua=True),))

    with pytest.raises(ValueError, match='Ningún componente'):
        _calcular(cfg)


def test_r_meses_es_cero_solo_en_revision_continua():
    assert PoliticaInsumo('X', 'X', continua=True).R_meses == 0.0
    assert PoliticaInsumo('X', 'X', continua=False).R_meses == 1.0
