import numpy as np
import pytest

from planificacion.forecast import metricas as m


def test_metricas_con_prediccion_perfecta():
    y = [10.0, 20.0, 30.0]

    resultado = m.metricas(y, y)

    assert resultado['MSE'] == pytest.approx(0.0)
    assert resultado['RMSE'] == pytest.approx(0.0)
    assert resultado['MAE'] == pytest.approx(0.0)
    assert resultado['R2'] == pytest.approx(1.0)


def test_metricas_valores_conocidos():
    # Errores: +2, -2, +2  ->  MSE = 4, RMSE = 2, MAE = 2
    resultado = m.metricas([10.0, 20.0, 30.0], [12.0, 18.0, 32.0])

    assert resultado['MSE'] == pytest.approx(4.0)
    assert resultado['RMSE'] == pytest.approx(2.0)
    assert resultado['MAE'] == pytest.approx(2.0)
    # Varianza de y = 200/3  ->  R2 = 1 - 4 / (200/3)
    assert resultado['R2'] == pytest.approx(1 - 4 / (200 / 3))


def test_mape_valores_conocidos():
    # Errores relativos: 2/10 = 20 %, 2/20 = 10 %, 2/30 = 6,67 %  ->  media 12,22 %
    resultado = m.metricas([10.0, 20.0, 30.0], [12.0, 18.0, 32.0])

    assert resultado['MAPE'] == pytest.approx((20 + 10 + 200 / 30) / 3)


def test_mape_es_cero_con_prediccion_perfecta():
    y = [10.0, 20.0, 30.0]

    assert m.metricas(y, y)['MAPE'] == pytest.approx(0.0)


def test_mape_excluye_los_periodos_con_valor_real_cero():
    # El período con y=0 no tiene error porcentual: solo cuenta el de y=10 (error 20 %).
    assert m.mape([0.0, 10.0], [5.0, 12.0]) == pytest.approx(20.0)


def test_mape_es_nan_si_todo_el_real_es_cero():
    assert np.isnan(m.mape([0.0, 0.0], [1.0, 2.0]))
