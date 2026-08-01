import numpy as np
import pandas as pd
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


def test_baseline_naive_repite_el_mismo_mes_del_ano_previo():
    # 24 meses: los últimos 12 son idénticos a los 12 previos -> predicción exacta.
    ciclo = list(range(1, 13))
    serie = pd.DataFrame({'y': np.array(ciclo + ciclo, dtype=float)})

    resultado = m.baseline_naive_estacional(serie, test_periods=12)

    assert resultado['RMSE'] == pytest.approx(0.0)
    assert resultado['R2'] == pytest.approx(1.0)


def test_baseline_naive_usa_el_ano_anterior_no_el_actual():
    serie = pd.DataFrame({'y': [1.0, 2.0, 5.0, 9.0]})

    resultado = m.baseline_naive_estacional(serie, test_periods=2)

    # Compara [5, 9] contra [1, 2]: errores 4 y 7 -> MAE 5.5
    assert resultado['MAE'] == pytest.approx(5.5)


def test_baseline_naive_exige_dos_ciclos_completos():
    serie = pd.DataFrame({'y': [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match='al menos 4'):
        m.baseline_naive_estacional(serie, test_periods=2)
