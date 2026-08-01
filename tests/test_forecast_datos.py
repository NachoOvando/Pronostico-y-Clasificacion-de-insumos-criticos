import pandas as pd
import pytest

from planificacion.forecast import datos


def _wide():
    return pd.DataFrame({
        'Articulo': ['A', 'B'],
        'ene-23': [10, 100],
        'feb-23': [20, 200],
        'dic-24': [30, 300],
    })


def test_wide_a_long_parsea_meses_en_espanol():
    df = datos.pasar_a_formato_largo(_wide())

    assert list(df.columns) == ['Articulo', 'Volumen_Ventas', 'Fecha']
    assert df['Fecha'].min() == pd.Timestamp('2023-01-01')
    assert df['Fecha'].max() == pd.Timestamp('2024-12-01')


def test_wide_a_long_ordena_por_articulo_y_fecha():
    df = datos.pasar_a_formato_largo(_wide())

    assert df['Articulo'].tolist() == ['A', 'A', 'A', 'B', 'B', 'B']
    for articulo in ('A', 'B'):
        fechas = df.loc[df['Articulo'] == articulo, 'Fecha']
        assert fechas.is_monotonic_increasing


def test_wide_a_long_avisa_si_el_mes_no_es_reconocible():
    df_wide = _wide().rename(columns={'ene-23': 'jan-23'})

    with pytest.raises(ValueError, match='jan-23'):
        datos.pasar_a_formato_largo(df_wide)


def test_validar_detecta_nulos_y_negativos():
    df = pd.DataFrame({
        'Articulo': ['A', 'A', 'A'],
        'Fecha': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']),
        'Volumen_Ventas': [10.0, None, -5.0],
    })

    resultado = datos.validar_calidad_de_la_serie(df)

    assert resultado.nulos == 1
    assert len(resultado.negativos) == 1
    assert not resultado.ok


def test_validar_ok_con_datos_limpios():
    resultado = datos.validar_calidad_de_la_serie(datos.pasar_a_formato_largo(_wide()))

    assert resultado.nulos == 0
    assert resultado.negativos.empty
    assert resultado.ok


def test_interpolar_nulos_rellena_dentro_de_cada_articulo():
    df = pd.DataFrame({
        'Articulo': ['A', 'A', 'A', 'B', 'B', 'B'],
        'Fecha': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01'] * 2),
        'Volumen_Ventas': [10.0, None, 30.0, 100.0, None, 300.0],
    })

    relleno = datos.interpolar_nulos(df)['Volumen_Ventas']

    assert relleno.tolist() == [10.0, 20.0, 30.0, 100.0, 200.0, 300.0]


def test_serie_articulo_devuelve_ds_y_ordenada():
    df_prophet = datos.a_formato_prophet(datos.pasar_a_formato_largo(_wide()))

    serie = datos.serie_articulo(df_prophet, 'B')

    assert list(serie.columns) == ['ds', 'y']
    assert serie['y'].tolist() == [100, 200, 300]
