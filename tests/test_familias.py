import pandas as pd
import pytest

from planificacion.insumos import familias


def test_nombre_base_quita_el_talle_con_coma_y_punto():
    assert (familias.nombre_base('CAPELLADA CRONOS, -N, (NA), T. 42')
            == 'CAPELLADA CRONOS, -N, (NA)')


def test_nombre_base_quita_el_talle_pegado_con_punto():
    assert familias.nombre_base('PLANTILLA X T.42') == 'PLANTILLA X'
    assert familias.nombre_base('PLANTILLA X T.45,5') == 'PLANTILLA X'


def test_nombre_base_deja_intacto_el_talle_sin_punto():
    # Nomenclatura real del BOM: estos se consolidan después, por REGLAS_FAMILIA.
    nombre = 'PLANT ARM STROB UL H450/3 T45,5'

    assert familias.nombre_base(nombre) == nombre


def test_nombre_base_no_toca_un_nombre_sin_talle():
    assert familias.nombre_base('  CONJ SISTEMA PU  ') == 'CONJ SISTEMA PU'


def test_asignar_familia_consolida_variantes_bajo_una_regla():
    familia_a, um_a = familias.asignar_familia('PUNTERA ACERO 59 NORMAL T9', 'UN')
    familia_b, um_b = familias.asignar_familia('PUNTERA ACERO 59 NORMAL T10', 'UN')

    assert familia_a == familia_b == 'PUNTERA ACERO 59 NORMAL'
    # La regla también fija la UM de la familia.
    assert um_a == um_b == 'PAA'


def test_asignar_familia_es_insensible_a_mayusculas():
    assert familias.asignar_familia('caja empaque bota voran', 'UN')[0] == \
        'CAJA EMPAQUE (BOTA/BOTÍN)'


def test_asignar_familia_sin_regla_devuelve_el_nombre_y_su_um():
    assert familias.asignar_familia('CORDON TRENZ NEGRO', 'UN') == \
        ('CORDON TRENZ NEGRO', 'UN')


def test_asignar_familia_gana_la_primera_regla_que_matchea():
    # 'PLANT VIS PLANA 4mm FUNC' aparece antes que la variante VORAN en la lista.
    familia, _ = familias.asignar_familia('PLANT VIS PLANA 4mm FUNC MA T40', 'PAA')

    assert familia == 'PLANTILLA VIS PLANA FUNC MA'


def _consumo(nombres, ums):
    return pd.DataFrame({
        'Nombre_Componente_Insumo': nombres,
        'UM': ums,
        'consumo_proyectado': [1.0] * len(nombres),
    })


def test_agregar_columnas_propaga_familia_y_um():
    df = _consumo(
        ['PUNTERA ACERO 59 NORMAL T9', 'PUNTERA ACERO 59 NORMAL T10',
         'CAPELLADA CRONOS, -N, (NA), T. 42'],
        ['UN', 'UN', 'PAA'])

    resultado = familias.asignar_familias_de_compra(df)

    assert resultado['Familia'].tolist() == [
        'PUNTERA ACERO 59 NORMAL', 'PUNTERA ACERO 59 NORMAL', 'CAPELLADA CRONOS, -N, (NA)']
    assert resultado['UM_familia'].tolist() == ['PAA', 'PAA', 'PAA']
    assert resultado['Nombre_Base'].iloc[2] == 'CAPELLADA CRONOS, -N, (NA)'


def test_agregar_columnas_no_muta_el_dataframe_recibido():
    df = _consumo(['CONJ SISTEMA PU'], ['G'])

    familias.asignar_familias_de_compra(df)

    assert 'Familia' not in df.columns


def test_agregar_columnas_falla_claro_si_el_cruce_quedo_vacio():
    df = _consumo([], []).astype({'Nombre_Componente_Insumo': 'object', 'UM': 'object'})

    with pytest.raises(ValueError, match='Forecast × BOM'):
        familias.asignar_familias_de_compra(df)
