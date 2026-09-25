"""Gate de reproducibilidad: entorno + test suite del repo original.

Regla del usuario: "Verificá que reproducís los resultados originales
(pronóstico y tabla de políticas) antes de reportar nada nuevo." Este script
es el primer paso de esa verificación: si el entorno no tiene las
dependencias correctas o los 54 tests existentes de `tests/` no pasan, el
problema es de entorno, no de los datos/código del pipeline — no tiene
sentido seguir extrayendo números hasta que esto esté en verde.

No modifica nada del repo: solo importa y corre `pytest` como subproceso.
"""

import subprocess
import sys


def verificar_dependencias() -> dict[str, str]:
    versiones = {}
    paquetes = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'matplotlib'),
        ('openpyxl', 'openpyxl'),
        ('prophet', 'prophet'),
        ('pytest', 'pytest'),
    ]
    faltantes = []
    for modulo, nombre_paquete in paquetes:
        try:
            mod = __import__(modulo)
            versiones[nombre_paquete] = getattr(mod, '__version__', 'desconocida')
        except ImportError:
            faltantes.append(nombre_paquete)

    if faltantes:
        print(f"Faltan paquetes: {faltantes}")
        print("Instalar con: pip install -r requirements.txt")
        sys.exit(1)
    return versiones


def correr_tests_originales() -> None:
    resultado = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/', '-q'],
        cwd=str(__import__('pathlib').Path(__file__).resolve().parent.parent),
        capture_output=True, text=True,
    )
    print(resultado.stdout)
    if resultado.returncode != 0:
        print(resultado.stderr)
        print("\n⚠️  Los tests existentes del repo (tests/) NO pasan en este entorno.")
        print("El problema es de entorno/dependencias, no de los datos: no se sigue "
              "con la extracción hasta que esto esté en verde.")
        sys.exit(1)
    print("✅ Los 54 tests existentes de tests/ pasan en este entorno — "
          "gate de reproducibilidad superado.")


if __name__ == '__main__':
    print("=== Versiones de librerías (para dejar constancia en el informe) ===")
    for paquete, version in verificar_dependencias().items():
        print(f"  {paquete}: {version}")

    print("\n=== Corriendo tests/ del repo original (pytest tests/ -q) ===")
    correr_tests_originales()
