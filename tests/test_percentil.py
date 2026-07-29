import numpy as np
import pytest

from stats.minhastats import percentil


def test_percentil_zero():

    dados = [1, 2, 3, 4, 5]

    esperado = np.percentile(dados, 0, method="linear")

    assert percentil(dados, 0) == pytest.approx(esperado)


def test_percentil_interpolacao_linear():

    dados = [10, 20, 30, 40]

    esperado = np.percentile(dados, 25, method="linear")

    assert percentil(dados, 25) == pytest.approx(esperado)


def test_percentil_vinte_cinco():

    dados = [1, 2, 3, 4, 5]

    esperado = np.percentile(dados, 25, method="linear")

    assert percentil(dados, 25) == pytest.approx(esperado)


def test_percentil_cinquenta():

    dados = [1, 2, 3, 4, 5]

    esperado = np.percentile(dados, 50, method="linear")

    assert percentil(dados, 50) == pytest.approx(esperado)


def test_percentil_setenta_cinco():

    dados = [1, 2, 3, 4, 5]

    esperado = np.percentile(dados, 75, method="linear")

    assert percentil(dados, 75) == pytest.approx(esperado)


def test_percentil_cem():

    dados = [1, 2, 3, 4, 5]

    esperado = np.percentile(dados, 100, method="linear")

    assert percentil(dados, 100) == pytest.approx(esperado)


def test_percentil_lista_desordenada():

    dados = [8, 2, 10, 4, 6]

    esperado = np.percentile(dados, 40, method="linear")

    assert percentil(dados, 40) == pytest.approx(esperado)


def test_percentil_valores_decimais():

    dados = [2.5, 1.3, 7.8, 4.6, 3.1]

    esperado = np.percentile(dados, 60, method="linear")

    assert percentil(dados, 60) == pytest.approx(esperado)


def test_percentil_valores_negativos():

    dados = [-10, -5, -3, -8, -1]

    esperado = np.percentile(dados, 30, method="linear")

    assert percentil(dados, 30) == pytest.approx(esperado)


def test_percentil_um_elemento():

    dados = [42]

    esperado = np.percentile(dados, 50, method="linear")

    assert percentil(dados, 50) == pytest.approx(esperado)


def test_percentil_lista_vazia():

    with pytest.raises(ValueError):
        percentil([], 50)


def test_percentil_menor_que_zero():

    dados = [1, 2, 3]

    with pytest.raises(ValueError):
        percentil(dados, -10)


def test_percentil_maior_que_cem():

    dados = [1, 2, 3]

    with pytest.raises(ValueError):
        percentil(dados, 110)
  
