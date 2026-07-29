import numpy as np
import pytest

from stats.minhastats import desvio_padrao


def test_desvio_padrao_populacional_lista_inteiros():

    dados = [1, 2, 3, 4, 5]

    esperado = np.std(dados)

    assert desvio_padrao(dados) == pytest.approx(esperado)


def test_desvio_padrao_amostral_lista_inteiros():

    dados = [1, 2, 3, 4, 5]

    esperado = np.std(dados, ddof=1)

    assert desvio_padrao(dados, amostral=True) == pytest.approx(esperado)


def test_desvio_padrao_populacional_valores_decimais():

    dados = [2.5, 4.8, 1.3, 7.1]

    esperado = np.std(dados)

    assert desvio_padrao(dados) == pytest.approx(esperado)


def test_desvio_padrao_amostral_valores_decimais():

    dados = [2.5, 4.8, 1.3, 7.1]

    esperado = np.std(dados, ddof=1)

    assert desvio_padrao(dados, amostral=True) == pytest.approx(esperado)


def test_desvio_padrao_populacional_valores_negativos():

    dados = [-10, -5, -2, -8]

    esperado = np.std(dados)

    assert desvio_padrao(dados) == pytest.approx(esperado)


def test_desvio_padrao_amostral_valores_negativos():

    dados = [-10, -5, -2, -8]

    esperado = np.std(dados, ddof=1)

    assert desvio_padrao(dados, amostral=True) == pytest.approx(esperado)


def test_desvio_padrao_populacional_um_elemento():

    dados = [42]

    esperado = np.std(dados)

    assert desvio_padrao(dados) == pytest.approx(esperado)


def test_desvio_padrao_populacional_todos_valores_iguais():

    dados = [5, 5, 5, 5, 5]

    esperado = np.std(dados)

    assert desvio_padrao(dados) == pytest.approx(esperado)


def test_desvio_padrao_amostral_todos_valores_iguais():

    dados = [5, 5, 5, 5, 5]

    esperado = np.std(dados, ddof=1)

    assert desvio_padrao(dados, amostral=True) == pytest.approx(esperado)


def test_desvio_padrao_populacional_lista_vazia():

    with pytest.raises(ValueError):
        desvio_padrao([])


def test_desvio_padrao_amostral_lista_vazia():

    with pytest.raises(ValueError):
        desvio_padrao([], amostral=True)


def test_desvio_padrao_amostral_um_elemento():

    with pytest.raises(ValueError):
        desvio_padrao([42], amostral=True)
