import numpy as np
import pytest

from stats.minhastats import variancia


def test_variancia_populacional_lista_inteiros():

    dados = [1, 2, 3, 4, 5]

    esperado = np.var(dados)

    assert variancia(dados) == pytest.approx(esperado)


def test_variancia_amostral_lista_inteiros():

    dados = [1, 2, 3, 4, 5]

    esperado = np.var(dados, ddof=1)

    assert variancia(dados, amostral=True) == pytest.approx(esperado)


def test_variancia_populacional_valores_decimais():

    dados = [2.5, 4.8, 1.3, 7.1]

    esperado = np.var(dados)

    assert variancia(dados) == pytest.approx(esperado)


def test_variancia_amostral_valores_decimais():

    dados = [2.5, 4.8, 1.3, 7.1]

    esperado = np.var(dados, ddof=1)

    assert variancia(dados, amostral=True) == pytest.approx(esperado)


def test_variancia_populacional_valores_negativos():

    dados = [-10, -5, -2, -8]

    esperado = np.var(dados)

    assert variancia(dados) == pytest.approx(esperado)


def test_variancia_amostral_valores_negativos():

    dados = [-10, -5, -2, -8]

    esperado = np.var(dados, ddof=1)

    assert variancia(dados, amostral=True) == pytest.approx(esperado)


def test_variancia_populacional_um_elemento():

    dados = [42]

    esperado = np.var(dados)

    assert variancia(dados) == pytest.approx(esperado)


def test_variancia_amostral_um_elemento():

    with pytest.raises(ValueError):
        variancia([42], amostral=True)


def test_variancia_populacional_todos_valores_iguais():

    dados = [5, 5, 5, 5, 5]

    esperado = np.var(dados)

    assert variancia(dados) == pytest.approx(esperado)


def test_variancia_amostral_todos_valores_iguais():

    dados = [5, 5, 5, 5, 5]

    esperado = np.var(dados, ddof=1)

    assert variancia(dados, amostral=True) == pytest.approx(esperado)


def test_variancia_amostral_lista_vazia():

    with pytest.raises(ValueError):
        variancia([], amostral=True)


def test_variancia_populacional_lista_vazia():

    with pytest.raises(ValueError):
        variancia([])
