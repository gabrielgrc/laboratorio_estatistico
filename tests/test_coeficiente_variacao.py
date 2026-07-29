import numpy as np
import pytest

from stats.minhastats import coeficiente_variacao


def test_coeficiente_variacao_populacional_lista_inteiros():

    dados = [1, 2, 3, 4, 5]

    esperado = (np.std(dados) / np.mean(dados)) * 100

    assert coeficiente_variacao(dados) == pytest.approx(esperado)


def test_coeficiente_variacao_amostral_lista_inteiros():

    dados = [1, 2, 3, 4, 5]

    esperado = (np.std(dados, ddof=1) / np.mean(dados)) * 100

    assert coeficiente_variacao(dados, amostral=True) == pytest.approx(esperado)


def test_coeficiente_variacao_populacional_valores_decimais():

    dados = [2.5, 4.8, 1.3, 7.1]

    esperado = (np.std(dados) / np.mean(dados)) * 100

    assert coeficiente_variacao(dados) == pytest.approx(esperado)


def test_coeficiente_variacao_amostral_valores_decimais():

    dados = [2.5, 4.8, 1.3, 7.1]

    esperado = (np.std(dados, ddof=1) / np.mean(dados)) * 100

    assert coeficiente_variacao(dados, amostral=True) == pytest.approx(esperado)


def test_coeficiente_variacao_populacional_valores_negativos():

    dados = [-10, -8, -6, -4]

    esperado = (np.std(dados) / np.mean(dados)) * 100

    assert coeficiente_variacao(dados) == pytest.approx(esperado)


def test_coeficiente_variacao_amostral_valores_negativos():

    dados = [-10, -8, -6, -4]

    esperado = (np.std(dados, ddof=1) / np.mean(dados)) * 100

    assert coeficiente_variacao(dados, amostral=True) == pytest.approx(esperado)


def test_coeficiente_variacao_um_elemento():

    dados = [42]

    esperado = (np.std(dados) / np.mean(dados)) * 100

    assert coeficiente_variacao(dados) == pytest.approx(esperado)


def test_coeficiente_variacao_todos_valores_iguais():

    dados = [5, 5, 5, 5, 5]

    esperado = (np.std(dados) / np.mean(dados)) * 100

    assert coeficiente_variacao(dados) == pytest.approx(esperado)


def test_coeficiente_variacao_lista_vazia():

    with pytest.raises(ValueError):
        coeficiente_variacao([])


def test_coeficiente_variacao_media_igual_zero():

    dados = [-2, -1, 1, 2]

    with pytest.raises(ValueError):
        coeficiente_variacao(dados)


def test_coeficiente_variacao_media_igual_zero_amostral():

    dados = [-2, -1, 1, 2]

    with pytest.raises(ValueError):
        coeficiente_variacao(dados, amostral=True)
