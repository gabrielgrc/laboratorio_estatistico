import pytest

from stats.minhastats import amplitude


def test_amplitude_lista_inteiros():

    dados = [1, 2, 3, 4, 5]

    esperado = max(dados) - min(dados)

    assert amplitude(dados) == esperado


def test_amplitude_lista_desordenada():

    dados = [8, 2, 15, 4, 10]

    esperado = max(dados) - min(dados)

    assert amplitude(dados) == esperado


def test_amplitude_valores_negativos():

    dados = [-10, -5, -20, -3]

    esperado = max(dados) - min(dados)

    assert amplitude(dados) == esperado


def test_amplitude_valores_decimais():

    dados = [1.5, 4.8, 2.3, 0.7]

    esperado = max(dados) - min(dados)

    assert amplitude(dados) == pytest.approx(esperado)


def test_amplitude_um_elemento():

    dados = [42]

    esperado = max(dados) - min(dados)

    assert amplitude(dados) == esperado


def test_amplitude_valores_repetidos():

    dados = [5, 5, 5, 5]

    esperado = max(dados) - min(dados)

    assert amplitude(dados) == esperado


def test_amplitude_lista_vazia():

    with pytest.raises(ValueError):
        amplitude([])
