import numpy as np
import pytest

from stats.minhastats import quartis


def test_quartis_retorna_dicionario_com_chaves_corretas():

    dados = [1, 2, 3, 4, 5]

    resultado = quartis(dados)

    assert isinstance(resultado, dict)
    assert set(resultado.keys()) == {"Q1", "Q2", "Q3"}


def test_quartis_lista_inteiros():

    dados = [1, 2, 3, 4, 5]

    esperado = np.percentile(dados, [25, 50, 75], method="linear")

    obtido = quartis(dados)

    assert obtido["Q1"] == pytest.approx(esperado[0])
    assert obtido["Q2"] == pytest.approx(esperado[1])
    assert obtido["Q3"] == pytest.approx(esperado[2])


def test_quartis_lista_desordenada():

    dados = [8, 2, 10, 4, 6]

    esperado = np.percentile(dados, [25, 50, 75], method="linear")

    obtido = quartis(dados)

    assert obtido["Q1"] == pytest.approx(esperado[0])
    assert obtido["Q2"] == pytest.approx(esperado[1])
    assert obtido["Q3"] == pytest.approx(esperado[2])


def test_quartis_valores_decimais():

    dados = [2.5, 1.3, 7.8, 4.6, 3.1]

    esperado = np.percentile(dados, [25, 50, 75], method="linear")

    obtido = quartis(dados)

    assert obtido["Q1"] == pytest.approx(esperado[0])
    assert obtido["Q2"] == pytest.approx(esperado[1])
    assert obtido["Q3"] == pytest.approx(esperado[2])


def test_quartis_valores_negativos():

    dados = [-10, -5, -3, -8, -1]

    esperado = np.percentile(dados, [25, 50, 75], method="linear")

    obtido = quartis(dados)

    assert obtido["Q1"] == pytest.approx(esperado[0])
    assert obtido["Q2"] == pytest.approx(esperado[1])
    assert obtido["Q3"] == pytest.approx(esperado[2])


def test_quartis_um_elemento():

    dados = [42]

    esperado = np.percentile(dados, [25, 50, 75], method="linear")

    obtido = quartis(dados)

    assert obtido["Q1"] == pytest.approx(esperado[0])
    assert obtido["Q2"] == pytest.approx(esperado[1])
    assert obtido["Q3"] == pytest.approx(esperado[2])


def test_quartis_lista_vazia():

    with pytest.raises(ValueError):
        quartis([])
