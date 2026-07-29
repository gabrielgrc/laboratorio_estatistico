from statistics import multimode

import pytest

from stats.minhastats import moda


def test_moda_unica():

    dados = [1, 2, 2, 3, 4]

    esperado = multimode(dados)

    assert moda(dados) == esperado


def test_moda_duas_modas():

    dados = [1, 1, 2, 2, 3]

    esperado = multimode(dados)

    assert sorted(moda(dados)) == sorted(esperado)


def test_moda_tres_modas():

    dados = [1, 1, 2, 2, 3, 3, 4]

    esperado = multimode(dados)

    assert sorted(moda(dados)) == sorted(esperado)


def test_moda_todos_mesma_frequencia():

    dados = [10, 20, 30, 40]

    esperado = multimode(dados)

    assert sorted(moda(dados)) == sorted(esperado)


def test_moda_um_elemento():

    dados = [42]

    esperado = multimode(dados)

    assert moda(dados) == esperado


def test_moda_valores_negativos():

    dados = [-1, -1, -2, -3, -3, -3]

    esperado = multimode(dados)

    assert moda(dados) == esperado


def test_moda_valores_decimais():

    dados = [1.5, 2.5, 1.5, 3.8]

    esperado = multimode(dados)

    assert moda(dados) == esperado


def test_moda_lista_vazia():

    with pytest.raises(ValueError):
        moda([])
