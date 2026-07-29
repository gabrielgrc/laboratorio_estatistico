import numpy as np
import pytest

from stats.minhastats import mediana


def test_mediana_impar():

    dados = [5, 1, 7]

    assert mediana(dados) == pytest.approx(np.median(dados))


def test_mediana_par():

    dados = [2, 8, 10, 4]

    assert mediana(dados) == pytest.approx(np.median(dados))


def test_mediana_um_elemento():

    dados = [9]

    assert mediana(dados) == pytest.approx(np.median(dados))


def test_mediana_lista_vazia():

    with pytest.raises(ValueError):
        mediana([])
