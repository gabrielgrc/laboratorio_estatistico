import numpy as np
import pytest

from stats.minhastats import media


def test_media_lista_inteiros():

    dados = [1, 2, 3, 4, 5]

    assert media(dados) == pytest.approx(np.mean(dados))


def test_media_lista_decimais():

    dados = [2.5, 4.8, 1.3, 7.1]

    assert media(dados) == pytest.approx(np.mean(dados))


def test_media_negativos():

    dados = [-10, 3, 15, -8]

    assert media(dados) == pytest.approx(np.mean(dados))


def test_media_um_elemento():

    dados = [42]

    assert media(dados) == pytest.approx(np.mean(dados))


def test_media_lista_vazia():

    with pytest.raises(ValueError):
        media([])
