import numpy as np
import pytest

from stats.minhastats import covariancia


def test_covariancia_simetrica():

    x = [1, 2, 3, 4, 5]
    y = [2, 1, 5, 3, 4]

    assert covariancia(x, y) == pytest.approx(covariancia(y, x))


def test_covariancia_populacional_lista_inteiros():

    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    esperado = np.cov(x, y, bias=True)[0, 1]

    assert covariancia(x, y) == pytest.approx(esperado)


def test_covariancia_amostral_lista_inteiros():

    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    esperado = np.cov(x, y, bias=False)[0, 1]

    assert covariancia(x, y, amostral=True) == pytest.approx(esperado)


def test_covariancia_populacional_valores_decimais():

    x = [1.5, 2.7, 3.2, 4.8]
    y = [5.1, 6.4, 7.8, 8.9]

    esperado = np.cov(x, y, bias=True)[0, 1]

    assert covariancia(x, y) == pytest.approx(esperado)


def test_covariancia_amostral_valores_decimais():

    x = [1.5, 2.7, 3.2, 4.8]
    y = [5.1, 6.4, 7.8, 8.9]

    esperado = np.cov(x, y, bias=False)[0, 1]

    assert covariancia(x, y, amostral=True) == pytest.approx(esperado)


def test_covariancia_populacional_valores_negativos():

    x = [-4, -3, -2, -1]
    y = [-8, -6, -4, -2]

    esperado = np.cov(x, y, bias=True)[0, 1]

    assert covariancia(x, y) == pytest.approx(esperado)


def test_covariancia_amostral_valores_negativos():

    x = [-4, -3, -2, -1]
    y = [-8, -6, -4, -2]

    esperado = np.cov(x, y, bias=False)[0, 1]

    assert covariancia(x, y, amostral=True) == pytest.approx(esperado)


def test_covariancia_um_elemento_populacional():

    x = [10]
    y = [20]

    esperado = np.cov(x, y, bias=True)[0, 1]

    assert covariancia(x, y) == pytest.approx(esperado)


def test_covariancia_todos_valores_iguais():

    x = [5, 5, 5, 5]
    y = [8, 8, 8, 8]

    esperado = np.cov(x, y, bias=True)[0, 1]

    assert covariancia(x, y) == pytest.approx(esperado)


def test_covariancia_listas_vazias():

    with pytest.raises(ValueError):
        covariancia([], [])


def test_covariancia_tamanhos_diferentes():

    with pytest.raises(ValueError):
        covariancia([1, 2, 3], [1, 2])


def test_covariancia_amostral_um_elemento():

    with pytest.raises(ValueError):
        covariancia([1], [2], amostral=True)
