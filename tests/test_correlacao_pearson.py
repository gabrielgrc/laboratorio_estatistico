import numpy as np
import pytest

from stats.minhastats import correlacao_pearson


def test_correlacao_simetrica():

    x = [1, 2, 3, 4, 5]
    y = [5, 1, 4, 2, 3]

    assert correlacao_pearson(x, y) == pytest.approx(
        correlacao_pearson(y, x)
    )


def test_correlacao_positiva_perfeita():

    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    esperado = np.corrcoef(x, y)[0, 1]

    assert correlacao_pearson(x, y) == pytest.approx(esperado)


def test_correlacao_negativa_perfeita():

    x = [1, 2, 3, 4, 5]
    y = [10, 8, 6, 4, 2]

    esperado = np.corrcoef(x, y)[0, 1]

    assert correlacao_pearson(x, y) == pytest.approx(esperado)


def test_correlacao_valores_decimais():

    x = [1.5, 2.7, 3.8, 4.2]
    y = [3.1, 5.4, 7.8, 8.6]

    esperado = np.corrcoef(x, y)[0, 1]

    assert correlacao_pearson(x, y) == pytest.approx(esperado)


def test_correlacao_valores_negativos():

    x = [-4, -3, -2, -1]
    y = [-8, -6, -4, -2]

    esperado = np.corrcoef(x, y)[0, 1]

    assert correlacao_pearson(x, y) == pytest.approx(esperado)


def test_correlacao_amostral():

    x = [1, 3, 5, 7, 9]
    y = [2, 6, 10, 14, 18]

    esperado = np.corrcoef(x, y)[0, 1]

    assert correlacao_pearson(x, y, amostral=True) == pytest.approx(
        esperado
    )


def test_correlacao_populacional_igual_amostral():

    x = [2, 4, 6, 8, 10]
    y = [1, 3, 5, 7, 9]

    assert correlacao_pearson(
        x,
        y
    ) == pytest.approx(
        correlacao_pearson(
            x,
            y,
            amostral=True
        )
    )


def test_correlacao_listas_vazias():

    with pytest.raises(ValueError):
        correlacao_pearson([], [])


def test_correlacao_tamanhos_diferentes():

    with pytest.raises(ValueError):
        correlacao_pearson(
            [1, 2, 3],
            [1, 2]
        )


def test_correlacao_desvio_padrao_zero_x():

    x = [5, 5, 5, 5]
    y = [1, 2, 3, 4]

    with pytest.raises(ValueError):
        correlacao_pearson(x, y)


def test_correlacao_desvio_padrao_zero_y():

    x = [1, 2, 3, 4]
    y = [8, 8, 8, 8]

    with pytest.raises(ValueError):
        correlacao_pearson(x, y)


def test_correlacao_desvio_padrao_zero_ambos():

    x = [5, 5, 5]
    y = [9, 9, 9]

    with pytest.raises(ValueError):
        correlacao_pearson(x, y)
