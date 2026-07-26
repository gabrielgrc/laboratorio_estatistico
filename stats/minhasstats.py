def media(valores):
    """
    Calcula a média aritmética de uma sequência numérica.

    Parameters
    ----------
    valores : list[float]

    Returns
    -------
    float
    """

    if len(valores) == 0:
        raise ValueError("A lista não pode ser vazia.")

    soma = 0

    for valor in valores:
        soma += valor

    return soma / len(valores)


def mediana(valores):
    """
    Calcula a mediana de uma sequência numérica.

    Caso a quantidade de elementos seja ímpar, retorna o elemento central.
    Caso seja par, retorna a média dos dois elementos centrais.

    Parameters
    ----------
    valores : list[float]

    Returns
    -------
    float
    """

    if len(valores) == 0:
        raise ValueError("Lista vazia.")

    dados = sorted(valores)

    n = len(dados)
    meio = n // 2

    if n % 2 == 1:
        return dados[meio]

    return (dados[meio - 1] + dados[meio]) / 2


def moda(valores):
    """
    Calcula a(s) moda(s) de uma sequência numérica.

    Caso exista mais de uma moda, retorna todas elas em uma lista.

    Parameters
    ----------
    valores : list[float]

    Returns
    -------
    list
    """

    if len(valores) == 0:
        raise ValueError("Lista vazia.")

    frequencias = {}

    for valor in valores:
        if valor in frequencias:
            frequencias[valor] += 1
        else:
            frequencias[valor] = 1

    maior = max(frequencias.values())

    modas = []

    for chave, freq in frequencias.items():
        if freq == maior:
            modas.append(chave)

    return modas


def amplitude(valores):
    """
    Calcula a amplitude de uma sequência numérica.

    A amplitude é definida como a diferença entre o maior e o menor valor.

    Parameters
    ----------
    valores : list[float]

    Returns
    -------
    float
    """

    if len(valores) == 0:
        raise ValueError("Lista vazia.")

    menor = valores[0]
    maior = valores[0]

    for valor in valores:

        if valor < menor:
            menor = valor

        if valor > maior:
            maior = valor

    return maior - menor


def variancia(valores, amostral=False):
    """
    Calcula a variância de uma sequência numérica.

    Pode calcular tanto a variância populacional quanto a amostral.

    Parameters
    ----------
    valores : list[float]
    amostral : bool, optional

    Returns
    -------
    float
    """

    n = len(valores)

    if n == 0:
        raise ValueError("A lista não pode estar vazia.")

    if amostral and n < 2:
        raise ValueError("Uma amostra deve possuir pelo menos dois elementos.")

    media_valores = media(valores)

    soma_quadrados = 0

    for valor in valores:
        diferenca = valor - media_valores
        soma_quadrados += diferenca ** 2

    divisor = n - 1 if amostral else n

    return soma_quadrados / divisor


def desvio_padrao(valores, amostral=False):
    """
    Calcula o desvio padrão de uma sequência numérica.

    Pode calcular tanto o desvio padrão populacional quanto o amostral.

    Parameters
    ----------
    valores : list[float]
    amostral : bool, optional

    Returns
    -------
    float
    """

    return variancia(valores, amostral) ** 0.5


def percentil(valores, p):
    """
    Calcula o percentil de uma sequência numérica por interpolação linear.

    Parameters
    ----------
    valores : list[float]
    p : float

    Returns
    -------
    float
    """

    if len(valores) == 0:
        raise ValueError("A lista não pode estar vazia.")

    if p < 0 or p > 100:
        raise ValueError("O percentil deve estar entre 0 e 100.")

    valores = sorted(valores)

    n = len(valores)

    posicao = (p / 100) * (n - 1)

    inferior = int(posicao)
    superior = inferior + 1

    if superior >= n:
        return valores[inferior]

    fracao = posicao - inferior

    return valores[inferior] + fracao * (
        valores[superior] - valores[inferior]
    )


def quartis(valores):
    """
    Calcula os três quartis de uma sequência numérica.

    Os quartis são obtidos a partir dos percentis de 25%, 50% e 75%.

    Parameters
    ----------
    valores : list[float]

    Returns
    -------
    dict
    """

    return {
        "Q1": percentil(valores, 25),
        "Q2": percentil(valores, 50),
        "Q3": percentil(valores, 75)
    }


def coeficiente_variacao(valores, amostral=False):
    """
    Calcula o coeficiente de variação de uma sequência numérica.

    O resultado é expresso em porcentagem.

    Parameters
    ----------
    valores : list[float]
    amostral : bool, optional

    Returns
    -------
    float
    """

    if len(valores) == 0:
        raise ValueError("A lista não pode estar vazia.")

    media_valores = media(valores)

    if media_valores == 0:
        raise ValueError(
            "Não é possível calcular o coeficiente de variação quando a média é igual a zero."
        )

    desvio = desvio_padrao(valores, amostral)

    return (desvio / media_valores) * 100


def covariancia(x, y, amostral=False):
    """
    Calcula a covariância entre duas variáveis.

    Pode calcular tanto a covariância populacional quanto a amostral.

    Parameters
    ----------
    x : list[float]
    y : list[float]
    amostral : bool, optional

    Returns
    -------
    float
    """

    if len(x) != len(y):
        raise ValueError("As duas listas devem possuir o mesmo tamanho.")

    n = len(x)

    if n == 0:
        raise ValueError("As listas não podem estar vazias.")

    if amostral and n < 2:
        raise ValueError("A amostra deve possuir pelo menos dois elementos.")

    media_x = media(x)
    media_y = media(y)

    soma = 0

    for i in range(n):
        soma += (x[i] - media_x) * (y[i] - media_y)

    if amostral:
        return soma / (n - 1)

    return soma / n


def correlacao_pearson(x, y, amostral=False):
    """
    Calcula o coeficiente de correlação de Pearson entre duas variáveis.

    O resultado varia entre -1 e 1, indicando a intensidade e a direção
    da relação linear entre as variáveis.

    Parameters
    ----------
    x : list[float]
    y : list[float]
    amostral : bool, optional

    Returns
    -------
    float
    """

    if len(x) != len(y):
        raise ValueError("As listas devem possuir o mesmo tamanho.")

    if len(x) == 0:
        raise ValueError("As listas não podem estar vazias.")

    cov = covariancia(x, y, amostral)
    dp_x = desvio_padrao(x, amostral)
    dp_y = desvio_padrao(y, amostral)

    if dp_x == 0 or dp_y == 0:
        raise ValueError(
            "Não é possível calcular a correlação quando uma das variáveis possui desvio padrão igual a zero."
        )

    return cov / (dp_x * dp_y)
