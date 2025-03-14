# Importação das bibliotecas
from sklearn import datasets
import numpy as np
from sklearn.metrics import confusion_matrix
#usada para implementar técnicas de lógica fuzzy e algoritmos fuzzy, como o Fuzzy C-Means (FCM), 
# que é uma técnica de agrupamento baseada em lógica fuzzy.
import skfuzzy



# base de dados iris, que já está disponível no sklearn.
iris = datasets.load_iris()

# Aplicação do algoritmo definindo três cluster (c = 3) e passando a matriz transposta (iris.data.T). 
# Os outros parâmetros são obrigatórios e são os default indicados na documentação

# c é o número de clusters que você deseja que o algoritmo forme.
# No caso, o valor 3 indica que você está tentando encontrar 3 clusters. Isso está de acordo com a base de dados Iris, que tem 3 classes de flores (Setosa, Versicolor, Virginica).
# m = 2:

# m é o índice de fuzzyfication ou parâmetro de suavização. Esse parâmetro determina o grau de "fuzzyfication" dos clusters.
# O valor de m é normalmente maior que 1 (geralmente 2 é escolhido). Quanto maior o valor de m, mais suave e difusa é a associação de um ponto a um cluster.
# O valor m = 2 é o mais comum para o Fuzzy C-Means e indica um comportamento típico de difusão.
# error = 0.005:

# error define a tolerância ou o critério de convergência para o algoritmo.
# O algoritmo Fuzzy C-Means irá iterar até que a mudança nas posições dos centroides (média ponderada dos pontos de dados pertencentes a um cluster) seja menor do que o valor do error.
# Portanto, o algoritmo irá parar quando a mudança for menor que 0.005.
# maxiter = 1000:

# maxiter é o número máximo de iterações que o algoritmo pode executar.
# Isso garante que o algoritmo não entre em um loop infinito se não conseguir convergir para uma solução dentro do critério de erro.
# O valor 1000 é um limite razoável de iterações.

# init = None:

# init é o método utilizado para inicializar os centroides dos clusters.
# Se init for None, o algoritmo irá escolher uma inicialização aleatória dos centroides, o que significa que os pontos iniciais dos clusters serão escolhidos aleatoriamente no espaço de dados.
# Se você fornecer uma matriz de centroids como init, o algoritmo começará a partir dessa inicialização específica.
r = skfuzzy.cmeans(data = iris.data.T, c = 3, m = 2, error = 0.005,
                   maxiter = 1000, init = None)


# Obtendo as porcentagens de um registros pertencente a um cluster, que está na posição 1 da matriz retornada
previsoes_porcentagem = r[1]

#Visualização da probabilidade de um registro pertencer a cada um dos cluster (o somatório é 1.0 que indica 100%)
for x in range(150):
  print( previsoes_porcentagem[0][x] ,previsoes_porcentagem[1][x] ,previsoes_porcentagem[2][x] )
  
  
# Geração de matriz de contingência para comparação com as classes originais da base de dados
# O argmax(axis=0) retorna os índices dos maiores valores em cada coluna.
previsoes = previsoes_porcentagem.argmax(axis = 0)
resultados = confusion_matrix(iris.target, previsoes)
print(resultados)