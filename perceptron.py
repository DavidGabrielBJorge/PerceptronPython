# -*- coding: utf-8 -*-

import numpy as np
import random as rd
# Importanto bibliotecas 

x=np.array([[1.0,1.0],[1.0,-1.0],[-1.0,1.0],[-1.0,-1.0]])
# Esse comando vai criar a primeira matriz
#    P    Q   PouQ
# |  1 |  1  |  1 |
# |  1 | -1  |  1 |
# | -1 |  1  |  1 |
# | -1 | -1  | -1 |

t=np.array([[1.0],[1.0],[1.0],[-1.0]])
#Esse comando é o target, o resultado que esperamos da matriz
# |  1 |
# |  1 |
# |  1 |
# | -1 |

limiar=0.0

alfa=0.1
#taxa de aprendizagem alfa

(amostras, entradas)=np.shape(x)
#Retorna a quantidade de elementos de linhas e colunas

v=np.zeros((entradas,1))
#Vetor que vai receber os pesos e seu tamanho
#np.zeros((amostras,1))=tantas linhas quanto se tem em "amostras" e uma coluna 

vanterior=np.zeros((entradas,1))
#armazena temporarariamente os valores atuais dos pesos

yin=np.zeros((amostras,1))
#Declarando as saídas puras

y=np.zeros((amostras,1))
#Declarando as saídas liquidas
for i in range(entradas):
    v[i]=rd.uniform(-0.5, 0.5)
v0=rd.uniform(-0.5, 0.5)
#Declarando o Byers
v0anterior=0.0

#"Enquanto" para ver quantas vezes o processo foi repetido e os seus acertos
test=1
ciclo=0
while test==1:
    cont=0
    for i in range(amostras):
        yin[i]=np.dot(x[i,:], v)+v0#Esse comando vai fazer a multiplicacao de 2 matrizes
        if yin[i]>=limiar:
            y[i]=1.0
        else:
            y[i]=-1.0
        if y[i]==t[i]:
            cont=cont+1#caso chegue em 4 os valores estão certos e deve para o treinamento
        vanterior=v
        for j in range(entradas):
            v[j]=vanterior[j]+ alfa*(t[i]-y[i]*x[i][j])#formula de treinamento do perceptron pag. 8
        v0anterior=v0
        v0=v0anterior+alfa*(t[i]-y[i])
    ciclo=ciclo+1
    print('ciclo')
    print(ciclo)
    if cont==amostras:
        test=0
print(v)
print(v0)