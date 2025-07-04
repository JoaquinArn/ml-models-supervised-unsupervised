# -*- coding: utf-8 -*-
"""

@author: Joaco
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree
import pandas as pd
from sklearn.metrics import accuracy_score
#%% CARGO DATASET A USAR

#Trabajaré con un dataset sobre los árboles Jacarandá, Ceibo, Pindó y Eucaliptus.
#El dataset contendrá información acerca de la altura, diámetro e inclinación de cada registro observado.
 
arboles = pd.read_csv('arboles.csv')

#%% EXPLORACIÓN DE DATOS: REALIZACIÓN DE HISTOGRAMAS

#En esta sección, me interesa conocer ciertas características de cada especie
#Para ello, realizaré histogramas.
#El objetivo es analizar entre qué valores de cada atributo mencionado suelen estar los árboles de cada especie.

#Empiezo con un histograma de altura total

plt.suptitle('Histograma de altura total de los árboles', size = 'large')

sns.histplot(data = arboles, x = 'altura_tot', hue = 'nombre_com', stat = 'count',palette = 'pastel',alpha = 0.7, edgecolor = 'black')

plt.xlabel('Altura Total (en metros)', fontsize = 'medium')
plt.ylabel('Cantidad de observaciones', fontsize = 'medium')

#%% HISTOGRAMA DIÁMETRO ÁRBOLES
plt.suptitle('Histograma de diámetro de los árboles', size = 'large')

sns.histplot(data = arboles, x = 'diametro', hue = 'nombre_com', bins = 30, stat = 'count', palette = 'pastel', alpha = 0.7, edgecolor = 'black')

plt.xlabel('Diámetro (en metros)', fontsize = 'medium')
plt.ylabel('Cantidad de observaciones', fontsize = 'medium')
#%% HISTOGRAMA INCLINACIÓN ÁRBOLES
bin_width = 3.5 *arboles['inclinacio'].std()*(9584**(-1/3))
bins = round((np.max(arboles['inclinacio']) - np.min(arboles['inclinacio']))/bin_width)

plt.suptitle('Histograma de la inclinación de los árboles', size = 'large')

sns.histplot(data = arboles, x = 'inclinacio', hue = 'nombre_com', bins= 40,stat = 'count', palette = 'pastel', alpha = 0.7, edgecolor = 'black')

plt.xlabel('Inclinación (en grados)', fontsize = 'medium')
plt.ylabel('Cantidad de observaciones', fontsize = 'medium')

#%% REALIZO SCATTERPLOT POR DIÁMETRO Y ALTURA
#Por último, siguiendo la línea del análisis exploratorio, haré otro tipo de gráfico.
#Según los histogramas, los atributos diámetro y altura son mejores que la inclinación para determinar la especie.
#Por eso me gustaría ver la distribución de los árboles en función de éstos.

plt.suptitle('Distribución árboles en función del diámetro y la altura', size = 'large')
sns.scatterplot(data = arboles, x = 'diametro', y = 'altura_tot', hue = 'nombre_com', palette = 'deep')

plt.legend(title = 'Especie')
plt.xlabel('Diámetro (en metros)', fontsize = 'medium')
plt.ylabel('Altura (en metros)', fontsize = 'medium')

#%% ÁRBOLES DE DECISIÓN.

#Entrenaré árboles de decisión para clasificar árboles.
#En primera instancia, será de profundidad 3; en segunda, 4.
#Para ambos, utilizaré criterio de entriopía y de Gini. 

#Separaré un conjunto de entrenamiento para usarlo en el árbol
criterios = arboles[['altura_tot', 'diametro', 'inclinacio']]
entrenamiento = criterios.iloc[:8000, :3]
etiquetas_entrenamiento = arboles.iloc[:8000, 3:]

#Separo el conjunto de prueba, sobre el que evaluaré las predicciones.
prueba = criterios.iloc[8000: , :3]
etiquetas_prueba = arboles['nombre_com'].iloc[8000:]

#%% CREACIÓN ÁRBOLES DE DECISIÓN CON CRITERIO ENTRIOPÍA PROFUNDIDAD 3

#Primero defino el árbol de profundidad 3.
arbol_clasificacion01 = tree.DecisionTreeClassifier(criterion = "entropy", max_depth= 3)
arbol_clasificacion01 = arbol_clasificacion01.fit(entrenamiento, etiquetas_entrenamiento)

#Hago la predicción del primer árbol
prediccion01 = arbol_clasificacion01.predict(prueba)

#Evalúo exactitud del árbol (en porcentaje)

exactitud01 = accuracy_score(prediccion01, etiquetas_prueba) *100

#Lo grafico
plt.figure(figsize= [20,10])
tree.plot_tree(arbol_clasificacion01,feature_names=['Altura total', 'Diametro', 'Inclinacio'],class_names = ['Ceibo', 'Eucalipto', 'Jacarandá', 'Pindó'],filled = True, rounded = True, fontsize = 10)
plt.suptitle('Árbol clasificador tres variables con profundidad 3 y criterio: entriopía', size = 30)

#Notar que no está siendo considerada la variable inclinación, es decir, se podría decir que clasifica en base a dos variables.

#%% CREACIÓN ÁRBOLES DE DECISIÓN CON CRITERIO GINI PROFUNDIDAD 3

#Primero defino el árbol de profundidad 3.
arbol_clasificacion11 = tree.DecisionTreeClassifier(criterion = "gini", max_depth= 3)
arbol_clasificacion11 = arbol_clasificacion11.fit(entrenamiento, etiquetas_entrenamiento)

#Hago la predicción del primer árbol
prediccion11 = arbol_clasificacion11.predict(prueba)

#Evalúo exactitud del árbol (en porcentaje)

exactitud11 = accuracy_score(prediccion11, etiquetas_prueba) *100

#Lo grafico
plt.figure(figsize= [20,10])
tree.plot_tree(arbol_clasificacion11,feature_names=['Altura total', 'Diametro', 'Inclinacio'],class_names = ['Ceibo', 'Eucalipto', 'Jacarandá', 'Pindó'],filled = True, rounded = True, fontsize = 10)
plt.suptitle('Árbol clasificador tres variables con profundidad 3 y criterio: Gini', size = 30)

#Notar que, a diferencia del uso del criterio Entropía, éste sí considera la variable inclinación

#%% CREACIÓN ÁRBOLES DE DECISIÓN CON CRITERIO ENTRIOPÍA PROFUNDIDAD 4
#Ahora defino el árbol de profundidad 4.
arbol_clasificacion02 = tree.DecisionTreeClassifier(criterion = "entropy", max_depth= 4)
arbol_clasificacion02 = arbol_clasificacion02.fit(entrenamiento, etiquetas_entrenamiento)

#Hago la predicción del segundo árbol
prediccion02 = arbol_clasificacion02.predict(prueba)

#Evalúo la exactitud del árbol (en porcentaje)

exactitud02 = accuracy_score(prediccion02, etiquetas_prueba) *100

#Lo grafico
plt.figure(figsize= [35,10])
tree.plot_tree(arbol_clasificacion02,feature_names=['Altura total', 'Diametro', 'Inclinacio'],class_names = ['Ceibo', 'Eucalipto', 'Jacarandá', 'Pindó'],filled = True, rounded = True, fontsize = 10)
plt.suptitle('Árbol clasificador tres variables con profundidad 4 y criterio: entriopía', size = 40)

#%% CREACIÓN ÁRBOLES DE DECISIÓN CON CRITERIO GINI PROFUNDIDAD 4
#Ahora defino el árbol de profundidad 4.
arbol_clasificacion22 = tree.DecisionTreeClassifier(criterion = "gini", max_depth= 4)
arbol_clasificacion22 = arbol_clasificacion22.fit(entrenamiento, etiquetas_entrenamiento)

#Hago la predicción del segundo árbol
prediccion22 = arbol_clasificacion22.predict(prueba)

#Evalúo la exactitud del árbol (en porcentaje)

exactitud22 = accuracy_score(prediccion02, etiquetas_prueba) *100

#Lo grafico
plt.figure(figsize= [35,10])
tree.plot_tree(arbol_clasificacion22,feature_names=['Altura total', 'Diametro', 'Inclinacio'],class_names = ['Ceibo', 'Eucalipto', 'Jacarandá', 'Pindó'],filled = True, rounded = True, fontsize = 10)
plt.suptitle('Árbol clasificador tres variables con profundidad 4 y criterio: entriopía', size = 40)

#%% PREDICCIÓN ÁRBOL
#Tengo una nueva fila: [22, 56, 8] (altura, diámetro, inclinación; respectivamente)
#Cómo lo clasificará cada modelo?
X_nuevo = pd.DataFrame([[22, 56, 8]], columns=['altura_tot', 'diametro', 'inclinacio'])

print(arbol_clasificacion01.predict(X_nuevo))
print(arbol_clasificacion11.predict(X_nuevo))
print(arbol_clasificacion02.predict(X_nuevo))
print(arbol_clasificacion22.predict(X_nuevo))

#Todos Eucalipto!
