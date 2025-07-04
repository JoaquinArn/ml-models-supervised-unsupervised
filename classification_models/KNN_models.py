# -*- coding: utf-8 -*-
"""

@author: Joaco
"""
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

#%% CARGO EL DATASET IRIS

iris = load_iris(as_frame = True)

data = iris.frame
atributos = iris.data
Y = iris.target

iris.target_names
diccionario = dict(zip( [0,1,2], iris.target_names))

#%% HAGO ANÁLISIS EXPLORATORIO A TRAVÉS DE SCATTERPLOTS

#Veo los sépalos
plt.figure(figsize=(10,10))
sns.scatterplot(data = data, x = 'sepal length (cm)' , y =  atributos['sepal width (cm)'], hue='target', palette='viridis')
plt.savefig('pairplot_iris')

#Veo los pétalos
plt.figure(figsize=(10,10))
sns.scatterplot(data = data, x = 'petal length (cm)' , y =  atributos['petal width (cm)'], hue='target', palette='viridis')
plt.savefig('pairplot_iris')

#%% ENTRENO UN CLASIFICADOR KNN
#Este clasificador lo hago, en primera instancia, en base a los cuatro atributos.
clasificador = KNeighborsClassifier(n_neighbors=4)

#Designo en primer lugar el conjunto de entrenamiento
entrenamiento = data.iloc[list(range(0, 25)) + list(range(50, 75)) + list(range(100, 125))]

#Divido al conjunto en atributos a evaluar y los resultados/etiqueta
atributos_entrenamiento = entrenamiento[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
etiquetas_entrenamiento = entrenamiento['target']

#Entreno el clasificador
clasificador.fit(atributos_entrenamiento, etiquetas_entrenamiento)

#%% REALIZO PREDICCIONES

#Designo el conjunto de prueba
prueba = data.iloc[list(range(25, 50)) + list(range(75, 100)) + list(range(125, 150))]

#Divido al conjunto en atributos a evaluar y los resultados esperados/etiquetas
atributos_prueba = prueba[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
etiquetas_prueba = prueba['target']

#Hago las predicciones
res = clasificador.predict(atributos_prueba)

#%% MIDO EXACTITUD
exactitud = sum(res == etiquetas_prueba)

#%% función para graficar la forontera de decision
def plot_decision_boundary(X, y, clf):
    fig, ax = plt.subplots(figsize=(6, 6))    
    # Crear grilla
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h)) 
    # Predecir clases en cada punto de la grilla
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Colores
    n_classes = len(np.unique(y))
    colors = plt.cm.Pastel1.colors[:n_classes]
    cmap_light = ListedColormap(colors)
    cmap_bold = ListedColormap(colors)
    # Graficar la frontera de decisión
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, s=40, edgecolor='k')
    ax.set_xlabel("Largo del pétalo (en centímetros)")
    ax.set_ylabel("Ancho del pétalo (en centímetros)")
    ax.set_title("Frontera de decisión IRIS")

#%% Defino los parámetros
X = data[['petal length (cm)', 'petal width (cm)']].values
Y = data['target'].values
#%% gráfico de dispersión
plt.figure(figsize=(6, 4))
plt.scatter(X[:,0], X[:,1], c=Y, cmap=ListedColormap(['#4CAF50', '#8e44ad', '#FFC0CB']), edgecolor='k')
plt.title("Clasificación Iris")
plt.xlabel("Petal Length (en centímetros)")
plt.ylabel("Petal width (en centímetros)")
plt.show()
#%% clasificador
clf = KNeighborsClassifier(n_neighbors=4)
atributos = data[['petal length (cm)', 'petal width (cm)']]
#Entreno el clasificador
clf.fit(atributos, data['target'])

#%% GRAFICO FRONTERAS DE DECISIÓN
plot_decision_boundary(X, Y,clf)

#%% HACEMOS PREDICCIÓN
clf.predict([[4.8,1.5]])
#Devuelve 1, tal como muestra el gráfico.
