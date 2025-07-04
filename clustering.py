# -*- coding: utf-8 -*-
"""
@author: Joaco
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram , cut_tree

#En este archivo, prubo diversos métodos de clustering

#%% CREACIÓN PUNTOS ALEATORIOS
np.random.seed (0);
X = np.random.standard_normal ((50 ,2));
X[:25 ,0] += 5;
X[:25 ,1] -= 4;

#Grafico el conjunto
fig , ax = plt.subplots (1, 1, figsize =(8 ,8))
ax.scatter(X[:,0], X[:,1])

#%% CLUSTERING USANDO ALGORITMO K-MEANS

#Pruebo aplicando el algoritmo con distintos k

#-- K=2 --#
kmeans = KMeans(n_clusters = 2, random_state = 2, n_init = 20)
kmeans.fit(X)
kmeans.labels_
kmeans.inertia_

#Grafico el resultado
fig , ax = plt.subplots (1, 1, figsize =(8 ,8))
ax.scatter(X[:,0], X[:,1], c=kmeans.labels_)
ax.set_title("K-Means Clustering Results with K=2");

#-- K=3 --#
kmeans = KMeans(n_clusters =3, random_state =3, n_init =20)
kmeans.fit(X)
kmeans.inertia_

#Grafico el resultado
fig , ax = plt.subplots(figsize =(8 ,8))
ax.scatter(X[:,0], X[:,1], c=kmeans.labels_)
ax.set_title("K-Means Clustering Results with K=3");


#%% CLUSTERING USANDO ALGORITMO DBSCAN

#Ahora pruebo hacer clustering con el algoritmo de DBSCAN
#Hago variaciones de eps y min_samples

#-- EJEMPLO 1: bajo eps --#
dbclust = DBSCAN(eps=1, min_samples=15)
dbclust.fit(X)
fig , ax = plt.subplots(figsize =(8 ,8))
ax.scatter(X[:,0], X[:,1], c=dbclust.labels_)
ax.set_title("DBSCAN Results");

#-- EJEMPLO 2: aumento eps --#
dbclust = DBSCAN(eps=2, min_samples=15)
dbclust.fit(X)
fig , ax = plt.subplots(figsize =(8 ,8))
ax.scatter(X[:,0], X[:,1], c=dbclust.labels_)
ax.set_title("DBSCAN Results");

#-- EJEMPLO 3: eps óptimo --#
dbclust = DBSCAN(eps=2.3, min_samples=15)
dbclust.fit(X)
fig , ax = plt.subplots(figsize =(8 ,8))
ax.scatter(X[:,0], X[:,1], c=dbclust.labels_)
ax.set_title("DBSCAN Results");

#-- EJEMPLO 6: aumento mucho los min_samples --#
dbclust = DBSCAN(eps=2.3, min_samples=40)
dbclust.fit(X)
fig , ax = plt.subplots(figsize =(8 ,8))
ax.scatter(X[:,0], X[:,1], c=dbclust.labels_)
ax.set_title("DBSCAN Results");

#%% FUNCIONES AUXILIARES PARA VISUALIZACIÓN DE CLUSTERING JERÁRQUICO

def calcular_vinculaciones(modelo):
    # Crear la matriz de vinculaciones
    cuentas = np.zeros(modelo.children_.shape[0])
    n_muestras = len(modelo.labels_)
    for i, fusion in enumerate(modelo.children_):
        cuenta_actual = 0
        for idx_hijo in fusion:
            if idx_hijo < n_muestras:
                cuenta_actual += 1  # nodo hoja
            else:
                cuenta_actual += cuentas[idx_hijo - n_muestras]
        cuentas[i] = cuenta_actual

    matriz_vinculaciones = np.column_stack(
        [modelo.children_, modelo.distances_, cuentas]
    ).astype(float)

    return matriz_vinculaciones


def graficar_dendrograma(modelo, **kwargs):
    # Crear la matriz de vinculaciones y luego graficar el dendrograma

    cuentas = np.zeros(modelo.children_.shape[0])
    n_muestras = len(modelo.labels_)
    for i, fusion in enumerate(modelo.children_):
        cuenta_actual = 0
        for idx_hijo in fusion:
            if idx_hijo < n_muestras:
                cuenta_actual += 1  # nodo hoja
            else:
                cuenta_actual += cuentas[idx_hijo - n_muestras]
        cuentas[i] = cuenta_actual

    matriz_vinculaciones = np.column_stack(
        [modelo.children_, modelo.distances_, cuentas]
    ).astype(float)

    # Graficar el dendrograma correspondiente
    dendrogram(matriz_vinculaciones, **kwargs)


#%% CLUSTERING JERÁRQUICO AGLOMERATIVO

#Por último, pruebo con la aplicación de un algoritmo de clustering jerárquico

#Primero sin número de clusters definido
HClust = AgglomerativeClustering
hc_comp = HClust(distance_threshold =0, n_clusters=None , linkage='complete')
hc_comp.fit(X)
#Grafico el resultado
fig , ax = plt.subplots(figsize =(8 ,8))
ax.scatter(X[:,0], X[:,1], c=hc_comp.labels_)
ax.set_title("Agglomerative Clustering Results with no-predifine number of clusters");
plt.figure(figsize = (15,15))
plt.title("Hierarchical Clustering Dendrogram", fontsize = 20)
graficar_dendrograma(hc_comp)

#Ahora defino 2 clusters
HClust = AgglomerativeClustering
hc_comp = HClust( distance_threshold =None, n_clusters=2 , linkage='complete')
hc_comp.fit(X)
#Grafico el resultado
fig , ax = plt.subplots(figsize =(8 ,8))
ax.scatter(X[:,0], X[:,1], c=hc_comp.labels_)
ax.set_title("Agglomerative Clustering Results in two clusters");

#Desarrollo uno sin número de clusters definidos y con criterio de cercanía 'single linkage'
HClust = AgglomerativeClustering
hc_comp = HClust( distance_threshold =0, n_clusters=None , linkage='single')
hc_comp.fit(X)
#Grafico el resultado
fig , ax = plt.subplots(figsize =(8 ,8))
ax.scatter(X[:,0], X[:,1], c=hc_comp.labels_)
ax.set_title("Agglomerative Clustering Results with single linkage");
plt.figure(figsize = (15,15))
plt.title("Hierarchical Clustering Dendrogram", fontsize = 20)
graficar_dendrograma(hc_comp)

#Desarrollo uno sin número de clusters definidos y con criterio de cercanía 'ward linkage'
HClust = AgglomerativeClustering
hc_comp = HClust( distance_threshold =0, n_clusters=None , linkage='ward')
hc_comp.fit(X)
#Grafico el resultado
fig , ax = plt.subplots(figsize =(8 ,8))
ax.scatter(X[:,0], X[:,1], c=hc_comp.labels_)
ax.set_title("Agglomerative Clustering Results with ward linkage");
plt.figure(figsize = (15,15))
plt.title("Hierarchical Clustering Dendrogram", fontsize = 20)
graficar_dendrograma(hc_comp)
plt.show()

#%% SELECCIONO OTROS DATASETS 

#En este bloque, visualizo diversos datasets en los que usaré cada técnica de clustering vista
#Para ello, primero me interesa ver cómo son los datos de cada uno

seed = 75
n_samples = 500 #elijo tener 500 datos en cada dataset
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)

noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)

blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)

varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=seed)


for dataset in [noisy_moons, noisy_circles, blobs, varied]:
    X, y = dataset
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=10)
    

    plt.xticks(())
    plt.yticks(())
    plt.show()

#En el siguiente bloque, visualizamos los clusters formados a partir del uso de cada método.

#%% CLUSTERING EN DATASET NOISY_MOONS

#En este bloque, visualizo los resultados de aplicar cada método de clustering en el dataset Noisy Moons de sklearn 

#-- Resultados DBSCAN --#
dbclustpruebaMoons = DBSCAN(eps=0.3, min_samples=2)
dbclustpruebaMoons.fit(noisy_moons[0])
fig , ax = plt.subplots(figsize =(8 ,8))
ax.scatter(noisy_moons[0][:,0], noisy_moons[0][:,1], c=dbclustpruebaMoons.labels_)
ax.set_title("DBSCAN Results");

#-- Resultados K-MEANS --#
kmeans = KMeans(n_clusters =2, random_state =3, n_init =20)
kmeans.fit(noisy_moons[0])
fig , ax = plt.subplots(figsize =(8 ,8))
ax.scatter(noisy_moons[0][:,0], noisy_moons[0][:,1], c=kmeans.labels_)
ax.set_title("K-Means Clustering Results with K=2");

#-- Resultados Clustering JERÁRQUICO AGLOMERATIVO --#
HClust = AgglomerativeClustering
hc_compPrueba = HClust( distance_threshold =None, n_clusters=2 , linkage='complete')
hc_compPrueba.fit(noisy_moons[0])
fig , ax = plt.subplots(figsize =(8 ,8))
ax.scatter(noisy_moons[0][:,0], noisy_moons[0][:,1], c=hc_compPrueba.labels_)
ax.set_title("Agglomerative Clustering Results");

#%% CLUSTERING EN DATASET NOISY_CIRCLES

#En este bloque, visualizo los resultados de aplicar cada método de clustering en el dataset Noisy Circles de sklearn

#-- Resultados DBSCAN --#
dbclustpruebaCircle = DBSCAN(eps=0.2, min_samples=2)
dbclustpruebaCircle.fit(noisy_circles[0])
fig , ax = plt.subplots(figsize =(8 ,8))
ax.scatter(noisy_circles[0][:,0], noisy_circles[0][:,1], c=dbclustpruebaCircle.labels_)
ax.set_title("DBSCAN Results");

#-- Resultados K-MEANS --#
kmeans = KMeans(n_clusters =2, random_state =3, n_init =20)
kmeans.fit(noisy_circles[0])
fig , ax = plt.subplots(figsize =(8 ,8))
ax.scatter(noisy_circles[0][:,0], noisy_circles[0][:,1], c=kmeans.labels_)
ax.set_title("K-Means Clustering Results with K=3");

#-- Resultados Clustering JERÁRQUICO AGLOMERATIVO --#
HClust = AgglomerativeClustering
hc_compPrueba = HClust( distance_threshold =None, n_clusters=2 , linkage='complete')
hc_compPrueba.fit(noisy_circles[0])
fig , ax = plt.subplots(figsize =(8 ,8))
ax.scatter(noisy_circles[0][:,0], noisy_circles[0][:,1], c=hc_compPrueba.labels_)
ax.set_title("Agglomerative Clustering Results");

#%% CLUSTERING EN DATASET BLOBS

#En este bloque, visualizo los resultados de aplicar cada método de clustering en el dataset Blobs de sklearn

#-- Resultados DBSCAN --#
dbclustpruebaBlobs = DBSCAN(eps=0.2, min_samples=2)
dbclustpruebaBlobs.fit(noisy_circles[0])
fig , ax = plt.subplots(figsize =(8 ,8))
ax.scatter(blobs[0][:,0], blobs[0][:,1], c=dbclustpruebaBlobs.labels_)
ax.set_title("DBSCAN Results");

#-- Resultados K-MEANS --#
kmeans = KMeans(n_clusters =3, random_state =3, n_init =20)
kmeans.fit(blobs[0])
fig , ax = plt.subplots(figsize =(8 ,8))
ax.scatter(blobs[0][:,0], blobs[0][:,1], c=kmeans.labels_)
ax.set_title("K-Means Clustering Results with K=3");

#-- Resultados Clustering JERÁRQUICO AGLOMERATIVO --#
HClust = AgglomerativeClustering
hc_compPrueba = HClust( distance_threshold =None, n_clusters=3 , linkage='complete')
hc_compPrueba.fit(blobs[0])
fig , ax = plt.subplots(figsize =(8 ,8))
ax.scatter(blobs[0][:,0], blobs[0][:,1], c=hc_compPrueba.labels_)
ax.set_title("Agglomerative Clustering Results");

#%% CLUSTERING EN DATASET VARIED

#En este bloque, visualizo los resultados de aplicar cada método de clustering en el dataset Varied de sklearn

#-- Resultados DBSCAN --#
dbclustpruebaVaried = DBSCAN(eps=0.2, min_samples=2)
dbclustpruebaVaried.fit(varied[0])
fig , ax = plt.subplots(figsize =(8 ,8))
ax.scatter(varied[0][:,0], varied[0][:,1], c=dbclustpruebaVaried.labels_)
ax.set_title("DBSCAN Results");

#-- Resultados K-MEANS --#
kmeans = KMeans(n_clusters =3, random_state =3, n_init =20)
kmeans.fit(varied[0])
fig , ax = plt.subplots(figsize =(8 ,8))
ax.scatter(varied[0][:,0], varied[0][:,1], c=kmeans.labels_)
ax.set_title("K-Means Clustering Results with K=3");

#-- Resultados Clustering JERÁRQUICO AGLOMERATIVO --#
HClust = AgglomerativeClustering
hc_compPrueba = HClust( distance_threshold =None, n_clusters=3 , linkage='complete')
hc_compPrueba.fit(varied[0])
fig , ax = plt.subplots(figsize =(8 ,8))
ax.scatter(varied[0][:,0], varied[0][:,1], c=hc_compPrueba.labels_)
ax.set_title("Agglomerative Clustering Results");