# -*- coding: utf-8 -*-
"""

@author: Joaco
"""

#Este archivo contiene diversos modelos de regresión desarrollados

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error


#%% CARGO DATASET PARA REGRESIÓN

#El dataset contiene información sobre distintos autos. Entre las características detalladas, se encuentra por ejemplo su gasto de combustible y su peso. 
df_mpg = pd.read_csv("auto-mpg.xls")

#La variable a predecir será el gasto en combustible de un auto en millas por galón (mpg).

#%% VISUALIZAMOS LA RELACIÓN ENTRE LAS DISTINTAS VARIABLES

#SCATTER MPG - WEIGHT
plt.figure()
plt.scatter(df_mpg['weight'], df_mpg['mpg'], alpha=0.3)
plt.title("MPG vs Weight")
plt.xlabel("Weight")
plt.ylabel("MPG")

#SCATTER MPG - DISPLACEMENT
plt.figure()
plt.scatter(df_mpg['displacement'], df_mpg['mpg'], alpha = 0.3)
plt.title("MPG vs Displacement")
plt.xlabel("Displacement")
plt.ylabel("MPG")

#SCATTER MPG - ACCELERATION
plt.figure()
plt.scatter(df_mpg['acceleration'], df_mpg['mpg'], alpha = 0.3)
plt.title("MPG vs Acceleration")
plt.xlabel("Acceleration")
plt.ylabel("MPG")

#SCATTER MPG - HORSEPOWER
plt.figure()
plt.scatter(df_mpg['horsepower'], df_mpg['mpg'], alpha = 0.3)
plt.title("MPG vs Horsepower")
plt.xlabel("Horsepower")
plt.ylabel("MPG")

#%% SELECCIONO LAS VARIABLES A UTILIZAR PARA MODELO KNN, HAGO REESCALONAMIENTOS Y SEPARO CONJUNTOS DE TRAIN Y TEST

#Usaré los atributos 'weight', 'horsepower', 'displacement' para determinar el 'wpg' pues por los gráficos se puede vislumbrar cierta relación.
#No obstante, se requiere reescalar los datos.
#Esto se debe a que KNN es un algoritmo que se basa en distancias.
#Luego, si los valores que se presentan en cada atributo de la tabla presentan números muy distintos, entonces el algoritmo priorizará uno sobre el otro.

#Selecciono los parámetros a utilizar para hacer las predicciones
X = df_mpg[['displacement', 'horsepower', 'weight']].copy()
#Selecciono la variable a predecir
y = df_mpg['mpg']

#Hago los reescalonamientos pertinentes
X['weight'] = (X['weight']-min(X['weight']))/(max(X['weight'])-min(X['weight']))
X['displacement'] = (X['displacement']-min(X['displacement']))/(max(X['displacement'])-min(X['displacement']))
X['horsepower'] = (X['horsepower']-min(X['horsepower']))/(max(X['horsepower'])-min(X['horsepower']))

#Separo en conjuntos de desarrollo y testeo
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.25, shuffle = True, random_state=12)

#El conjunto de testeo será reservado para, tras la selección del modelo que mejor rendimiento posee, evaluar la performance del mismo.

#%% ENTRENO Y EVALUO DISTINTOS MODELOS KNN CON DISTINTOS K

#Habiendo visualizado la relación entre las distintas variables, desarrollo modelos KNN de regresión.
#La técnica K-Folding me servirá para vislumbrar cuál es la combinación de parámetros que mejor predice los datos.
#Utilizaré el Error Cuadrático Medio (MSE) como medida a tener en cuenta para la selección de un modelo.

valores_k = [1, 3, 5, 10, 20, 30, 50] #selecciono diversas cantidades de vecinos para probar varios modelos.
nsplits = 3 #tengo 294 datos que los divido en tres folds de 98 cada uno.
kf = KFold(n_splits=nsplits)
resultados = np.zeros((nsplits, len(valores_k)))

for i, (train_index, test_index) in enumerate(kf.split(X_dev)):

    kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
    kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]
    
    for j, k in enumerate(valores_k):
        
        modelo = KNeighborsRegressor(n_neighbors=k)
        modelo.fit(kf_X_train, kf_y_train)
        pred = modelo.predict(kf_X_test)
        score = mean_squared_error(kf_y_test,pred)
        
        resultados[i, j] = score
        
#Evaluando los resultados para los distintos modelos en cada fold, determino que los modelos con k=3 y k=5 son los más óptimos.
#Esto se debe a que no solo presentan MSE bajos en relación a los distintos modelos, sino que son los que mayor estabilidad de MSE presentan en sus distintos folds.

#Me inclino finalmente, por mínima diferencia de promedio en los folds, con k=5.
#Evalúo el rendimiento en el conjunto de testeo.
modelo_knn_elegido_autos = KNeighborsRegressor(n_neighbors=20)
modelo_knn_elegido_autos.fit(X_dev, y_dev)
y_pred = modelo.predict(X_test)
print(f'El MSE del modelo elegido es: {mean_squared_error(y_test, y_pred)}')

#Habilito sector para borrar variables auxiliares. Puede comentarse si así se desea.
del valores_k, nsplits, kf, i, train_index, test_index, kf_X_train, kf_X_test, kf_y_test, kf_y_train, j, k, modelo, pred, score

#%% ENTRENO Y EVALUO DISTINTOS MODELOS DE REGRESIÓN LINEAL SIMPLE CON DISTINTOS ATRIBUTOS

#Entreno modelos del estilo SLR en función de los distintos atributos seleccionados.

#Separo en conjuntos de train y test
x_train, x_test, y_train, y_test = train_test_split(df_mpg[['weight', 'displacement', 'horsepower']], df_mpg['mpg'], test_size=0.25, shuffle = True, random_state=12)

#---Desarrollo el modelo con la variable peso (WEIGHT)--#
modelo_lineal_weight = LinearRegression()
modelo_lineal_weight.fit(x_train[['weight']], y_train)

y_pred = modelo_lineal_weight.predict(x_test[['weight']])
print(f'El MSE del modelo SLR usando WEIGHT como variable X es: {mean_squared_error(y_test, y_pred)}')

#Grafico la recta creada y los datos de mi tabla
X = np.linspace(min(df_mpg['weight']), max(df_mpg['weight']))
Y = modelo_lineal_weight.intercept_ + modelo_lineal_weight.coef_*X

plt.scatter(df_mpg['weight'], df_mpg['mpg'])
plt.plot(X, Y, 'black')
plt.title('Modelo de Regresión Lineal Simple: WEIGHT vs MPG')
plt.xlabel('Peso')
plt.ylabel('MPG')
plt.show()


#--Desarrollo el modelo con la variable DISPLACEMENT---#
modelo_lineal_displacement = LinearRegression()
modelo_lineal_displacement.fit(x_train[['displacement']], y_train)

y_pred = modelo_lineal_displacement.predict(x_test[['displacement']])
print(f'El MSE del modelo SLR usando DISPLACEMENT como variable X es: {mean_squared_error(y_test, y_pred)}')

#Grafico la recta creada y los datos de mi tabla
X = np.linspace(min(df_mpg['displacement']), max(df_mpg['displacement']))
Y = modelo_lineal_displacement.intercept_ + modelo_lineal_displacement.coef_*X

plt.scatter(df_mpg['displacement'], df_mpg['mpg'])
plt.plot(X, Y, 'black')
plt.title('Modelo de Regresión Lineal Simple: DISPLACEMENT vs MPG')
plt.xlabel('Displacement')
plt.ylabel('MPG')
plt.show()


#--Desarrollo el modelo con la variable HORSEPOWER---#
modelo_lineal_horsepower = LinearRegression()
modelo_lineal_horsepower.fit(x_train[['horsepower']], y_train)

y_pred = modelo_lineal_horsepower.predict(x_test[['horsepower']])
print(f'El MSE del modelo SLR usando HORSEPOWER como variable X es: {mean_squared_error(y_test, y_pred)}')

#Grafico la recta creada y los datos de mi tabla
X = np.linspace(min(df_mpg['horsepower']), max(df_mpg['horsepower']))
Y = modelo_lineal_horsepower.intercept_ + modelo_lineal_horsepower.coef_*X

plt.scatter(df_mpg['horsepower'], df_mpg['mpg'])
plt.plot(X, Y, 'black')
plt.title('Modelo de Regresión Lineal Simple: HORSEPOWER vs MPG')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.show()


#La variable WEIGHT, para los modelos SLR, fue la más útil para predecir el MPG de los autos

#%% ENTRENO Y EVALUO DISTINTOS MODELOS DE REGRESIÓN LINEAL MULTIVARIABLE CON DISTINTOS ATRIBUTOS

#En este bloque, me sigue interesando ver modelos de regresión lineal, pero ahora la predicción utiliza múltiples variables.

#--Modelo multivariable usando WEIGHT y DISPLACEMENT--#
modelo_lineal_multivariable_weight_displacement = LinearRegression()
modelo_lineal_multivariable_weight_displacement.fit(x_train[['weight', 'displacement']], y_train)
y_pred = modelo_lineal_multivariable_weight_displacement.predict(x_test[['weight', 'displacement']])
print(f'El MSE del modelo lineal multivariable usando WEIGHT y DISPLACEMENT es: {mean_squared_error(y_test, y_pred)}')


#--Modelo multivariable usando WEIGHT y HORSEPOWER--#
modelo_lineal_multivariable_weight_horsepower = LinearRegression()
modelo_lineal_multivariable_weight_horsepower.fit(x_train[['weight', 'horsepower']], y_train)
y_pred = modelo_lineal_multivariable_weight_horsepower.predict(x_test[['weight', 'horsepower']])
print(f'El MSE del modelo lineal multivariable usando WEIGHT y HORSEPOWER es: {mean_squared_error(y_test, y_pred)}')


#--Modelo multivariable usando DISPLACEMENT y HORSEPOWER--#
modelo_lineal_multivariable_displacement_horsepower = LinearRegression()
modelo_lineal_multivariable_displacement_horsepower.fit(x_train[['displacement', 'horsepower']], y_train)
y_pred = modelo_lineal_multivariable_displacement_horsepower.predict(x_test[['displacement', 'horsepower']])
print(f'El MSE del modelo lineal multivariable usando DISPLACEMENT y HORSEPOWER es: {mean_squared_error(y_test, y_pred)}')


#--Modelo multivariable usando WEIGHT, DISPLACEMENT y HORSEPOWER-- #
modelo_lineal_multivariable = LinearRegression()
modelo_lineal_multivariable.fit(x_train, y_train)
y_pred = modelo_lineal_multivariable.predict(x_test)
print(f'El MSE del modelo lineal multivariable usando WEIGHT, DISPLACEMENT y HORSEPOWER es: {mean_squared_error(y_test, y_pred)}')

#La variables WEIGHT y HORSEPOWER combinadas, para los modelos lineales multivariables, fueron los más útiles para predecir el MPG de los autos
#Esto evidencia que DISPLACEMENT resulta ser, de las tres, la que menor implicancia a con el gasto de combustible posee.

#%% ENTRENO Y EVALUO DISTINTOS MODELOS DE REGRESIÓN CUADRÁTICA Y MULTIVARIABLE CON DISTINTOS ATRIBUTOS

#En este bloque, me interesa ver modelos de regresión cuadráticos, tanto simples como multivariables

#-- Modelo cuadrático simple usando WEIGHT--#
poly_feat_weight = PolynomialFeatures(degree=2).fit_transform(x_train[['weight']])
modelo_cuad_weight = LinearRegression(fit_intercept=False)
modelo_cuad_weight.fit(poly_feat_weight, y_train)
poly_feat_test = PolynomialFeatures(degree=2).fit_transform(x_test[['weight']])
y_pred = modelo_cuad_weight.predict(poly_feat_test)
print(f'El MSE del modelo cuadrático usando WEIGHT es: {mean_squared_error(y_test, y_pred)}')


#-- Modelo cuadrático simple usando DISPLACEMENT--#
poly_feat_displacement = PolynomialFeatures(degree=2).fit_transform(x_train[['displacement']])
modelo_cuad_displacement = LinearRegression(fit_intercept=False)
modelo_cuad_displacement.fit(poly_feat_displacement, y_train)
poly_feat_test = PolynomialFeatures(degree=2).fit_transform(x_test[['displacement']])
y_pred = modelo_cuad_displacement.predict(poly_feat_test)
print(f'El MSE del modelo cuadrático usando DISPLACEMENT es: {mean_squared_error(y_test, y_pred)}')


#-- Modelo cuadrático simple usando HORSEPOWER--#
poly_feat_horsepower = PolynomialFeatures(degree=2).fit_transform(x_train[['horsepower']])
modelo_cuad_horsepower = LinearRegression(fit_intercept=False)
modelo_cuad_horsepower.fit(poly_feat_horsepower, y_train)
poly_feat_test = PolynomialFeatures(degree=2).fit_transform(x_test[['horsepower']])
y_pred = modelo_cuad_horsepower.predict(poly_feat_test)
print(f'El MSE del modelo cuadrático usando HORSEPOWER es: {mean_squared_error(y_test, y_pred)}')


#HORSEPOWER y WEIGHT presentan los mejores resultados; luego, decido armar un modelo cuadrático multivariable que use estos dos atributos
poly_feat_weight_displacement = PolynomialFeatures(degree=2).fit_transform(x_train[['weight', 'horsepower']])
modelo_cuad_weight_displacement = LinearRegression(fit_intercept=False)
modelo_cuad_weight_displacement.fit(poly_feat_weight_displacement, y_train)
poly_feat_test = PolynomialFeatures(degree=2).fit_transform(x_test[['weight', 'horsepower']])
y_pred = modelo_cuad_weight_displacement.predict(poly_feat_test)
print(f'El MSE del modelo cuadrático usando WEIGHT y HORSEPOWER es: {mean_squared_error(y_test, y_pred)}')

#Para finalizar los modelos de regresión, armo un modelo cuadrático multivariable que agarre los tres atributos con los que estuve trabajando
poly_feat = PolynomialFeatures(degree=2).fit_transform(x_train)
modelo_cuad = LinearRegression(fit_intercept=False)
modelo_cuad.fit(poly_feat, y_train)
poly_feat_test = PolynomialFeatures(degree=2).fit_transform(x_test)
y_pred = modelo_cuad.predict(poly_feat_test)
print(f'El MSE del modelo cuadrático usando WEIGHT, DISPLACEMENT y HORSEPOWER es: {mean_squared_error(y_test, y_pred)}')


#Como conclusión, determino que el mejor modelo de regresión desarrollado fue el modelo cuadrático multivariable en el que utilicé las variables WEIGHT y HORSEPOWER
