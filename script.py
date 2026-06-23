import numpy as np
import pandas as pd
from matplotlib.pyplot import subplot

import statsmodels.api as sm

from statsmodels.stats.outliers_influence \
    import variance_inflation_factor as VIF

from statsmodels.stats.anova import anova_lm

from ISLP import load_data
from ISLP.models import (ModelSpec as MS, 
                         summarize, 
                         poly)

#Carga de Datos

Boston = load_data("Boston")
print(Boston.columns)
sample=Boston[:3]
print(sample)

#Se construirá un modelo de regresión lineal para predecir medv (median house value) usando 13 variables
# predictoras. Comenzamos usando un solo predictor (lstat- porcentaje de propietarios con bajo nivel so
# cioeconómico). Podemos construir la matriz a mano: 

X = pd.DataFrame({'intercept': np.ones(Boston.shape[0]),
                  'lstat': Boston['lstat']})
print(X[:4])

#Fiteamos el modelo:

y = Boston['medv']
model = sm.OLS(y, X)
results = model.fit()

print(summarize(results))