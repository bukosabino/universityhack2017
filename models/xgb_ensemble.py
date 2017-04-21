# -*- coding: utf-8 -*-

"""
Modelo XGBoost


Resultados:

Precisión:          50.88 %
Coeficiente kappa:  46.06 %
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn import preprocessing

train = pd.read_csv('../data/train_final.csv', sep=',')
val = pd.read_csv('../data/validacion_final.csv', sep=',')

# Labels encoding
le = preprocessing.LabelEncoder()
le.fit(np.concatenate([train['y'].unique(), val["y"].unique()]))

train['y'] = le.transform(train['y'])
val["y"] = le.transform(val["y"])

# Características ya seleccionadas
cols = [u'SD1', u'SD2', u'SD3', u'SD4', u'SD5', u'P4', u'P5', u'P10', u'P11',
       u'P13', u'P15', u'P18', u'P21', u'P30', u'P37', u'P45', u'P48', u'P49',
       u'P54', u'P57', u'P60', u'P64', u'P69', u'P71', u'P72', u'P73', u'P74',
       u'P76', u'P77', u'P78', u'P79', u'P80', u'P82', u'P83', u'P85', u'P90',
       u'P91', u'P92', u'P93']

# Entrenamiento
xgmat_train = xgb.DMatrix(train[cols], label=train["y"])

params_xgb = {'objective'        : 'multi:softmax',
              'num_class'        : 94,
              'eta'              : 0.05,
              'subsample'        : 0.8,
              'tree_method'      : 'hist',
              'grow_policy'      : 'depthwise',
              'seed'             : 17,
              'max_depth'        : 10,
              'silent'           : True,
}

n_round = 50

xgb_train = xgb.train(params_xgb,
                 xgmat_train,
                 num_boost_round=n_round,
                 verbose_eval=False).__copy__()

# Predicción
xgmat_test = xgb.DMatrix(val[cols])
ypred = xgb_train.predict(xgmat_test)

# Métricas
acc = accuracy_score(val.y.values, ypred)
ck = cohen_kappa_score(val.y.values, ypred)
print('Resultados para xgboost:')
print('Precisión: {}'.format(acc))
print('Coeficiente kappa: {}'.format(ck))
