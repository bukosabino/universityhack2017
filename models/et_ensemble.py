# -*- coding: utf-8 -*-

"""
Modelo ExtraTrees


Resultados:

Precisión:          48.61 %
Coeficiente kappa:  43.5 %

"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn import preprocessing

#train = pd.read_hdf('../data/train_final.h5')
#val = pd.read_hdf('../data/validacion_final.h5')
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
et = ExtraTreesClassifier(n_estimators=40, criterion="entropy", bootstrap=True, max_depth=20 , n_jobs=-1, random_state=17)
et.fit(train[cols], train["y"])

# Predicción
ypred = et.predict(val[cols])

del et

# Métricas
acc = accuracy_score(val.y.values, ypred)
ck = cohen_kappa_score(val.y.values, ypred)
print('Resultados para ExtraTrees:')
print('Precisión: {}'.format(acc))
print('Coeficiente kappa: {}'.format(ck))
