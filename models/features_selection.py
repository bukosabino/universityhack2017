# -*- coding: utf-8 -*-

"""
Seleccionado de características.

Lanzaremos un ensemble como ExtraTreesClassifier con todas las columnas, para
entender a cuáles de ellas da más importancia el ensemble. Luego, podemos jugar
con el umbral para quedarnos con aproximadamente las 40 más importantes, que
serán las que inyectemos al modelo a la hora de predecir.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing

#train = pd.read_hdf('../data/train_final.h5')
train = pd.read_csv('../data/train_final.csv', sep=',')

# Labels encoding
le = preprocessing.LabelEncoder()
le.fit(train['y'].unique())
train['y'] = le.transform(train['y'])

# Seleccionamos todas las características
excl = ['ID', 'y']
cols = [c for c in train.columns if c not in excl]

# Train
rf = ExtraTreesClassifier(n_estimators=30, criterion="entropy", bootstrap=True, max_depth=20 , n_jobs=-1, random_state=17)
rf.fit(train[cols], train["y"])

cols = train.drop(["ID","y"] , axis=1).columns[rf.feature_importances_ > 0.0016]

print(u"Características seleccionadas:")

print(cols)
