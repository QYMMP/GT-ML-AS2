from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import sklearn.model_selection as ms
import numpy as np
import os


OUTPUT_DIRECTORY = './output'

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
if not os.path.exists(OUTPUT_DIRECTORY + '/images'):
    os.makedirs(OUTPUT_DIRECTORY + '/images')
subdirs = ['NN_OUTPUT', 'CONTPEAK', 'KNAPSACK', 'TSP']
for subdir in subdirs:
    if not os.path.exists('{}/{}'.format(OUTPUT_DIRECTORY, subdir)):
        os.makedirs('{}/{}'.format(OUTPUT_DIRECTORY, subdir))
    if not os.path.exists('{}/images/{}'.format(OUTPUT_DIRECTORY, subdir)):
        os.makedirs('{}/images/{}'.format(OUTPUT_DIRECTORY, subdir))

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:23:40 2017

@author: JTay
"""


df = pd.read_csv('datasets/tic-tac-toe.data', header=None)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# X = X.apply(LabelEncoder().fit_transform)

oe = OrdinalEncoder().fit(X)
print(oe.categories_)
X = oe.transform(X)
y = LabelEncoder().fit_transform(y)
X2, tstX, Y2, y_test = ms.train_test_split(
    X, y, test_size=.2, random_state=1)
trgX, valX, y_train, y_val = ms.train_test_split(
    X2, Y2, test_size=.2, random_state=1)

# pipe = Pipeline([('Scale', StandardScaler()),
#                  ('Cull1', SelectFromModel(RandomForestClassifier(
#                      random_state=1), threshold='median')),
#                  ('Cull2', SelectFromModel(RandomForestClassifier(
#                      random_state=2), threshold='median')),
#                  ('Cull3', SelectFromModel(RandomForestClassifier(
#                      random_state=3), threshold='median')),
#                  ('Cull4', SelectFromModel(RandomForestClassifier(random_state=4), threshold='median')), ])
# trgX = pipe.fit_transform(X_train, y_train)
trgY = np.atleast_2d(y_train).T
# tstX = pipe.transform(X_test)
tstY = np.atleast_2d(y_test).T
valY = np.atleast_2d(y_val).T
# trgX, valX, trgY, valY = ms.train_test_split(
#     trgX, trgY, test_size=.2, random_state=1)
tst = pd.DataFrame(np.hstack((tstX, tstY)))
trg = pd.DataFrame(np.hstack((trgX, trgY)))
val = pd.DataFrame(np.hstack((valX, valY)))
tst.to_csv('datasets/tic-tac-toe_test.csv', index=False, header=False)
trg.to_csv('datasets/tic-tac-toe_trg.csv', index=False, header=False)
val.to_csv('datasets/tic-tac-toe_val.csv', index=False, header=False)
