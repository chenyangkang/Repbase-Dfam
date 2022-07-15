import sys
import pandas as pd
import numpy as np
import numpy
import matplotlib.pyplot as plt
import math
import psycopg2
from psycopg2 import Error
import os
import numpy as np
import warnings
import pickle
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score,recall_score, precision_score, roc_auc_score, confusion_matrix, cohen_kappa_score
from pygam import LinearGAM 
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.inspection import partial_dependence
import time
from sklearn.preprocessing import MinMaxScaler
def GBDT2_model_training(X_train_for_GBDT2, X_test_for_GBDT2, y_train_for_GBDT2, weights_for_GBDT2):
    # ### Poisson distributed HistGradientBoostingRegressor
    # GBDT2 = HistGradientBoostingRegressor(loss = 'poisson',
    #                                 max_depth=3,
    #                                 learning_rate=0.01,
    #                                 min_samples_leaf=5)
    ### GradientBoostingRegressor
    GBDT2 = RandomForestRegressor(max_depth=10,n_estimators=2000, criterion="poisson")
    
#                                     min_samples_leaf=3,

    GBDT2.fit(X_train_for_GBDT2, y_train_for_GBDT2, sample_weight=weights_for_GBDT2)
#     , sample_weight=weights_for_GBDT2
    abundance_pred_train = GBDT2.predict(X_train_for_GBDT2)
    abundance_pred_test = GBDT2.predict(X_test_for_GBDT2)
#     abundance_pred_train = abundance_pred_train * GBDT_pred_train[train_used_index]
#     abundance_pred_test = abundance_pred_test * GBDT_pred_test[test_used_index]
    return GBDT2, abundance_pred_train, abundance_pred_test
