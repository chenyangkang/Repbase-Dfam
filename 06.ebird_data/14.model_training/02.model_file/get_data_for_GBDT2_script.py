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
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score,recall_score, precision_score, roc_auc_score, confusion_matrix, cohen_kappa_score, mean_poisson_deviance
from pygam import LinearGAM 
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.inspection import partial_dependence
import time
from sklearn.preprocessing import MinMaxScaler
def get_data_for_GBDT2(X_train, X_test, y_train, y_test, gam_pred_train, gam_pred_test, weights):
    ##### Only these data is used in the following analysis
    
    scaler_minmax = MinMaxScaler().fit(gam_pred_train.reshape(-1,1))
    gam_pred_train = scaler_minmax.transform(gam_pred_train.reshape(-1,1))
    gam_pred_test = scaler_minmax.transform(gam_pred_test.reshape(-1,1))
    train_used_index = np.where(gam_pred_train.flatten()>0.5)[0]
    if len(train_used_index) == 0:
        return "Killed_for_GAM_predict_all_train_data_absence"
    
    X_train_for_GBDT2 = X_train.iloc[[i for i in train_used_index],:]
    y_train_for_GBDT2 = y_train.values[[i for i in train_used_index]].reshape(-1,1)
    weights_for_GBDT2 = np.array(weights)[[i for i in train_used_index]]

    test_used_index = np.where(gam_pred_test.flatten()>0.5)[0]
    if len(test_used_index) == 0:
        return "Killed_for_GAM_predict_all_test_data_absence"
    
    X_test_for_GBDT2 = X_test.iloc[[i for i in test_used_index],:]
    y_test_for_GBDT2 = y_test.values[[i for i in test_used_index]].reshape(-1,1)
    
    return train_used_index, test_used_index, X_train_for_GBDT2, X_test_for_GBDT2, y_train_for_GBDT2, y_test_for_GBDT2, weights_for_GBDT2, scaler_minmax



