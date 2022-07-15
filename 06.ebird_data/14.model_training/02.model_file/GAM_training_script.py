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
def GAM_training(X_train, X_test, GBDT_pred_train, weights):
    try:
        gam = LinearGAM(constraints='monotonic_inc',max_iter=100).fit(X_train.values, GBDT_pred_train, weights=weights)
    except Exception as e:
        print(e)
        return "GAM_not_converged"
        
    #### predict the result using both training data and testing data
    gam_pred_train = gam.predict(X_train.values).reshape(-1,1)
    gam_pred_test = gam.predict(X_test.values).reshape(-1,1)
    
    ###### saving the gam model stats result
    gam_stats_dict={}
    print(' ')
    print("######## GAM model stats ######## ")
    for i in ['AIC','GCV','pseudo_r2','edof','n_samples']:
        gam_stats_dict[i]=gam.statistics_[i]
        print((i, gam.statistics_[i]))

    return gam, gam_pred_train, gam_pred_test, gam_stats_dict




