
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
def print_GAM_metrics(y_train_binary, y_test_binary, gam_pred_train, gam_pred_test):
    print(' ')
    print('######## Model Metrics on the GAM step #######')
    print('#### Score on training set ####')
    print(f'auc for training scroe: {roc_auc_score(y_train_binary, np.where(gam_pred_train>0.5, 1, 0).flatten())}')
    print(f'recall_score: {recall_score(y_train_binary, np.where(gam_pred_train>0.5, 1, 0))}')
    print(f'precision_score: {precision_score(y_train_binary, np.where(gam_pred_train>0.5, 1, 0))}')
    print(f'cohen_kappa_score: {cohen_kappa_score(y_train_binary, np.where(gam_pred_train>0.5, 1, 0))}')
    print(' ')
    print('#### Score on testing set ####')
    print(f'accuracy score: {accuracy_score(y_test_binary, np.where(gam_pred_test>0.5, 1, 0))}')
    print(f'recall_score: {recall_score(y_test_binary, np.where(gam_pred_test>0.5, 1, 0))}')
    print(f'precision_score: {precision_score(y_test_binary, np.where(gam_pred_test>0.5, 1, 0))}')
    print(f'auc: {roc_auc_score(y_test_binary, np.where(gam_pred_test>0.5, 1, 0).flatten())}')
    print(f'f1_score: {f1_score(y_test_binary, np.where(gam_pred_test>0.5, 1, 0).flatten())}')
    print(f'cohen_kappa_score: {cohen_kappa_score(y_test_binary, np.where(gam_pred_test>0.5, 1, 0))}')
    print(f'confustion_matrix: \npred0 pred1\n{confusion_matrix(y_test_binary, np.where(gam_pred_test>0.5, 1, 0), labels=[0,1])}')
    
