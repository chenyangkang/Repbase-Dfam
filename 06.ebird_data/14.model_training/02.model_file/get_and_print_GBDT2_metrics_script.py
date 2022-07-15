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
from sklearn.metrics import accuracy_score, f1_score,recall_score, precision_score, roc_auc_score, confusion_matrix, cohen_kappa_score, mean_poisson_deviance, d2_tweedie_score
from pygam import LinearGAM 
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.inspection import partial_dependence
import time
from sklearn.preprocessing import MinMaxScaler
from calc_weight_script import *
def get_and_print_GBDT2_metrics(y_train_for_GBDT2, y_test_for_GBDT2, abundance_pred_train, abundance_pred_test):
    print(' ')
    print('######## Model Metrics on the GBDT2 step #######')
    print('#### Score on training set ####')
    train_sample_weight=calc_sample_weights(pd.Series(y_train_for_GBDT2.flatten()))
    r2_training = r2_score(y_train_for_GBDT2, abundance_pred_train, sample_weight=train_sample_weight)
    mse_training = mean_squared_error(y_train_for_GBDT2, abundance_pred_train, sample_weight=train_sample_weight)
    explained_variance_training = explained_variance_score(y_train_for_GBDT2, abundance_pred_train, sample_weight=train_sample_weight)
    spearman_rank_correlation_training = spearmanr(y_train_for_GBDT2, abundance_pred_train)
    try:
        d2_tweedie_score_training = d2_tweedie_score(y_train_for_GBDT2, abundance_pred_train, power=1, sample_weight=train_sample_weight)
    except:
        d2_tweedie_score_training = np.nan
    print(f'r2_score: {r2_training}')
    print(f'mean_squared_error: {mse_training}')
    print(f'explained_variance_score: {explained_variance_training}') 
    print(f'Spearman\'s Rank Correlation: coef {spearman_rank_correlation_training[0]}, P-value {spearman_rank_correlation_training[1]}')
    print(f'poisson deviance explained: {d2_tweedie_score_training}')

    
    test_sample_weight=calc_sample_weights(pd.Series(y_test_for_GBDT2.flatten()))
    r2_testing = r2_score(y_test_for_GBDT2, abundance_pred_test, sample_weight=test_sample_weight)
    mse_testing = mean_squared_error(y_test_for_GBDT2, abundance_pred_test, sample_weight=test_sample_weight)
    explained_variance_testing = explained_variance_score(y_test_for_GBDT2, abundance_pred_test, sample_weight=test_sample_weight)
    spearman_rank_correlation_testing = spearmanr(y_test_for_GBDT2, abundance_pred_test)
    try:
        d2_tweedie_score_testing = d2_tweedie_score(y_test_for_GBDT2, abundance_pred_test, power=1, sample_weight=test_sample_weight)
    except:
        d2_tweedie_score_testing = np.nan
    print(' ')
    print('#### Score on testing set ####')
    print(f'r2_score: {r2_testing}')
    print(f'mean_squared_error: {mse_testing}')
    print(f'explained_variance_score: {explained_variance_testing}')
    print(f'Spearman\'s Rank Correlation: coef {spearman_rank_correlation_testing[0]}, P-value {spearman_rank_correlation_testing[1]}')
    print(f'poisson deviance explained: {d2_tweedie_score_testing}')
    return r2_training, mse_training, explained_variance_training, spearman_rank_correlation_training, d2_tweedie_score_training,\
                r2_testing, mse_testing, explained_variance_testing, spearman_rank_correlation_testing, d2_tweedie_score_testing


