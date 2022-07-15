
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
def GBDT1_training(X_train, X_test, y_train_binary, weights, y_test_binary=None):
    #### Binary (presence or not) Model training using GradientBoostingClassifier
    #### the loss function of GradientBoostingClassifier is ‘log_loss’ 
    #### refers to binomial and multinomial deviance
    GBDT = RandomForestClassifier(n_estimators=1000,
                                    max_depth=5)

    ### Training ########
    # print('########### Model Score Result ###############')
    GBDT.fit(X_train, y_train_binary, sample_weight=weights)
    GBDT_pred_train = GBDT.predict_proba(X_train)[:,1]
    GBDT_pred_test = GBDT.predict_proba(X_test)[:,1]
    return GBDT, GBDT_pred_train, GBDT_pred_test


def get_and_print_GBDT1_metrics(GBDT, y_train_binary, y_test_binary, GBDT_pred_train, GBDT_pred_test, X_train):
    print(' ')
    print('######## Model Metrics on the GBDT Classifier step #######')
    print('#### Score on training set ####')
    auc_training = roc_auc_score(y_train_binary, np.where(GBDT_pred_train>0.5, 1, 0).flatten())
    recall_training = recall_score(y_train_binary, np.where(GBDT_pred_train>0.5, 1, 0))
    precision_training = precision_score(y_train_binary, np.where(GBDT_pred_train>0.5, 1, 0))
    cohen_kappa_score_training = cohen_kappa_score(y_train_binary, np.where(GBDT_pred_train>0.5, 1, 0))
    print(f'auc for training scroe: {auc_training}')
    print(f'recall_score: {recall_training}')
    print(f'precision_score: {precision_training}')
    print(f'cohen_kappa_score: {cohen_kappa_score_training}')
    print(' ')
    print('#### Score on testing set ####')
    acc_testing = accuracy_score(y_test_binary, np.where(GBDT_pred_test>0.5, 1, 0))
    recall_testing = recall_score(y_test_binary, np.where(GBDT_pred_test>0.5, 1, 0))
    precision_testing = precision_score(y_test_binary, np.where(GBDT_pred_test>0.5, 1, 0))
    auc_testing = roc_auc_score(y_test_binary, np.where(GBDT_pred_test>0.5, 1, 0).flatten())
    f1_testing = f1_score(y_test_binary, np.where(GBDT_pred_test>0.5, 1, 0).flatten())
    cohen_kappa_score_testing = cohen_kappa_score(y_test_binary, np.where(GBDT_pred_test>0.5, 1, 0))
    confusion_matrix_testing = confusion_matrix(y_test_binary, np.where(GBDT_pred_test>0.5, 1, 0), labels=[0,1])
    print(f'accuracy score: {acc_testing}')
    print(f'recall_score: {recall_testing}')
    print(f'precision_score: {precision_testing}')
    print(f'auc: {auc_testing}')
    print(f'f1_score: {f1_testing}')
    print(f'cohen_kappa_score: {cohen_kappa_score_testing}')
    print(f'confustion_matrix: \npred0 pred1\n{confusion_matrix_testing}')
    print(' ')
    print("############## Feature Importance ranking (Top 10) ##############")
    occurance_feature_importance = []
    for index,i in enumerate(sorted([(a,b) for a,b in zip(X_train.columns,GBDT.feature_importances_)], key=lambda x:x[1], reverse=True)):
        if index<10:
            print(i)
        occurance_feature_importance.append(i)
    
    return occurance_feature_importance,\
                (auc_training, recall_training, precision_training, cohen_kappa_score_training,\
                 acc_testing, recall_testing, precision_testing, auc_testing, f1_testing, cohen_kappa_score_testing,\
                 confusion_matrix_testing)



