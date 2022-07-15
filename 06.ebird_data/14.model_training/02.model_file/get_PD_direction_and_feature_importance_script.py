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
def get_PD_direction_and_feature_importance(model, X):
    out_dict = {}
    for name, FI in zip(X.columns.tolist(), model.feature_importances_.tolist()):
        variable_name = name
        PD = partial_dependence(model, X, variable_name)
        X_PD = PD[1][0].tolist()
        y_PD = PD[0].flatten().tolist()
        max_value_point = X_PD[np.argsort(y_PD)[-1]]
        min_value_point = X_PD[np.argsort(y_PD)[0]]
        direction = LinearRegression().fit(np.array(X_PD).reshape(-1,1), np.array(y_PD).reshape(-1,1)).coef_[0][0]
        out_dict[name]={'partial_dependence_direction':direction, 
                        'feature_importance':FI, 
                        'max_value_point':max_value_point, 
                        'min_value_point':min_value_point}
    
    return out_dict
