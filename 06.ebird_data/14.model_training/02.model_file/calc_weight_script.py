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
def calc_sample_weights(y):
    #### Define the sample weights during the boosting training
    #### The more indiv count, the higher weight that point has.
    #### We need to keep the sum of weight the same for each class
    zero_weight = float(1/(np.sum(y==0)))
    # non_zero_weight = float(1/np.sum((y[y>0]+1)**(1/2)))
    # weights = [(zero_weight if i==0 else ((float(i)+1)**(1/2)*non_zero_weight)) \
    #            for i in y.values.flatten().tolist()]
    non_zero_weight = float(1/np.sum((y[y>0]+1)))
    weights = [(zero_weight if i==0 else ((float(i)+1)*non_zero_weight)) \
               for i in y.values.flatten().tolist()]
    return weights


