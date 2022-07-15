
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
class base_model():
    '''
        A base model class combine all the three base model, and store there parameters and result.
        They are:
            GBDT1: used for occurance estimation. Training sample was weighted on the indiv count.
            GAM: for smoothing the GBDT1 output
            GBDT2: for abundance estimation. Using HistGradientBoostingRegressor.
            metircs: contain metrics evaluation on training and testing set.
            occurance_feature_importance
            stixel_id
            ensemble
            species
            year
            species index
            PD_FI: partial_dependence and feature importance
    '''
    def __init__(self, GBDT1, GAM, GBDT2, metrics, occurance_feature_importance,                  ensemble, stixel_id, year, species, species_index, PD_FI, sample_size, scaler, scaler_minmax):
        from scipy.stats import spearmanr
        from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingRegressor, HistGradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score,            recall_score, precision_score, roc_auc_score, confusion_matrix, plot_confusion_matrix,             r2_score, mean_squared_error, explained_variance_score, cohen_kappa_score
        from pygam import LinearGAM 
        from sklearn.preprocessing import MinMaxScaler
        import pandas as pd
        import numpy as np
        
        self.GBDT1 = GBDT1
        self.GAM = GAM
        self.GBDT2 = GBDT2
        self.metrics = metrics
        self.occurance_feature_importance = occurance_feature_importance
        self.stixel_id = stixel_id
        self.ensemble = ensemble
        self.species = species
        self.year = year
        self.species_index = species_index
        self.PD_FI = PD_FI
        self.sample_size = sample_size
        self.scaler = scaler
        self.scaler_minmax = scaler_minmax
        print('All three model and metrics loaded.')
    
    def predict(self, X):
        if not len(X.shape)==2:
            print("Please Check Your X Input Shape.")
        
        X_columns = X.columns
        X = self.scaler.transform(X)
        X = pd.DataFrame(X, columns=X_columns)

        pred_occurance = self.GBDT1.predict_proba(X)[:,1]
        if not self.GAM == None:
            gam_smoothed_pred_occurance = self.GAM.predict(X)
        else:
            gam_smoothed_pred_occurance = pred_occurance
            
        scaled_gam_smoothed_pred_occurance = self.scaler_minmax.transform(gam_smoothed_pred_occurance.reshape(-1,1))
        occupied_index = np.where(scaled_gam_smoothed_pred_occurance>=0.5)[0]
        non_occupied_index = np.where(scaled_gam_smoothed_pred_occurance<0.5)[0]
        X_for_GBDT2 = X.iloc[[i for i in occupied_index],:]
        if not len(X_for_GBDT2)==0:
            relative_abundance = self.GBDT2.predict(X_for_GBDT2)
        else:
            relative_abundance = 0
        
        result = np.array([np.nan] * len(X))
        result[non_occupied_index] = 0
        result[occupied_index] = relative_abundance
        return result.reshape(-1,1), scaled_gam_smoothed_pred_occurance
