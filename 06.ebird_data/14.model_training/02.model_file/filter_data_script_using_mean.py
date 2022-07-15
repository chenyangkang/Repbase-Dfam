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
def filter_data(qresult, species):
    #### That's just for some checks.
    # del qresult['species_count']
    del qresult['ensemble_index']

    #### Some date time operation. Generate DOY value.
    qresult['observation_date'] = pd.to_datetime(qresult['observation_date'])
    qresult.insert(10,'DOY',qresult['observation_date'].dt.dayofyear)
    del qresult['observation_date']
    qresult.insert(2,'time_observation_started_minute_of_day',[i.hour*60+i.minute for i in qresult['time_observation_started']])
    del qresult['time_observation_started']

    ### transform the protocal "Travaling"... into dummies
    dummy = pd.get_dummies(qresult.protocol_type)
    qresult.insert(3,dummy.columns[0],dummy.iloc[:,0])
    qresult.insert(3,dummy.columns[1],dummy.iloc[:,1])
    qresult.insert(3,dummy.columns[2],dummy.iloc[:,2])

    ### fill effort distance with -1 and add a column to indicate that the value is missing.
    del qresult['protocol_type']
    qresult.insert(7,"effort_distance_km_missing",[0 if i>0 else 1 for i in qresult.effort_distance_km])


    ######################## Inputation using decisiontree! ###################
    fillna_data = qresult[['duration_minutes','time_observation_started_minute_of_day',\
             'Traveling','Stationary','Area',\
             'effort_distance_km','number_observers','DOY','obsvr_species_count']]

#     from sklearn.experimental import enable_iterative_imputer  
    from sklearn.impute import SimpleImputer
#     from sklearn.ensemble import RandomForestRegressor
#     from sklearn.tree import DecisionTreeRegressor

    del qresult['species_count']
    del qresult['group_identifier']
    del qresult['observer_id'] 
    del qresult['country']

#     imputer = IterativeImputer(missing_values=np.nan, add_indicator=True,
#                               random_state=42, estimator = DecisionTreeRegressor(random_state=42),
#                               max_iter=5)
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    imputer.fit(fillna_data)
    filled_data = imputer.transform(fillna_data)

    qresult[['duration_minutes','time_observation_started_minute_of_day',\
             'Traveling','Stationary','Area',\
             'effort_distance_km','number_observers','DOY','obsvr_species_count']]=filled_data[:,0:9]


    qresult['effort_distance_km'] = np.where(qresult['effort_distance_km']>0, qresult['effort_distance_km'], -1)
    qresult=qresult.fillna(0)  ####### REMEMBER TO CHANGE THIS IF ORDER/FEATURE CHANGE!!!

    #### Del those useless strings column

    qresult[f'{species.replace(" ","_")}_indiv_count'][qresult[f'{species.replace(" ","_")}_indiv_count'].isnull()]=0 
    
    #### select only duration < 3h
    qresult = qresult[qresult['duration_minutes']<=180]
    
    return qresult

