
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
def stixel_level_filtering(qresult, stixel_id, species):
    ### First take a toy set. Say one stixel
    toy = qresult[qresult['unique_stixel_id']==stixel_id]

    ### random sampling one checklist in one day for each locality. To abvoid over investigation of hot spot.
    toy = toy.sample(replace=False, frac=1).\
                    groupby(['locality_id','DOY']).\
                                first().\
                                        reset_index(drop=False)

    # Kill this model/stixel if the checklist count is less than 30
    if len(toy)<30:
        return "Model terminated because checklist count<30 for this stixel", np.nan, np.nan


#     toy['obsvr_species_count'] = toy['obsvr_species_count'].rank(ascending=False)

    del toy['locality_id']
    del toy['sampling_event_identifier']
    del toy['unique_stixel_id']

    toy = toy.rename(columns={f"{species.replace(' ','_')}_indiv_count":'indiv_count'})

    y = toy['indiv_count']
    X = toy[[i for i in toy.columns if not i=='indiv_count']]
    del toy
    
    ################### Train test split before anything ##########################################
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

    ##################################### case control spatial temporal sampling ###################################
    def case_control_st_sample(X, y):
        training_data = pd.concat([X.reset_index(drop=True),y.reset_index(drop=True)], axis=1)

        present_set = training_data[training_data['indiv_count']>0]
        print(f'before st balanced sample, present set shape: {present_set.shape}')
        absent_set = training_data[training_data['indiv_count']==0]
        print(f'before st balanced sample, absent set shape: {absent_set.shape}')
        
        if absent_set.shape[0]==0:
            return "Killed_for_all_occupied, using mean abundance", np.nan, np.nan
        if present_set.shape[0]==0:
            return "Killed_for_all_absence, not occupied", np.nan, np.nan

        def st_sample(df):
            present_set_long = np.arange(df['longitude'].min()-0.05,df['longitude'].max()+0.05,0.05)
            present_set_lat = np.arange(df['latitude'].min()-0.05,df['latitude'].max()+0.05,0.05)
            present_set_day = np.arange(df['DOY'].min()-1,df['DOY'].max()+1,1)
            
            long_membership = np.digitize(df['longitude'], present_set_long)
            lat_membership = np.digitize(df['latitude'], present_set_lat)
            time_membership = np.digitize(df['DOY'], present_set_day)
            unique_membership_code = [str(a)+"_"+str(b)+"_"+str(c) for a,b,c in zip(long_membership, lat_membership, time_membership)]
            df['unique_membership_code'] = unique_membership_code
            df = df.sample(frac=1,replace=False).groupby(['unique_membership_code']).first().reset_index(drop=True)
            return df

        ######## spatial temporal sampling. 0.1 degree * 0.1 degree *1 week
        present_set = st_sample(present_set)
        print(f'after st balanced sample, present set shape: {present_set.shape}')
        absent_set = st_sample(absent_set)
        print(f'after st balanced sample, absent set shape: {absent_set.shape}')

        detection_rate_before_oversampling = len(present_set)/(len(present_set)+len(absent_set))
        print(f'detection rate before oversampling: {detection_rate_before_oversampling}')

        if detection_rate_before_oversampling>0.5:
            print('no need to adjust the case control sampling')
            new_df = pd.concat([present_set.reset_index(drop=True), absent_set.reset_index(drop=True)], axis=0)
            return new_df.iloc[:,0:-1], new_df.iloc[:,-1], detection_rate_before_oversampling
        else:
            oversample_fraction = (len(absent_set)-len(present_set))/len(present_set)

            additional_present_set = present_set.sample(frac=oversample_fraction, replace=True)

#             ##### add jitter
#             columns_std = present_set.iloc[:,13:-1].std()
#             additional_present_set.iloc[:,13:-1] += np.array([np.random.normal(loc=0, scale=columns_std/5) for i in range(len(additional_present_set))])
#             additional_present_set.iloc[:,13:-1]=np.where(additional_present_set.iloc[:,13:-1]>0, additional_present_set.iloc[:,13:-1], 0)

            new_present_set = pd.concat([present_set, additional_present_set],axis=0)

        new_df = pd.concat([new_present_set.reset_index(drop=True), absent_set.reset_index(drop=True)], axis=0)

        return new_df.iloc[:,0:-1], new_df.iloc[:,-1], detection_rate_before_oversampling

    try:
        X_train, y_train, detection_rate_before_oversampling = case_control_st_sample(X_train, y_train)
        if isinstance(X_train, str):
            return X_train, np.nan, np.nan
    except Exception as e:
        print(f"Model terminated for: {e}")
        return str(e), np.nan, np.nan


    del X_train['longitude']
    del X_train['latitude']
    del X_test['longitude']
    del X_test['latitude']
    
    X_columns = X_train.columns
    scaler = MinMaxScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train))
    X_train.columns = X_columns
    X_test = pd.DataFrame(scaler.transform(X_test))
    X_test.columns = X_columns
    
    return (X_train, X_test, y_train, y_test), scaler, detection_rate_before_oversampling
