#!/usr/bin/env python
# coding: utf-8

# ## This is the loop version of model_training.ipynb
# 

# In[1]:


### import libraries

import sys
if not len(sys.argv) == 5:
    print(len(sys.argv))
    print(sys.argv)
    print("   ")
    print("Look!!!")
    print("Usage: python script.py 1) year 2) user_defined_species_index 3) species_name 4) ensemble_index")
    print("   ")
    raise
    
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

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",None)

np.random.seed(42)


# In[2]:


### We will first go through year 2002, ensemble 0
# year = 2004
# ensemble_index = 72
# user_defined_species_index = 0
# species = "Wood Thrush"
# model_saving_dir = "/beegfs/store4/chenyangkang/06.ebird_data/14.model_training/01.trained_model"


# In[3]:

year = int(sys.argv[1])
ensemble_index = int(sys.argv[4])
user_defined_species_index = int(sys.argv[2])

### notice: you should pass a name without gap but _. I will generate the _ for the later analysis. Because sys.argv donot allow gaps.
species = str(sys.argv[3]).replace("_"," ")
model_saving_dir = "/beegfs/store4/chenyangkang/06.ebird_data/14.model_training/01.trained_model"


# In[4]:


if not os.path.exists(model_saving_dir):
    print(f'Out_dir {model_saving_dir} not exists! Please Check it.')
if not os.path.exists(f"{model_saving_dir}/{str(user_defined_species_index).zfill(5)}.{species.replace(' ','_')}"):
    os.mkdir(f"{model_saving_dir}/{str(user_defined_species_index).zfill(5)}.{species.replace(' ','_')}")
    print(f'''You should first make a folder named by the userdefined species index and the species name. 
            For example: \"00002.Wood_Thrush\". Remember to zfill the index to ten thousands place! ''')
    print("Now I have made it for you. Not need to do that.")


# ## 1) Request the data

# In[5]:


connect = psycopg2.connect('dbname=ebird user=chenyangkang host=fat01 port=5432')
cur = connect.cursor()
cur.execute(f"""
    SELECT
    DISTINCT
        checklist_origin_{year}.sampling_event_identifier,
        checklist_origin_{year}.duration_minutes,                    --- effort_predictors_1
        checklist_origin_{year}.protocol_type,                       --- effort_predictors_2
        checklist_origin_{year}.effort_distance_km,                  --- effort_predictors_3
        checklist_origin_{year}.number_observers,                    --- effort_predictors_4
        ---- checklist calibration index,                            --- effort_predictors_5 
        checklist_origin_{year}.time_observation_started,            --- variation_availability_detection_predictors_1
        checklist_origin_{year}.observation_date,                    --- variation_availability_detection_predictors_2 should be DOY later
        checklist_origin_{year}.country,                             --- variation_availability_detection_predictors_3
        checklist_origin_{year}.locality_id,               --- used to filter checklists. Dropped later. 
        checklist_origin_{year}.observer_id,               --- used to check observer expertise. Later dropped.
        checklist_origin_{year}.longitude,
        checklist_origin_{year}.latitude,
        observer_expertise_cumulative_{year}.species_count AS obsvr_species_count,                    --- observer_expertise_1
        --- observer_expertise_cumulative_{year}.checklist_count AS obsvr_checklist_count,                    --- observer_expertise_2
        --- observer_expertise_cumulative_{year}.mean_species_count_each_checklist AS obsvr_species_record_rate,        --- observer_expertise_2
        checklist_origin_{year}.group_identifier,          --- used for checking. Later dropped.
        checklist_stixel_assignment_exploded_{year}.ensemble_index,     --- used for checking. Later dropped.
        checklist_stixel_assignment_exploded_{year}.unique_stixel_id,   --- used for checking. Later dropped.
        checklist_species_count_{year}.count AS species_count,           --- used for checking. Later dropped.
        {species.replace(" ","_")}_checklist_indiv_count.count AS {species.replace(" ","_")}_indiv_count,   --- this is the target (y/response) 
        environment_4_each_checklist.elevation,           --- env1
        environment_4_each_checklist.slope,               --- env2
        environment_4_each_checklist.eastness,            --- env3
        environment_4_each_checklist.northness,            --- env4
        land_type_stats_{year}.closed_shrublands,                                 --- env4
        land_type_stats_{year}.closed_shrublands_ed,                              --- env4
        land_type_stats_{year}.closed_shrublands_lpi,                             --- env4
        land_type_stats_{year}.closed_shrublands_pd,                              --- env4
        land_type_stats_{year}.cropland_or_natural_vegetation_mosaics,            --- env4
        land_type_stats_{year}.cropland_or_natural_vegetation_mosaics_ed,         --- env4
        land_type_stats_{year}.cropland_or_natural_vegetation_mosaics_lpi,        --- env4
        land_type_stats_{year}.cropland_or_natural_vegetation_mosaics_pd,         --- env4
        land_type_stats_{year}.croplands,                                        --- env4            
        land_type_stats_{year}.croplands_ed,                                        --- env4       
        land_type_stats_{year}.croplands_lpi,                                       --- env4       
        land_type_stats_{year}.croplands_pd,                                                 --- env4
        land_type_stats_{year}.deciduous_broadleaf_forests,                               --- env4
        land_type_stats_{year}.deciduous_broadleaf_forests_ed,                          --- env4
        land_type_stats_{year}.deciduous_broadleaf_forests_lpi,                           --- env4
        land_type_stats_{year}.deciduous_broadleaf_forests_pd,                          --- env4
        land_type_stats_{year}.deciduous_needleleaf_forests,                               --- env4
        land_type_stats_{year}.deciduous_needleleaf_forests_ed,                          --- env4
        land_type_stats_{year}.deciduous_needleleaf_forests_lpi,                          --- env4
        land_type_stats_{year}.deciduous_needleleaf_forests_pd,                          --- env4
        land_type_stats_{year}.evergreen_broadleaf_forests,                              --- env4
        land_type_stats_{year}.evergreen_broadleaf_forests_ed,                           --- env4
        land_type_stats_{year}.evergreen_broadleaf_forests_lpi,                         --- env4
        land_type_stats_{year}.evergreen_broadleaf_forests_pd,                            --- env4
        land_type_stats_{year}.evergreen_needleleaf_forests,                               --- env4
        land_type_stats_{year}.evergreen_needleleaf_forests_ed,                             --- env4
        land_type_stats_{year}.evergreen_needleleaf_forests_lpi,                            --- env4
        land_type_stats_{year}.evergreen_needleleaf_forests_pd,                             --- env4
        land_type_stats_{year}.grasslands,                                                  --- env4
        land_type_stats_{year}.grasslands_ed,                                               --- env4
        land_type_stats_{year}.grasslands_lpi,                                              --- env4
        land_type_stats_{year}.grasslands_pd,                                               --- env4
        land_type_stats_{year}.mixed_forests,                                               --- env4
        land_type_stats_{year}.mixed_forests_ed,                                            --- env4
        land_type_stats_{year}.mixed_forests_lpi,                                           --- env4
        land_type_stats_{year}.mixed_forests_pd,                                            --- env4
        land_type_stats_{year}.non_vegetated_lands,                                         --- env4
        land_type_stats_{year}.non_vegetated_lands_ed,                                      --- env4
        land_type_stats_{year}.non_vegetated_lands_lpi,                                     --- env4
        land_type_stats_{year}.non_vegetated_lands_pd,                                      --- env4
        land_type_stats_{year}.open_shrublands,                                             --- env4
        land_type_stats_{year}.open_shrublands_ed,                                          --- env4
        land_type_stats_{year}.open_shrublands_lpi,                                         --- env4
        land_type_stats_{year}.open_shrublands_pd,                                          --- env4
        land_type_stats_{year}.permanent_wetlands,                                          --- env4
        land_type_stats_{year}.permanent_wetlands_ed,                                       --- env4
        land_type_stats_{year}.permanent_wetlands_lpi,                                      --- env4
        land_type_stats_{year}.permanent_wetlands_pd,                                       --- env4
        land_type_stats_{year}.savannas,                                                    --- env4
        land_type_stats_{year}.savannas_ed,                                                 --- env4
        land_type_stats_{year}.savannas_lpi,                                                --- env4
        land_type_stats_{year}.savannas_pd,                                                 --- env4
        land_type_stats_{year}.urban_and_built_up_lands,                                    --- env4
        land_type_stats_{year}.urban_and_built_up_lands_ed,                                 --- env4
        land_type_stats_{year}.urban_and_built_up_lands_lpi,                                --- env4
        land_type_stats_{year}.urban_and_built_up_lands_pd,                                 --- env4
        land_type_stats_{year}.water_bodies,                                                --- env4
        land_type_stats_{year}.water_bodies_ed,                                             --- env4
        land_type_stats_{year}.water_bodies_lpi,                                            --- env4
        land_type_stats_{year}.water_bodies_pd,                                             --- env4
        land_type_stats_{year}.woody_savannas,                                              --- env4
        land_type_stats_{year}.woody_savannas_ed,                                           --- env4
        land_type_stats_{year}.woody_savannas_lpi,                                          --- env4
        land_type_stats_{year}.woody_savannas_pd,                                           --- env4
        land_type_stats_{year}.entropy                                                     --- env4
        
    FROM
        checklist_origin_{year}
    LEFT JOIN checklist_stixel_assignment_exploded_{year} 
                ON 
                checklist_origin_{year}.sampling_event_identifier = checklist_stixel_assignment_exploded_{year}.sampling_event_identifier
    LEFT JOIN checklist_species_count_{year} 
                ON 
                checklist_origin_{year}.sampling_event_identifier = checklist_species_count_{year}.sampling_event_identifier
    LEFT JOIN (SELECT * FROM checklist_indiv_count_each_species_{year} WHERE common_name LIKE '{species}') 
            AS  {species.replace(" ","_")}_checklist_indiv_count
                ON 
                checklist_origin_{year}.sampling_event_identifier = {species.replace(" ","_")}_checklist_indiv_count.sampling_event_identifier
    LEFT JOIN environment_4_each_checklist 
                ON 
                checklist_origin_{year}.sampling_event_identifier = environment_4_each_checklist.sampling_event_identifier
    LEFT JOIN land_type_stats_{year}
                ON
                checklist_origin_{year}.sampling_event_identifier = land_type_stats_{year}.sampling_event_identifier
    LEFT JOIN observer_expertise_cumulative_{year}
                ON
                checklist_origin_{year}.observer_id = REPLACE(observer_expertise_cumulative_{year}.observer_id, 'obsr', 'obs') ---- A Huge trap!!!
                
    WHERE
        checklist_stixel_assignment_exploded_{year}.unique_stixel_id is NOT NULL
        AND
        checklist_stixel_assignment_exploded_{year}.ensemble_index = {ensemble_index}
        AND
        checklist_species_count_{year}.count >= 10
""")
qresult = pd.DataFrame(cur.fetchall())
cur.close()
connect.close()
print(f"""sucessfully fetching year {year} speices \"{species}\" ensemble {ensemble_index},\n           
                          raw data contain {len(qresult)} checklists""")


# In[6]:


qresult


# In[7]:


#### Change the column names.
qresult.columns = [
    'sampling_event_identifier',
    'duration_minutes',
    'protocol_type',
    'effort_distance_km',
    'number_observers',
    'time_observation_started',
    'observation_date',
    'country',
    'locality_id',
    'observer_id',
    'longitude',
    'latitude',
    'obsvr_species_count',
#     'obsvr_checklist_count',
#     'obsvr_species_record_rate',
    'group_identifier',
    'ensemble_index',
    'unique_stixel_id',
    'species_count',
    f'{species.replace(" ","_")}_indiv_count',
    'elevation',
    'slope',
    'eastness',
    'northness',
    'closed_shrublands',
    'closed_shrublands_ed',
    'closed_shrublands_lpi',
    'closed_shrublands_pd',
    'cropland_or_natural_vegetation_mosaics',
    'cropland_or_natural_vegetation_mosaics_ed',
    'cropland_or_natural_vegetation_mosaics_lpi',
    'cropland_or_natural_vegetation_mosaics_pd',
    'croplands',                                     
    'croplands_ed',                                
    'croplands_lpi',                                              
    'croplands_pd',                                                 
    'deciduous_broadleaf_forests',                               
    'deciduous_broadleaf_forests_ed',                          
    'deciduous_broadleaf_forests_lpi',                           
    'deciduous_broadleaf_forests_pd',                          
    'deciduous_needleleaf_forests',                               
    'deciduous_needleleaf_forests_ed',                          
    'deciduous_needleleaf_forests_lpi',                          
    'deciduous_needleleaf_forests_pd',                          
   'evergreen_broadleaf_forests',                              
   'evergreen_broadleaf_forests_ed',                           
  'evergreen_broadleaf_forests_lpi',                         
   'evergreen_broadleaf_forests_pd',                            
   'evergreen_needleleaf_forests',                               
   'evergreen_needleleaf_forests_ed',                             
   'evergreen_needleleaf_forests_lpi',                            
   'evergreen_needleleaf_forests_pd',                             
   'grasslands',                                                  
   'grasslands_ed',                                               
   'grasslands_lpi',                                              
   'grasslands_pd',                                               
   'mixed_forests',                                               
   'mixed_forests_ed',                                            
   'mixed_forests_lpi',                                           
   'mixed_forests_pd',                                            
   'non_vegetated_lands',                                         
   'non_vegetated_lands_ed',                                      
   'non_vegetated_lands_lpi',                                     
   'non_vegetated_lands_pd',                                      
   'open_shrublands',                                             
   'open_shrublands_ed',                                          
   'open_shrublands_lpi',                                         
   'open_shrublands_pd',                                          
   'permanent_wetlands',                                          
   'permanent_wetlands_ed',                                       
   'permanent_wetlands_lpi',                                      
   'permanent_wetlands_pd',                                       
   'savannas',                                                    
   'savannas_ed',                                                 
   'savannas_lpi',                                                
   'savannas_pd',                                                 
   'urban_and_built_up_lands',                                    
   'urban_and_built_up_lands_ed',                                 
   'urban_and_built_up_lands_lpi',                                
   'urban_and_built_up_lands_pd',                                 
  'water_bodies',                                                
   'water_bodies_ed',                                             
   'water_bodies_lpi',                                            
   'water_bodies_pd',                                             
   'woody_savannas',                                              
   'woody_savannas_ed',                                           
   'woody_savannas_lpi',                                          
   'woody_savannas_pd',                                           
   'entropy'
]


# In[8]:


pd.DataFrame(qresult.isnull().sum(axis=0)).T


# In[9]:


with_ = sum(qresult[f'{species.replace(" ","_")}_indiv_count']>0)
without_ = len(qresult) - with_
print(f'''For raw data, No. of Checklists with {species.replace(" ","_")}: {with_}, No. of Checklists without {species.replace(" ","_")}: {without_},\n
{round((with_/(with_+without_))*100, 2)}% checklists has {species.replace(" ","_")}''')


# In[10]:


from filter_data_script import *
from stixel_level_filtering_script import *
from calc_weight_script import *
from get_binary_response_script import *
from GBDT1_training_script import *
from GAM_training_script import *
from print_GAM_metrics_script import *
from get_data_for_GBDT2_script import *
from GBDT2_model_training_script import *
from get_and_print_GBDT2_metrics_script import *
from get_PD_direction_and_feature_importance_script import *
from basemodel_class import *
import filter_data_script


# ## 2) Filter data

# In[11]:

qresult = filter_data(qresult, species)
                                                           
# # NEXT, we will just define some functions and then run the model

# ## 3) Define the base training model

# In[12]:


def THE_WHOLE_PIPELINE(qresult, stixel_id):
    start_time = time.time()
    print('############           1) Get the sub data for this stixel and filter        ############')
    filtering_result, scaler, detection_rate_before_oversampling = stixel_level_filtering(qresult, stixel_id, species)
    if filtering_result == "Model terminated because checklist count<50 for this stixel":
        print(f"{stixel_id}: Killed_for_checklist_count")
        this_time = time.time()
        print(f"Total Time Used For This Stixel: {round(this_time - start_time, 2)} second.")
        return "Killed_for_checklist_count"
    if filtering_result =="Killed_for_all_occupied, using mean abundance":
        print("Killed_for_all_occupied, using mean abundance")
        return "Killed_for_all_occupied, using mean abundance"
    if filtering_result =="Killed_for_all_absence, not occupied":
        print("Killed_for_all_absence, not occupied")
        return "Killed_for_all_absence, not occupied"
    elif isinstance(filtering_result, str):
        print(filtering_result)
        return f"Killed for: {filtering_result}"
    

        
    this_time = time.time()
    print(f'time used for this stage1: {round((this_time - start_time),2)} seconds.')
    last_time = this_time
    print(' ')
    
    print('############                  2) Train test split and resampling             ############')
    X_train, X_test, y_train, y_test = filtering_result

    if len(y_train.unique()) == 1 or len(y_test.unique()) == 1:
        if y_train.unique()[0] == 1 or y_test.unique()[0] == 1:
            print(f"{stixel_id}: Killed_for_all_presense, using mean abundance value")
            this_time = time.time()
            print(f"Total Time Used For This Stixel: {round(this_time - start_time, 2)} second.")
            return np.mean(np.array(list(y_train.values) + list(y_test.values)))
        elif y_train.unique()[0] == 0 or y_test.unique()[0] == 0:
            print(f"{stixel_id}: Killed_for_all_absence, not occupied")
            this_time = time.time()
            print(f"Total Time Used For This Stixel: {round(this_time - start_time, 2)} second.")
            return "Killed_for_all_absence, not occupied"
    
#     X_train, y_train = unbalance_resample(X_train, y_train)
    weights = calc_sample_weights(y_train)
    y_train_binary, y_test_binary = get_binary_response(y_train, y_test) 
    
    this_time = time.time()
    print(f'time used for this stage2: {round((this_time - last_time),2)} seconds.')
    last_time = this_time
    print(' ')
    
    print('############          3) Train the first GBDT classifier model               ############')
    GBDT, GBDT_pred_train, GBDT_pred_test = \
            GBDT1_training(X_train, X_test, y_train_binary, weights, y_test_binary=None)
    
    #### adjust for detection rate
    GBDT_pred_train_adj = 1/(np.e**(-np.log(1/GBDT_pred_train - 1) - np.log((1-detection_rate_before_oversampling)/detection_rate_before_oversampling * (0.5/(1-0.5)))) + 1)
#     GBDT_pred_train_adj = GBDT_pred_train

    #### get its metrics
    occurance_feature_importance, GBDT1_metrics = \
        get_and_print_GBDT1_metrics(GBDT, y_train_binary, y_test_binary, GBDT_pred_train_adj, GBDT_pred_test, X_train)


    auc_training, recall_training, precision_training, cohen_kappa_score_training,\
                    acc_testing, recall_testing, precision_testing, auc_testing, f1_testing, cohen_kappa_score_testing,\
                    confusion_matrix_testing = GBDT1_metrics
    
    this_time = time.time()
    print(f'time used for this stage3: {round((this_time - last_time),2)} seconds.')
    last_time = this_time
    print(' ')
    
    print('############  4) Train the GAM model to smooth the occurance estimates       ############')
    gam_result = GAM_training(X_train, X_test, GBDT_pred_train_adj, weights)

    if gam_result == "GAM_not_converged":
#         return "Killed_for_GAM_not_converged"
        print("GAM_not_converged, using GBDT1 result for downstream analysis")
#         print(get_data_for_GBDT2(X_train, X_test, y_train, y_test, GBDT_pred_train, GBDT_pred_test))
        get_data_for_GBDT2_result = get_data_for_GBDT2(X_train, X_test, y_train, y_test, GBDT_pred_train, GBDT_pred_test, weights)
        gam = None
        gam_stats_dict = None
        
    else:
        gam, gam_pred_train, gam_pred_test, gam_stats_dict = gam_result
        
        #### get its metrics
        print_GAM_metrics(y_train_binary, y_test_binary, gam_pred_train, gam_pred_test)

        get_data_for_GBDT2_result = get_data_for_GBDT2(X_train, X_test, y_train, y_test, gam_pred_train, gam_pred_test, weights)

    if get_data_for_GBDT2_result == "Killed_for_GAM_predict_all_train_data_absence":
        print("Killed_for_GAM_predict_all_train_data_absence")
        this_time = time.time()
        print(f"Total Time Used For This Stixel: {round(this_time - start_time, 2)} second.")
        return "Killed_for_GAM_predict_all_train_data_absence"
    elif get_data_for_GBDT2_result == "Killed_for_GAM_predict_all_test_data_absence":
        print("Killed_for_GAM_predict_all_test_data_absence")
        this_time = time.time()
        print(f"Total Time Used For This Stixel: {round(this_time - start_time, 2)} second.")
        return "Killed_for_GAM_predict_all_test_data_absence"
    else:
        train_used_index, test_used_index, X_train_for_GBDT2, X_test_for_GBDT2, y_train_for_GBDT2, y_test_for_GBDT2, weights_for_GBDT2 =\
            get_data_for_GBDT2_result
            
    
    this_time = time.time()
    print(f'time used for this stage4: {round((this_time - last_time),2)} seconds.')
    last_time = this_time
    print(' ')
    
    print('############  5) Train the second GBDT model to get the abundance estimates   ############')
    #### Here, I did not weight it by the occurance because if will underestimate the abundance
    GBDT2, abundance_pred_train, abundance_pred_test = GBDT2_model_training(X_train_for_GBDT2, X_test_for_GBDT2, y_train_for_GBDT2, weights_for_GBDT2)
    
    
    #### get its metrics
    r2_training, mse_training, explained_variance_training, spearman_rank_correlation_training,\
                    r2_testing, mse_testing, explained_variance_testing, spearman_rank_correlation_testing =\
            get_and_print_GBDT2_metrics(y_train_for_GBDT2, y_test_for_GBDT2, abundance_pred_train, abundance_pred_test)
    
    #### plot it
    fig,ax = plt.subplots(1,2,figsize=(15,5))
    plt.sca(ax[0])
    plt.scatter(abundance_pred_train, y_train_for_GBDT2)
    plt.xlabel('abundance_pred_train')
    plt.ylabel('y_train_for_GBDT2')
    plt.title('performance on train')
    plt.sca(ax[1])
    plt.scatter(abundance_pred_test, y_test_for_GBDT2)
    max_y=np.max(abundance_pred_test.flatten().tolist() + y_test_for_GBDT2.flatten().tolist())
    plt.plot([0,max_y],[0,max_y],c='r')
    plt.xlabel('abundance_pred_test')
    plt.ylabel('y_test_for_GBDT2')
    plt.title('performance on test')
    plt.show()
    
    #### get PD
    pd_res = get_PD_direction_and_feature_importance(GBDT2, X_train)
    PD_FI = pd.DataFrame(pd_res).T
    
    this_time = time.time()
    print(f'time used for this stage5: {round((this_time - last_time),2)} seconds.')
    last_time = this_time
    print(' ')  

    print('############          6) Store everything into the base_model class          ############')
    #### Store the metrics
    metrics = {
        'training':{
            'abundance_r2':r2_training,
            'abundance_mse':mse_training,
            'abundance_explained_variance':explained_variance_training,
            'abundance_spearman_rank_correlation':spearman_rank_correlation_training,
            'occurance_auc':auc_training,
            'occurance_recall':recall_training,
            'occurance_precision':precision_training,
            'occurance_cohen_kappa':cohen_kappa_score_training
        },
        'testing':{
            'abundance_r2_testing':r2_testing,
            'abundance_mse':mse_testing,
            'abundance_explained_variance':explained_variance_testing,
            'abundance_spearman_rank_correlation':spearman_rank_correlation_testing,
            'occurance_acc':acc_testing,
            'occurance_recall':recall_testing,
            'occurance_precision':precision_testing,
            'occurance_auc':auc_testing,
            'occurance_f1':f1_testing,
            'occurance_cohen_kappa':cohen_kappa_score_testing,
            'occurance_confusion_matrix':confusion_matrix_testing
        }
    }
    occurance_feature_importance = occurance_feature_importance
    stixel_id = stixel_id
    bm = base_model(GBDT, gam, GBDT2, metrics, occurance_feature_importance, \
                    ensemble_index, stixel_id, year, species, user_defined_species_index, PD_FI, len(X_train), scaler)
    
    this_time = time.time()
    print(f"Total Time Used For This Stixel: {round(this_time - start_time, 2)} second.")
    return bm
    


    


# In[45]:


# %load_ext line_profiler


# # LOOP

# In[62]:


# %lprun -f GBDT1_training \
##### GET ROCKKKKKKK TRAIN THEM!
file_path = f"{model_saving_dir}/{str(user_defined_species_index).zfill(5)}.{species.replace(' ','_')}/{species.replace(' ','_')}_model_ensemble{ensemble_index}_year{year}.log"
sys.stdout = open(file_path, "w")
print(f"total model count: {len(qresult['unique_stixel_id'].unique())}")
whole_ensemble_stixels_dict={}
print(f"======================= {str(user_defined_species_index).zfill(5)}.{species.replace(' ','_')}, ensemble {ensemble_index} ====== TOTAL MODEL COUNT: {len(qresult['unique_stixel_id'].unique())} ==============================")
sys.stdout.flush()
for count,stixel_id in enumerate(qresult['unique_stixel_id'].unique()):
    sys.stdout = open(file_path, "a")
    print("====================================== NEW MODEL TRAINING SESSION ================================================")
    print("===================================================================================================================")
    print(f"No.{count}, Start model training on stixel {stixel_id}")
    print("===================================================================================================================")
    try:
        this_bm = THE_WHOLE_PIPELINE(qresult, stixel_id)     
        whole_ensemble_stixels_dict[stixel_id] = this_bm
        
    except Exception as e:
        print(f"The training for stixel {stixel_id} is killed for some reason: {e}. :(")
        
    print("===================================================================================================================")
    print("  ")
    print("  ")
    sys.stdout.flush()
    print(count)

    
# In[32]:


model_file_path = f"{model_saving_dir}/{str(user_defined_species_index).zfill(5)}.{species.replace(' ','_')}/{species.replace(' ','_')}_model_ensemble{ensemble_index}_year{year}.pkl"
with open(model_file_path,'wb') as file:
    pickle.dump(whole_ensemble_stixels_dict, file)
    print("Successfully saved the models!")


# In[33]:


print(f"""=================================== Congrats!!! ============================================================
"Species \"{species}\", index {user_defined_species_index}: Base Model Training for Ensemble {ensemble_index}, Year {year}. Complete!")
============================================================================================================""")


# In[43]:


# model_file = open(model_file_path,'rb')
# model_list = pickle.load(model_file)
# model_list


# In[42]:


# list(model_list.values())[0].PD_FI.loc['time_observation_started_minute_of_day']['max_value_point']


# In[ ]:


################################################ Prediction ################################################
from geotiff import GeoTiff
import matplotlib.pyplot as plt
import rasterio as rs
from psycopg2 import Error
from osgeo import gdal, osr, ogr

# this_ensemble_path = f"/beegfs/store4/chenyangkang/06.ebird_data/14.model_training/01.trained_model/00000.Wood_Thrush/Wood_Thrush_model_ensemble{ensemble_index}_year{year}.pickle"
# model_file = open(this_ensemble_path, 'rb')
# models = pickle.load(model_file)
# model_file.close()
models = whole_ensemble_stixels_dict


### Request ensemble/stixel info
connect = psycopg2.connect('dbname=ebird user=chenyangkang host=fat01 port=5432')
cur = connect.cursor()
cur.execute(f"""
    SELECT * from stixel_info_{year} WHERE ensemble_index={ensemble_index}
""")
ensemble_info = pd.DataFrame(cur.fetchall())
ensemble_info.columns = ['stixel_indexes','stixel_width','stixel_height',\
                         'stixel_checklist_count','stixel_calibration_point_transformed','rotation',\
                         'space_jitter_first_rotate_by_zero_then_add_this',
                         'ensemble_index','doy_start','doy_end','year','unique_stixel_id']
cur.close()
connect.close()

#######
prediction_data = \
pd.read_csv(f'/beegfs/store4/chenyangkang/06.ebird_data/13.prediction_set/prediction_set_gridlen10_{year}.txt',\
                              sep="\t")


### Two set. One for location data, one for predictor data
prediction_set_coordinates = prediction_data[['longitude','latitude']]

prediction_set_values = prediction_data.iloc[:,5:]
prediction_set_values = prediction_set_values.dropna(subset=['elevation', 'slope',\
       'eastness', 'northness']).fillna(0)

prediction_set_values.insert(0,'DOY',np.nan)
prediction_set_values.insert(1,'duration_minutes',60)
prediction_set_values.insert(2,'time_observation_started_minute_of_day',np.nan)
### should be models['2010_0_26_5'].PD_FI['max_value_point']['time_observation_started_minute_of_day']
prediction_set_values.insert(3,'Traveling',1)
prediction_set_values.insert(4,'Stationary',0)
prediction_set_values.insert(5,'Area',0) 
prediction_set_values.insert(6,'effort_distance_km',1)
prediction_set_values.insert(7,'effort_distance_km_missing',0)
prediction_set_values.insert(8,'number_observers',1)
prediction_set_values.insert(9,'obsvr_species_count',1)

### Desired Week of Year, and corresbonding mid-day-of-week of year

dates = pd.Series(pd.date_range(f'{year}-01-01', f'{year}-12-31'))
desired_doy = dates.dt.dayofyear[dates.dt.dayofweek == 3].values


for item in ['stixel_calibration_point_transformed', 'space_jitter_first_rotate_by_zero_then_add_this']: 
    exec(f"ensemble_info['{item}']="+",".join(ensemble_info[item].values.tolist()))

    
    
    
#### Transform the coordinate system (rotation and jitter) of the prediction set points. 
#### So that it match the coordinates of rotated and jittered QuadTrees.
x_array = prediction_set_coordinates['longitude']
y_array = prediction_set_coordinates['latitude']
coord = np.array([x_array, y_array]).T
angle = float(ensemble_info.iloc[0,:]['rotation'])
r = angle/360
theta = r * np.pi * 2
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

coord = coord @ rotation_matrix
calibration_point_x_jitter = \
        float(ensemble_info.iloc[0,:]['space_jitter_first_rotate_by_zero_then_add_this'][0])
calibration_point_y_jitter = \
        float(ensemble_info.iloc[0,:]['space_jitter_first_rotate_by_zero_then_add_this'][1])

long_new = (coord[:,0] + calibration_point_x_jitter).tolist()
lat_new = (coord[:,1] + calibration_point_y_jitter).tolist()

prediction_set_coordinates['long_new'] = long_new
prediction_set_coordinates['lat_new'] = lat_new


### Get the bound of each stixel in four direction
ensemble_info['stixel_calibration_point_transformed_left_bound'] = \
            [i[0] for i in ensemble_info['stixel_calibration_point_transformed']]

ensemble_info['stixel_calibration_point_transformed_lower_bound'] = \
            [i[1] for i in ensemble_info['stixel_calibration_point_transformed']]

ensemble_info['stixel_calibration_point_transformed_right_bound'] = \
            ensemble_info['stixel_calibration_point_transformed_left_bound'] + ensemble_info['stixel_width']

ensemble_info['stixel_calibration_point_transformed_upper_bound'] = \
            ensemble_info['stixel_calibration_point_transformed_lower_bound'] + ensemble_info['stixel_height']



### get data for each middle day of week. Predict the abundance with the corresbonding stixel.
def write_geotiff(filename, arr, metadata):
    arr_type = gdal.GDT_Float32
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(filename, arr.shape[1], arr.shape[0], 1, arr_type)
#     out_ds.SetProjection(in_ds.GetProjection())
#     out_ds.SetGeoTransform(in_ds.GetGeoTransform())
#     out_ds.SetSpatialRef(in_ds.GetSpatialRef())
    out_ds.SetMetadata(metadata)
    band = out_ds.GetRasterBand(1)
    band.WriteArray(arr)
    band.FlushCache()
    band.ComputeStatistics(False)




sys.stdout = open(file_path, "a")
for week_count,doy in enumerate(desired_doy):
    month = pd.to_datetime(pd.to_datetime(f'{year}-01-01')+pd.Timedelta(f'{doy-1}D')).month
    print(f"Prediction: week of year {week_count+1}, day of year {doy}, month {month}")
    this_prediction_set_values = prediction_set_values.copy()
    this_prediction_set_values['DOY'] = doy
    this_prediction_set_coordinates = prediction_set_coordinates.copy()
    this_prediction_set_coordinates['DOY'] = doy
    
    this_doy_abundance = pd.Series([np.nan] * this_prediction_set_coordinates.shape[0])
    
    desired_stixels = ensemble_info[
        (ensemble_info['doy_start']<=doy) & (ensemble_info['doy_end']>=doy)
    ]
    
    if len(desired_stixels) == 0:
        print(f"Not a single stixel useful for PREDICTION of week of year {week_count}, day of year {doy}, month {month}")
    else:
        for index, stixel in desired_stixels.iterrows():
            if not stixel['unique_stixel_id'] in models.keys():
                print(f"The stixel not exists in models? week of year {week_count}, day of year {doy}, month {month}")
                continue
            correspond_model = models[stixel['unique_stixel_id']]
    #         print(correspond_model)
            sample_points_falls_in = this_prediction_set_coordinates[
                      (stixel['stixel_calibration_point_transformed_left_bound'] <= this_prediction_set_coordinates['long_new'])\
                    & (stixel['stixel_calibration_point_transformed_right_bound'] >= this_prediction_set_coordinates['long_new'])\
                    & (stixel['stixel_calibration_point_transformed_lower_bound'] <= this_prediction_set_coordinates['lat_new'])\
                    & (stixel['stixel_calibration_point_transformed_upper_bound'] >= this_prediction_set_coordinates['lat_new'])
            ]
            if len(sample_points_falls_in)==0:
                print(f"Has stixel but no points? week of year {week_count}, day of year {doy}, month {month}")
                continue
                
            if isinstance(correspond_model, str):
                if correspond_model.startswith('Killed_for_all_absence'):
                    abundance = 0
                elif correspond_model.startswith("Killed"):
                    abundance = np.nan

            elif isinstance(correspond_model, float) or isinstance(correspond_model, int):
                abundance = correspond_model
                print(f"{stixel['unique_stixel_id']} Using Mean Abundance Across Checklists!")
            else:
                try:
                    #### Do the real estimate
                    sample_points = this_prediction_set_values.loc[sample_points_falls_in.index,:]
                    if not isinstance(correspond_model.PD_FI['max_value_point']['time_observation_started_minute_of_day'], float):
                        sample_points['time_observation_started_minute_of_day'] = 360
                        print(f"{stixel['unique_stixel_id']} PD_FI time_observation_started_minute_of_day in error, using 6 oclock")
                    else:
                        sample_points['time_observation_started_minute_of_day'] = correspond_model.PD_FI['max_value_point']['time_observation_started_minute_of_day']
                    abundance = correspond_model.predict(sample_points).flatten().tolist()
                    
                except Exception as e:
                    print(f"{stixel['unique_stixel_id']} prediction error: {e}")
                    continue

            this_doy_abundance[sample_points_falls_in.index] = abundance
    #         print(abundance)
            del abundance
    
    
    grid_len = 10
    long_array = np.arange(-17367530.445161372,17367530.445161372,grid_len*1e3)
    lat_array = np.arange(-7342230.13649868,7342230.13649868,grid_len*1e3)
    
    im = np.flip(this_doy_abundance.values.reshape(len(lat_array), len(long_array)), axis=0)
    im = np.where(im<0, 0, im)
    im = np.where(im>=0, im, -1)
    metadata = {'species':species.replace(' ','_'),\
                'lower_left_point':(-17367530.445161372, -7342230.13649868),\
               'grid_length_km':grid_len,
                'grid_shape':(len(long_array), len(lat_array)),
               'projection':'ESRI:54017'}
    filename = f"{model_saving_dir}/{str(user_defined_species_index).zfill(5)}.{species.replace(' ','_')}/{species.replace(' ','_')}_year{year}_ensemble{ensemble_index}_week{week_count}_DOY{doy}_month{month}.tiff"
    write_geotiff(filename, im, metadata)
#     plt.imshow(im, cmap = 'viridis')
#     plt.show()
    print('done')
    sys.stdout.flush()
        


