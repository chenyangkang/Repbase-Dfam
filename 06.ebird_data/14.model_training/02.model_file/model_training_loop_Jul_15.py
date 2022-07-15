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
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score,recall_score, precision_score, roc_auc_score, confusion_matrix, cohen_kappa_score
from pygam import LinearGAM 
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score, mean_poisson_deviance
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.inspection import partial_dependence
import time
from sklearn.preprocessing import MinMaxScaler

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",None)

np.random.seed(42)




#################
global year, ensemble_index, user_defined_species_index, species, model_saving_dir
### notice: you should pass a name without gap but _. I will generate the _ for the later analysis. Because sys.argv donot allow gaps.
species = str(sys.argv[3]).replace("_"," ")
year = int(sys.argv[1])
ensemble_index = int(sys.argv[4])
user_defined_species_index = int(sys.argv[2])

# year = 2010
# ensemble_index = 15
# user_defined_species_index = 1
# species = "Wood Thrush"

model_saving_dir = "/beegfs/store4/chenyangkang/06.ebird_data/14.model_training/01.trained_model"

###################
if not os.path.exists(model_saving_dir):
    print(f'Out_dir {model_saving_dir} not exists! Please Check it.')
if not os.path.exists(f"{model_saving_dir}/{str(user_defined_species_index).zfill(5)}.{species.replace(' ','_')}"):
    os.mkdir(f"{model_saving_dir}/{str(user_defined_species_index).zfill(5)}.{species.replace(' ','_')}")
    print(f'''You should first make a folder named by the userdefined species index and the species name. 
            For example: \"00002.Wood_Thrush\". Remember to zfill the index to ten thousands place! ''')
    print("Now I have made it for you. Not need to do that.")

###################
from filter_data_script import *
from request_data import *
# from filter_data_script_using_mean import *
from stixel_level_filtering_script_without_balanced_sampling import *
from calc_weight_script import *
from get_binary_response_script import *
from GBDT1_training_script_balanced import *
from GAM_training_script import *
from print_GAM_metrics_script import *
from get_data_for_GBDT2_script import *
from GBDT2_model_training_script import *
from get_and_print_GBDT2_metrics_script import *
from get_PD_direction_and_feature_importance_script import *
from basemodel_class import *
from prediction_utils import *

################### request data
qresult = request_data(year,species,ensemble_index)

################### filter data
qresult = filter_data(qresult, species)

############## DEFINE LOOP TRAINING FUNCTION
def THE_WHOLE_PIPELINE(qresult, stixel_id):
    start_time = time.time()
    print('############           1) Get the sub data for this stixel and filter        ############')
    filtering_result, scaler, detection_rate_before_oversampling = stixel_level_filtering(qresult, stixel_id, species)
    if filtering_result == "Model terminated because checklist count<30 for this stixel":
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
#     gam_result = GAM_training(X_train, X_test, GBDT_pred_train_adj, weights)
    gam_result = "GAM_not_converged"
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
        train_used_index, test_used_index, X_train_for_GBDT2, X_test_for_GBDT2, y_train_for_GBDT2, y_test_for_GBDT2, weights_for_GBDT2, scaler_minmax =\
            get_data_for_GBDT2_result
            
    
    this_time = time.time()
    print(f'time used for this stage4: {round((this_time - last_time),2)} seconds.')
    last_time = this_time
    print(' ')
    
    print('############  5) Train the second GBDT model to get the abundance estimates   ############')
    #### Here, I did not weight it by the occurance because if will underestimate the abundance
    GBDT2, abundance_pred_train, abundance_pred_test = GBDT2_model_training(X_train_for_GBDT2, X_test_for_GBDT2, y_train_for_GBDT2, weights_for_GBDT2)
    
    
    #### get its metrics
    r2_training, mse_training, explained_variance_training, spearman_rank_correlation_training, poisson_explained_variance_training,\
                    r2_testing, mse_testing, explained_variance_testing, spearman_rank_correlation_testing, poisson_explained_variance_testing =\
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
            'abundance_poisson_explained_variance':poisson_explained_variance_training,
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
            'abundance_poisson_explained_variance':poisson_explained_variance_testing,
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
                    ensemble_index, stixel_id, year, species, user_defined_species_index, PD_FI, len(X_train), scaler, scaler_minmax)
    
    this_time = time.time()
    print(f"Total Time Used For This Stixel: {round(this_time - start_time, 2)} second.")
    return bm
    
    
    
################loop
# %lprun -f GBDT1_training \
##### GET ROCKKKKKKK TRAIN THEM!
file_path = f"{model_saving_dir}/{str(user_defined_species_index).zfill(5)}.{species.replace(' ','_')}/{species.replace(' ','_')}_model_ensemble{ensemble_index}_year{year}.log"
sys.stdout = open(file_path, "w")
mse_list=[]
r2_list=[]
print(f"total model count: {len(qresult['unique_stixel_id'].unique())}")
whole_ensemble_stixels_dict={}
print(f"======================= {str(user_defined_species_index).zfill(5)}.{species.replace(' ','_')}, ensemble {ensemble_index} ====== TOTAL MODEL COUNT: {len(qresult['unique_stixel_id'].unique())} ==============================")
sys.stdout.flush()
for count,stixel_id in enumerate(qresult['unique_stixel_id'].unique()[0:50]):
#     sys.stdout = open(file_path, "a")
    print("====================================== NEW MODEL TRAINING SESSION ================================================")
    print("===================================================================================================================")
    print(f"No.{count}, Start model training on stixel {stixel_id}")
    print("===================================================================================================================")
#     try:
    this_bm = THE_WHOLE_PIPELINE(qresult, stixel_id)

    whole_ensemble_stixels_dict[stixel_id] = this_bm
        
#     except Exception as e:
#         print(f"The training for stixel {stixel_id} is killed for some reason. :(")
#     except Exception as e:
#         print(f"The training for stixel {stixel_id} is killed for some reason. {e}")
    print("===================================================================================================================")
    print("  ")
    print("  ")
    sys.stdout.flush()
    print(count)
    

    
################ save
model_file_path = f"{model_saving_dir}/{str(user_defined_species_index).zfill(5)}.{species.replace(' ','_')}/{species.replace(' ','_')}_model_ensemble{ensemble_index}_year{year}.pkl"
with open(model_file_path,'wb') as file:
    pickle.dump(whole_ensemble_stixels_dict, file)
    print("Successfully saved the models!")

    
print(f"""=================================== Congrats!!! ============================================================
"Species \"{species}\", index {user_defined_species_index}: Base Model Training for Ensemble {ensemble_index}, Year {year}. Complete!")
============================================================================================================""")
    
################ making prediction

sys.stdout = open(file_path, "w")
make_prediction(whole_ensemble_stixels_dict, species, year, ensemble_index, model_saving_dir)
sys.stdout.flush()




