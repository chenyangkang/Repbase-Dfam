### import libraries

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



def request_ensemble_info(year, ensemble_index):
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
    prediction_set_values.insert(9,'obsvr_species_count',7000)

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
    
    return prediction_set_values, prediction_set_coordinates, ensemble_info, desired_doy



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
    



def make_prediction(whole_ensemble_stixels_dict, species, year, ensemble_index, output_dir):
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
    
    prediction_set_values, prediction_set_coordinates, ensemble_info, desired_doy =\
            request_ensemble_info(year, ensemble_index)
    

    # sys.stdout = open(file_path, "a")
    for week_count,doy in enumerate(desired_doy):
        month = pd.to_datetime(pd.to_datetime(f'{year}-01-01')+pd.Timedelta(f'{doy-1}D')).month
        print(f"Prediction: week of year {week_count+1}, day of year {doy}, month {month}")
        this_prediction_set_values = prediction_set_values.copy()
        this_prediction_set_values['DOY'] = doy
        this_prediction_set_coordinates = prediction_set_coordinates.copy()
        this_prediction_set_coordinates['DOY'] = doy

        this_doy_abundance = pd.Series([np.nan] * this_prediction_set_coordinates.shape[0])
        this_doy_occurance = pd.Series([np.nan] * this_prediction_set_coordinates.shape[0])

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
                        occurance = 0
                    elif correspond_model.startswith("Killed"):
                        abundance = np.nan
                        occurance = np.nan


                elif isinstance(correspond_model, float) or isinstance(correspond_model, int):
                    abundance = correspond_model
                    occurance = 0.5
                    print(f"{stixel['unique_stixel_id']} Using Mean Abundance Across Checklists!")
                else:
                    #### Do the real estimate
                    sample_points = this_prediction_set_values.loc[sample_points_falls_in.index,:]
                    sample_points['time_observation_started_minute_of_day'] = correspond_model.PD_FI['max_value_point']['time_observation_started_minute_of_day']
                    abundance, occurance = correspond_model.predict(sample_points).flatten().tolist()
                    print(abundance, occurance)

                this_doy_abundance[sample_points_falls_in.index] = abundance
                this_doy_occurance[sample_points_falls_in.index] = occurance
        #         print(abundance)
                del abundance
                del occurance


        grid_len = 10
        long_array = np.arange(-17367530.445161372,17367530.445161372,grid_len*1e3)
        lat_array = np.arange(-7342230.13649868,7342230.13649868,grid_len*1e3)

        im_abundance = np.flip(this_doy_abundance.values.reshape(len(lat_array), len(long_array)), axis=0)
        im_occurance = np.flip(this_doy_occurance.values.reshape(len(lat_array), len(long_array)), axis=0)
        im_abundance = np.where(im_occurance>=0.5, im_abundance, 0)
        im_abundance = np.where(im_abundance<0, 0, im_abundance)
        im_abundance = np.where(im_abundance>=0, im_abundance, -1)

        metadata = {'species':species.replace(' ','_'),\
                    'lower_left_point':(-17367530.445161372, -7342230.13649868),\
                   'grid_length_km':grid_len,
                    'grid_shape':(len(long_array), len(lat_array)),
                   'projection':'ESRI:54017'}
        filename1 = f"{output_dir}/{str(user_defined_species_index).zfill(5)}.{species.replace(' ','_')}/{species.replace(' ','_')}_year{year}_ensemble{ensemble_index}_week{week_count}_DOY{doy}_month{month}.abundance.tiff"
        filename2 = f"{output_dir}/{str(user_defined_species_index).zfill(5)}.{species.replace(' ','_')}/{species.replace(' ','_')}_year{year}_ensemble{ensemble_index}_week{week_count}_DOY{doy}_month{month}.occurance.tiff"


        write_geotiff(filename1, im_abundance, metadata)
        write_geotiff(filename2, im_occurance, metadata)
    #     plt.imshow(im, cmap = 'viridis')
    #     plt.show()
        print('done')
    #     sys.stdout.flush()





