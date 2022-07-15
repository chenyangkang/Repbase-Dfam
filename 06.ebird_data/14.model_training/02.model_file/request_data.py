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
import time

def request_data(year,species,ensemble_index):
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
   'entropy']
    
    #### Change the column names.
    pd.DataFrame(qresult.isnull().sum(axis=0)).T
    
    with_ = sum(qresult[f'{species.replace(" ","_")}_indiv_count']>0)
    without_ = len(qresult) - with_
    print(f'''For raw data, No. of Checklists with {species.replace(" ","_")}: {with_}, No. of Checklists without {species.replace(" ","_")}: {without_},\n
    {round((with_/(with_+without_))*100, 2)}% checklists has {species.replace(" ","_")}''')
    
    return qresult


