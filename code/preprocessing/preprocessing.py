# -*- coding: utf-8 -*-
"""
@author: VictorGueorguiev
"""
#------------------------------------------------------------------------------
### STEP 0: INITIALIZE LIBRARIES
#------------------------------------------------------------------------------

# import necessary pandas and pandas extension pandas-profiling
import pandas as pd
import numpy as np

from preprocessing import load_data
from preprocessing import feature_helpers

#------------------------------------------------------------------------------
### STEP 1: LOAD DATASET USING PANDAS INTO A PANDAS DATAFRAME
#------------------------------------------------------------------------------

PATH_TO_DATA = './../data/'
df_laptops = load_data.load_data_into_dataframe(path_to_data = PATH_TO_DATA)

#------------------------------------------------------------------------------
### STEP 2: DATA CLEANING
#------------------------------------------------------------------------------

# remove unnamed columns which result in error and replace empty strings with NA value
df_laptops = df_laptops[df_laptops.columns.drop(list(df_laptops.filter(regex='Unnamed')))]

df_laptops = df_laptops.replace(r'^\s*$', np.nan, regex=True)
df_laptops['screen_surface'] = df_laptops['screen_surface'].replace('matte', 'Matte')
df_laptops['screen_surface'] = df_laptops['screen_surface'].replace('glossy', 'Glossy')

#------------------------------------------------------------------------------
### STEP 3: FEATURE ENGINEERING
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#### HD RESOLUTION CATEGORY
#------------------------------------------------------------------------------


# Categorize the screen resolution into specific categories such as HD, FullHD, 4K, etc.
df_laptops['resolution_string'] = df_laptops['pixels_x'].astype(str) + 'x' + df_laptops['pixels_y'].astype(str)
df_laptops['hd_resolution_category'] = df_laptops.apply(feature_helpers.hd_resolution_categorizer, axis = 1)

# Test this feature to make sure it is well-distributed for each category, otherwise
# adjust the definition to make it more encompassing

df_laptops.groupby(['hd_resolution_category']).size()

#------------------------------------------------------------------------------
#### SSD CATEGORY
#------------------------------------------------------------------------------

# Categorize the SSD size into specific categories such as Small, Medium and Large, etc.
df_laptops['ssd_category'] = df_laptops.apply(feature_helpers.ssd_categorizer, axis = 1)

# Test this feature to make sure it is well-distributed for each category, otherwise
# adjust the definition to make it more encompassing

df_laptops.groupby(['ssd_category']).size()

#------------------------------------------------------------------------------
#### STORAGE CATEGORY
#------------------------------------------------------------------------------

# Categorize the main storage size into specific categories such as Small, Medium and Large, etc.
df_laptops['storage_category'] = df_laptops.apply(feature_helpers.storage_categorizer, axis = 1)

# Test this feature to make sure it is well-distributed for each category, otherwise
# adjust the definition to make it more encompassing

df_laptops.groupby(['storage_category']).size()

#------------------------------------------------------------------------------
#### CPU FREQUENCY
#------------------------------------------------------------------------------

df_laptops['cpu_frequency'] = df_laptops['cpu_details'].str.extract(r'\s*([0-9].[0-9][0-9])\s*[gG][hH][zZ]')
df_laptops['cpu_frequency2'] = df_laptops['cpu_details'].str.extract(r'\s*([0-9].[0-9])\s*[gG][hH][zZ]')

df_laptops['cpu_frequency'] = df_laptops['cpu_frequency'].replace(np.nan, '0')
df_laptops['cpu_frequency2'] = df_laptops['cpu_frequency2'].replace(np.nan, '0')

df_laptops['cpu_frequency'] = df_laptops['cpu_frequency'].astype(float)
df_laptops['cpu_frequency2'] = df_laptops['cpu_frequency2'].astype(float)
df_laptops['cpu_frequency'] = df_laptops['cpu_frequency'] + df_laptops['cpu_frequency2']

df_laptops = df_laptops.assign(cpu_frequency = [a if a > 0.0 else np.nan for a in df_laptops['cpu_frequency']])
df_laptops = df_laptops.drop(['cpu_frequency2'], axis = 1)
