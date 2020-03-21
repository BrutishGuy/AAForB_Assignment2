# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:46:35 2020

@author: VictorGueorguiev
"""
#------------------------------------------------------------------------------
### STEP 0: INITIALIZE
#------------------------------------------------------------------------------

# for saving the report to file
from pathlib import Path
# import necessary pandas and pandas extension pandas-profiling

import numpy as np
import pandas as pd
import pandas_profiling

#------------------------------------------------------------------------------
### STEP 1: LOAD DATASET USING PANDAS INTO A PANDAS DATAFRAME
#------------------------------------------------------------------------------

df_laptops = pd.read_csv("./data/train.csv")

#------------------------------------------------------------------------------
### STEP 2: DATA CLEANING
#------------------------------------------------------------------------------

# remove unnamed columns which result in error and replace empty strings with NA value
df_laptops = df_laptops[df_laptops.columns.drop(list(df_laptops.filter(regex='Unnamed')))]

# convert pixels into categorical variable (we want to see those kinds of statistics, average screen size is a bit meaningless)
df_laptops['pixels_y'] = df_laptops['pixels_y'].astype(str)
df_laptops['pixels_x'] = df_laptops['pixels_x'].astype(str)

df_laptops = df_laptops.replace(r'^\s*$', np.nan, regex=True)

#------------------------------------------------------------------------------
### STEP 3: GENERATE PANDAS PROFILING REPORT
#------------------------------------------------------------------------------

# generate report
profile = df_laptops.profile_report(title="AA4B - Assignment 2 Initial Leptop Dataset Report")
# this will save an HTML report in you current working directory
profile.to_file(output_file=Path("./output/initial_dataset_profiling_report.html"))
