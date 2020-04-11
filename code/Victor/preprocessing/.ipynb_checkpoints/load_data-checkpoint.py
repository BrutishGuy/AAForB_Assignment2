# -*- coding: utf-8 -*-
"""
@author: VictorGueorguiev
"""
#------------------------------------------------------------------------------
### STEP 0: INITIALIZE LIBRARIES
#------------------------------------------------------------------------------

# import necessary pandas and pandas extension pandas-profiling
import pandas as pd

#------------------------------------------------------------------------------
### STEP 1: LOAD DATASET USING PANDAS INTO A PANDAS DATAFRAME
#------------------------------------------------------------------------------

def load_data_into_dataframe(path_to_data):
    df_laptops = pd.read_csv(path_to_data + "train.csv")
    return df_laptops
