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

#------------------------------------------------------------------------------
### STEP 1: LOAD DATASET USING PANDAS INTO A PANDAS DATAFRAME
#------------------------------------------------------------------------------

def hd_resolution_categorizer(df):
    if df['resolution_string'] in ["2304x1440", "2560x1600", "2880x1800"]:
        return 'Retina'
    elif df['pixels_x'] >= 1200 and df['pixels_x'] <= 1600:
        return 'HD'
    elif df['pixels_x'] == 1920:
        return 'FullHD'
    elif df['pixels_x'] > 1920 and df['pixels_x'] < 3840:
        return 'QHD/UHD'
    elif df['pixels_x'] == 3840:
        return '4K'
    else:
        return 'SD'
    
def ssd_categorizer(df):
    if df['ssd'] == 0:
        return 'None'
    elif df['ssd'] < 64:
        return 'Small'
    elif df['ssd'] <= 256:
        return 'Medium'
    else:
        return 'Large'
    
def storage_categorizer(df):
    if df['storage'] == 0:
        return "None"
    elif df['storage'] <= 256:
        return 'Small'
    elif df['storage'] <= 1028:
        return 'Medium'
    elif df['storage'] <= 2056:
        return 'Large'
    else:
        return 'Very Large'
    