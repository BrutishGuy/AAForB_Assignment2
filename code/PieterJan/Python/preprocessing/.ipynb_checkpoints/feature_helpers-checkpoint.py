import pandas as pd
import numpy as np



def hd_resolution_categorizer(df):
    
    """


    Parameters
    ----------
    df : Pandas Data Frame
        the dataframe containing the features
    
    Returns
    -------
    Pandas Data Frame

    """
    
    
    if df['resolution_string'] in ["2304x1440", "2560x1600", "2880x1800"]:
        return 'retina'
    elif df['pixels_x'] >= 1200 and df['pixels_x'] <= 1600:
        return 'hd'
    elif df['pixels_x'] == 1920:
        return 'fullhd'
    elif df['pixels_x'] > 1920 and df['pixels_x'] < 3840:
        return 'qhd/uhd'
    elif df['pixels_x'] == 3840:
        return '4k'
    else:
        return 'sd'
    
def ssd_categorizer(df):
    
    """


    Parameters
    ----------
    df : Pandas Data Frame
        the dataframe containing the features
    
    Returns
    -------
    Pandas Data Frame

    """
    
    if df['ssd'] == 0:
        return 'none'
    elif df['ssd'] < 64:
        return 'small'
    elif df['ssd'] <= 256:
        return 'medium'
    else:
        return 'large'
    

def storage_categorizer(df):
    
    """


    Parameters
    ----------
    df : Pandas Data Frame
        the dataframe containing the features
    
    Returns
    -------
    Pandas Data Frame

    """
    
    if df['storage'] == 0:
        return "none"
    elif df['storage'] <= 256:
        return 'small'
    elif df['storage'] <= 1028:
        return 'medium'
    elif df['storage'] <= 2056:
        return 'large'
    else:
        return 'very large'
    

