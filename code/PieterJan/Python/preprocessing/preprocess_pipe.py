# base
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# own functions
from preprocessing.feature_helpers import hd_resolution_categorizer
from preprocessing.feature_helpers import ssd_categorizer
from preprocessing.feature_helpers import storage_categorizer



def print_missing(df, ascending=False):
    
    
    """
    prints the count and percentage of missing columns

    Parameters
    ----------
    df : Pandas Data Frame
        the dataframe containing the features
    ascending: Boolean
        determines how to sort the missing columns
        

    Returns
    -------
    Pandas Data Frame
        with the count and % of columns containing missing values

    """
    
    missing_count = df.isnull().sum().sort_values(ascending=ascending)
    missing_count = missing_count[missing_count>0]
    missing_perc = round(missing_count*100/len(df),2)
    
    return pd.DataFrame.from_dict({'missing count':missing_count, 
                                   'missing %':missing_perc})


                
    
class MakeLowerCase(BaseEstimator, TransformerMixin):
        
    """
    lower cases all columns containing str based features   
    """

    def __init__(self):
        # to store columns that were lower cased
        self.lower_case = []

    def fit(self, X, y=None):
        
        return self

    def transform(self, X):
        
        for c in X.columns:
            if X[c].dtypes == object:
                X[c] = X[c].str.lower()
                self.lower_case.append(c)
        return X

class HdResolutionCategorizer(BaseEstimator,TransformerMixin):
    
    """
    make a hd_resolution_category feature based on  resolution_string and pixels_x
    """
    
    def __init__(self, drop_columns=None):
        
         self.drop_columns = drop_columns
        
    def fit(self, X, y=None):
        
        return self
        

    def transform(self, X):
        
        X['resolution_string'] = X.loc[:,'pixels_x'].astype(str).copy() + 'x' + X.loc[:,'pixels_y'].astype(str).copy()
        new_feature = X.apply(hd_resolution_categorizer, axis=1)
        X = X.assign(hd_resolution_category=new_feature.values) 

        return X.drop(self.drop_columns, axis=1)   
    
    
class SsdCategorizer(BaseEstimator,TransformerMixin):
    
    """
    Categorize the SSD size into specific categories such as Small, Medium and Large, etc.
    """
    
    def __init__(self, drop_original_feature=True):
        
        self.drop_original_feature=drop_original_feature
        
    def fit(self, X, y=None):
        
        return self
        

    def transform(self, X):
              
        X['ssd_category'] = X.apply(ssd_categorizer, axis=1)
                
        if self.drop_original_feature:
            return X.drop(['ssd'], axis=1)
        else:
            return X

       
class StorageCategorizer(BaseEstimator,TransformerMixin):
    
    """
    Categorize the main storage size into specific categories such as Small, Medium and Large, etc.
    """
    
    def __init__(self, drop_original_feature=True):
        
        self.drop_original_feature=drop_original_feature
        
    def fit(self, X, y=None):
        
        return self
        

    def transform(self, X):
              
        X['storage_category'] = X.apply(storage_categorizer, axis=1)
        
        if self.drop_original_feature:
            return X.drop(['storage'], axis=1)
        else:
            return X

        


    
            
def calculate_perf(y_true, y_hat):
        
    """
    computes the performance of the prediction model

    Parameters
    ----------
    y_hat : numpy array or pandas data frame
        N*2 array of predictions 
        for minimum price and maximum price (in sequence)
    y_true: numpy array or pandas data frame
        N*2 array of the true valus 
        for minimum price and maximum price (in sequence)

    Returns
    -------
        dictionary
            error on the minimum price, 
            maximum price and total error (sum of the two)
    """

    # convert to numpy array
    if isinstance(y_hat, pd.DataFrame):
        y_hat = y_hat.values
    
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    
    
    error_min_p = np.mean(abs(y_hat[:,0] - y_true[:,0])) 
    error_max_p = np.mean(abs(y_hat[:,1] - y_true[:,1]))

    return  {"minimum price":error_min_p,
             "maximum price":error_max_p,
             "total error": (error_min_p + error_max_p)}
           
    
def custom_scoring_func(y_true, y_hat):
    
    """
    computes the performance of the prediction model

    Parameters
    ----------
    y_hat : numpy array or pandas data frame
        N*2 array of predictions 
        for minimum price and maximum price (in sequence)
    y_true: numpy array or pandas data frame
        N*2 array of the true valus 
        for minimum price and maximum price (in sequence)

    Returns
    -------
    float
        performance
    
    """
    
    # convert to numpy array
    if isinstance(y_hat, pd.DataFrame):
        y_hat = y_hat.values
    
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    
    error_min_p = np.mean(abs(y_hat[:,0] - y_true[:,0])) 
    error_max_p = np.mean(abs(y_hat[:,1] - y_true[:,1]))
    
    return (error_min_p + error_max_p)
 
    



