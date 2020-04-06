import pandas as pd
import numpy as np

def weights_samples(df=None, order=1, plot_weights=False):
    
    """
    computes the weight given to each observations

    Parameters
    ----------
    df : Pandas Data Frame
        containing the target variable (only min price or max price)
    order: integer
        determines the weights (higher more weight is given to higher prices)
    plot_weights: Boolean
        plots the weights   

    Returns
    -------
    Pandas Data Series
        
    """
    
    
    if order==0:
        # return equal weights
        weights = pd.Series(np.ones(len(df))/len(df))     
    elif order>0:
        weights = (df**order)/sum(df**order)
    if plot_weights:
        # visualize weights
        weights.plot()
    # check if sum is 1
    print(f"Sum weights: {round(weights.sum(),4)}")
    return weights