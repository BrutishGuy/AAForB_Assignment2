from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

def plot_predictions_results(ax, y_true, y_pred, title, log_scale=False, 
                             xlab = "Truth", ylab= "Predicted", color_scatter="black"):
    """Scatter plot of the predicted vs true targets."""
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            '--r', linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.2, color=color_scatter)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    scores = (r'$R^2={:.2f}$' + '\n' + r'$MAE={:.2f}$').format(r2_score(y_true, y_pred),
             mean_absolute_error(y_true, y_pred))
    ax.legend([extra], [scores], loc='upper left')
    ax.set_title(title)
    if log_scale:
        ax.set_yscale('log', basey=10)
        ax.set_xscale('log', basex=10)
        
        

def plot_residuals (ax, y_true, y_pred, log_scale, ylab="Residuals", 
                    xlab= "Observations", order=1, title="", color_scatter="black"):
    """plots residuals"""
    residuals = y_true - y_pred
    ax.scatter(x=np.arange(0,len(y_true)),y=residuals**order, alpha=0.2, color=color_scatter)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_title(title)
    if log_scale:
        ax.set_yscale('log', basey=10)