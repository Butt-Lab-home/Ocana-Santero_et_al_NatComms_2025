
import numpy as np
import scipy
from scipy import stats
import statsmodels
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sklearn.cluster as sc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import scipy.signal as sps
import statsmodels.api as sm

def exponential_fitting_detrend(data):
    x = np.arange(0,data.shape[0])
    x = sm.add_constant(x)
    model = sm.OLS(data,x)
    results = model.fit()
    return (data - results.fittedvalues) + np.nanmean(results.fittedvalues)

def function_detrend_both(data,data2):
    x = np.arange(0,data.shape[0])
    x = sm.add_constant(x)
    model = sm.OLS(data,x)
    results = model.fit()
    CH2 = (data - results.fittedvalues) + np.nanmean(results.fittedvalues)
    CH3 = (data2 - results.fittedvalues) + np.nanmean(results.fittedvalues)
    return CH3/CH2


def GRAB_dataload(Path, Path_ter ='_unreg.csv',only_CH3=False,detrend_CH2=False,detrend_both=True,return_only_ratio=False):
    '''
    Load GRAB data
    
    Parameters
    -----------------
    Path: str
        Path to the folder containing the GRAB data
        only_CH3: bool, If True, only CH3 data is returned, default False

    Returns
    -----------------
    ratio_nor: array
        Normalized ratio of green and red channel
    '''

    CH3 = np.genfromtxt(Path + '/CH3' + Path_ter, delimiter=',')
    if only_CH3:
        ratio = CH3[1:,2]
        if return_only_ratio:
            return ratio
        else:
            return (ratio - np.nanmean(ratio))/np.nanmean(ratio)
    
    else:
        CH2 = np.genfromtxt(Path + '/CH2' + Path_ter, delimiter=',')
        if detrend_CH2:
            CH2 = exponential_fitting_detrend(CH2[1:,2])
            ratio = CH3[1:,2]/CH2
            return (ratio - np.nanmean(ratio))/np.nanmean(ratio)
        if detrend_both:
            ratio = function_detrend_both(CH2[1:,2],CH3[1:,2])
        else:
            ratio = CH3[1:,2]/CH2[1:,2]
        ratio_nor = (ratio - np.nanmean(ratio))/np.nanmean(ratio)

        if return_only_ratio:
            return ratio
        else:
            return ratio_nor
    
def GRAB_regions_dataload(Path, Path_ter ='_regions_unreg.csv',num_regions=256,only_CH3=False,detrend_CH2=False,detrend_both=True):
    '''
    Load GRAB data from regions

    Parameters
    -----------------
    Path: str
        Path to the folder containing the GRAB data
    Path_ter: str
        Termination of the GRAB data file
    num_regions: int
        Number of regions in the GRAB data
    only_CH3: bool, If True, only CH3 data is returned, default False
    detrend_CH2: bool, If True, detrend CH2 data, default False
    detrend_both: bool, If True, detrend both channels, default True

    Returns
    -----------------
    ratio_nor: array
        Normalized fluorescence, either CH3 or CH3/CH2
    '''

    CH3 = np.genfromtxt(Path + '/CH3' + Path_ter, delimiter=',')
    CH3 = np.reshape(CH3[1:,2], (num_regions,int(CH3[1:,2].shape[0]/num_regions)))

    if only_CH3:
        baseline = np.nanmean(CH3,1)
        baseline = baseline[:,np.newaxis] #To vectorize the operation
        return ((CH3 - baseline)/baseline)
    else:
        CH2 = np.genfromtxt(Path + '/CH2' + Path_ter, delimiter=',')
        CH2 = np.reshape(CH2[1:,2], (num_regions,int(CH2[1:,2].shape[0]/num_regions)))
        if detrend_CH2:
            CH2_detrend = np.zeros((CH2.shape[0],CH2.shape[1]))
            for i in range(CH2.shape[0]):
                CH2_detrend[i] = exponential_fitting_detrend(CH2[i])
            ratio = CH3/CH2_detrend
            baseline = np.nanmean(ratio,1)
            baseline = baseline[:,np.newaxis] #To vectorize the operation
            return ((ratio - baseline)/baseline)
        if detrend_both:
            ratio = np.zeros((CH2.shape[0],CH2.shape[1]))
            for i in range(CH2.shape[0]):
                ratio[i] = function_detrend_both(CH2[i],CH3[i])
        else:
            ratio = CH3/CH2
        
        baseline = np.nanmean(ratio,1)
        baseline = baseline[:,np.newaxis]
        return (ratio - baseline)/baseline
    
def PeriStimFrames(dF_F,frames,regions = False,minus_fr=2,plus_fr=6,hz=30):
    '''
    These functions obtains the frames peri-whisker stimulation. minus_fr (default 2s) establish the seconds pre stim to obtain,
    plus_fr (default 6s) establishes the number of frames post stim to take. The default values are thought for GCaMP data. HZ (default 30 frames) is the 
    recording frequency, i.e., frames per second. dF_F is the F-Fmean/Fmean array with ROIs and Whisker_in_fn is the output of paq_extract. It says whiskers but you can
    provide data from other modalities to calculate any kind of peri stim response

        Parameters
    -------------
        dF_F: Frames dF_F values
        frames: Stim onset frame numbers
        minus_fr= default 2s, prestim trace to calculate
        plus_fr= default 6s, postim trace to calculate
        hz= recording frequency 30 by default in Rig3 at 512x
    
        Returns
    -------------
        stim_resp: Numpy array with Cells x Stim_repeats x frames 
    '''

    if regions:
        stim_resp = np.zeros([dF_F.shape[0],len(frames),int((minus_fr+plus_fr)*hz)]) # Regions x Stim_repeats x frames
        for yy in range(len(frames)):
            try:
                stim_resp[:,yy,:] = dF_F[:,frames[yy]-(int(minus_fr*hz)):frames[yy]+(int(plus_fr*hz))]
            except:
                pass
    else:
        stim_resp = np.zeros([len(frames),int((minus_fr+plus_fr)*hz)])
        stim_resp[stim_resp==0]=np.nan
        for yy in range(len(frames)):
            try:
                stim_resp[yy,:] = dF_F[frames[yy]-(int(minus_fr*hz)):frames[yy]+(int(plus_fr*hz))]  
            except:
                pass
    
    return stim_resp


