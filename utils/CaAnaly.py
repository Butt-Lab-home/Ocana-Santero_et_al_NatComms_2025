
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
import statsmodels.stats.multicomp as tukey
import utils.Pandas_analysis as PdA


def CaDataLoad(path, footprint_correction = True, footprint_threshold = 0.2, neuropil_correction = False, np_coeff = 0.7, artefact_correction = True):
    '''
    Function to load calcium imaging data from a specified path. 

    Parameters
    --------------
    path : string with the path to the data (suite2p folder)
    neuropil_correction : boolean, optional, if True corrects for neuropil contamination, default = True
    np_coeff : float, optional, coefficient to multiply the neuropil signal by, default = 0.7
    artefact_correction : boolean, optional, if True removes cells that are all zeros (pure neuropil), default = True

    Returns
    --------------
    Cell_flu : array of shape (n_cells, n_frames) with the fluorescence data of the cells, corrected (or not) by neuropil 
    '''

    #Load iscell data, select first coloumn and convert to integer to use as index
    iscell = np.load(path + 'iscell.npy')[:,0].astype(bool)
    #Load cell and neuropil fluorescence data
    Flu = np.load(path + 'F.npy')[iscell]
    Neuropil = np.load(path + 'Fneu.npy')[iscell]
    #Load stats
    stat = np.load(path + 'stat.npy',allow_pickle=True)[iscell]

    #Apply footprint correction
    if footprint_correction == True:

        footprint = [stat[i]['footprint'] for i in range(len(stat))]
        footprint_bool =(np.array(footprint)>footprint_threshold).astype(bool)

        Flu = Flu[footprint_bool]
        Neuropil = Neuropil[footprint_bool]

    #If artefact correction is true, remove cells that are all zeros (pure neuropil)
    if artefact_correction == True:
        Neuropil = Neuropil[~np.all(Flu == 0, axis=1)]
        Flu = Flu[~np.all(Flu == 0, axis=1)]
    #Return fluorescence data of ONLY cells, corrected (or not) by neuropil
    if neuropil_correction == True:
        Cell_flu = Flu-np_coeff*Neuropil
    else:
        Cell_flu = Flu
    
    return Cell_flu
    
def SpikeLoad(path):
    '''
    Function to load calcium imaging data from a specified path.

    Parameters
    --------------
    path : string with the path to the data (suite2p folder)

    Returns
    --------------
    Spikes : array of shape (n_cells, n_frames) with the spike data of the cells
    
    '''
    #Load iscell data, select first coloumn and convert to integer to use as index
    iscell = np.load(path + 'iscell.npy')[:,0].astype(bool)
    #Load spikes
    Spikes = np.load(path + 'spks.npy')[iscell]
    return Spikes

def CaldF_F(Data, use_baseline = False):
    '''From Raw fluorescence, it calculates dF_F in the Packer way (Double check last statement!)

        Parameters
    -------------
        Data: Raw fluorescence (numpyb array with ROIsxFrames)
        
    
        Returns
    -------------
        dF_F: dF_F trace of all cells (CellsxF), done as dF_F = (F - Fmean)/Fmean (where Fmean is the mean fluorescence
        of the whole trace)
    
    ''' 
    assert type(Data) == np.ndarray, 'this is not a nuympy array' 
    assert Data.ndim == 2, 'Data does not have 2 dims'
    dF_F = ((Data.T - np.mean(Data,axis=1))/np.mean(Data,axis=1)).T
    if use_baseline == True:
        dF_F = ((Data.T - np.mean(Data[:,:36000],axis=1))/np.mean(Data[:,:36000],axis=1)).T
    return dF_F


def CaldF_F_percentile(Data,percentile=1,specific_norm = False,start_norm = 0,stop_norm = 36000):
    '''From Raw fluorescence, it calculates dF_F using the lower x percentile as baseline

        Parameters
    -------------
        Data: Raw fluorescence (numpyb array with ROIsxFrames)
        percentile: percentile to use as baseline, default = 8
        
    
        Returns
    -------------
        dF_F: dF_F trace of all cells (CellsxF), done as dF_F = (F - Fmean)/Fmean (where Fmean is the mean fluorescence
        of the whole trace)
    '''
    assert type(Data) == np.ndarray, 'this is not a nuympy array'
    assert Data.ndim == 2, 'Data does not have 2 dims'
    if specific_norm == True:
        norm = np.percentile(Data[:,~np.all(Data == 0, axis=0)][:,start_norm:stop_norm],percentile,axis=1)
        dF_F = ((Data.T - norm)/norm).T
    else:
        dF_F = ((Data.T - np.percentile(Data[:,~np.any(Data <1, axis=0)],percentile,axis=1))/np.percentile(Data[:,~np.any(Data == 0, axis=0)],percentile,axis=1)).T
    #remove rows with inf (If percentile is exactly 0) --> Which shouldn't exists unless there was an imaging artefact, in which case we don't want to use that cell anyway
    dF_F = dF_F[~np.isinf(dF_F).any(axis=1)]
    return dF_F
    


def Z_score(Data):

    '''Code to calculate z-score trace

        Parameters
    -------------
        Raw_F: Raw fluorescence
    
        Returns
    -------------
        Z_score
    '''
    z_score = np.zeros([Data.shape[0],Data.shape[1]])
    for i in range(0,Data.shape[0]):
        z_score[i,:] = (Data[i,:]-(np.mean(Data[i,:])))/(np.std(Data[i,:]))
    return z_score


def Coordinates(path,):
    '''Function that calculates the x and y coordinates of each ROI

        Parameters
    -------------
        Path: Path where the Stats, F and iscell .npy files are. 


        Returns
    -------------
        Coordinate_x: x coordinates of each ROI
        Coordinate_y: y coordinates of each ROI
    '''
    # Loads data


    #Load iscell data, select first coloumn and convert to integer to use as index
    iscell = np.load(path + 'iscell.npy')[:,0].astype(bool)
    #Load stat file and cell and neuropil fluorescence data
    stat = np.load(path + 'stat.npy', allow_pickle=True)[iscell]
    Flu = np.load(path + 'F.npy')[iscell]
    Neuropil = np.load(path + 'Fneu.npy')[iscell]
    stat = stat[~np.all(Flu == 0, axis=1)]
    Neuropil = Neuropil[~np.all(Flu == 0, axis=1)]
    Flu = Flu[~np.all(Flu == 0, axis=1)]
    footprint = np.zeros(stat.shape[0])
    footprint = [stat[n]['footprint'] for n in range(stat.shape[0])]
    footprint = np.array(footprint)
    threshold_footprint = np.where(footprint > 0.2)
    Cells = Flu[threshold_footprint]
    Neuropil = Neuropil[threshold_footprint]
    stat = stat[threshold_footprint]
    # Calculate dF/F to remove additional cells which have an infinite value
    Data = Cells
    dF_F = ((Data[:,0:35850].T - np.percentile(Data[:,0:35850],1,axis=1))/np.percentile(Data[:,0:35850],1,axis=1)).T
    stat = stat[~np.isinf(dF_F).any(axis=1)]
    #Extract and return coordinates
    Coordinates = [stat[i]['med'] for i in range(stat.shape[0])]
    Coordinates_x = [Coordinates[i][1] for i in range(len(Coordinates))]
    Coordinates_y = [Coordinates[i][0] for i in range(len(Coordinates))]

    return (Coordinates_x, Coordinates_y)


def Event_classifier(dF_F, bin_seconds=3, fr = 30, Print = False):
    '''
    Based on dF_F of all cells calculates developmental synchronous events as H-event (>=80% cells firing),
    L-event (>=20% and <80% cells firing), S-event (<20% and >=1% cells firing) and no-event (<1% cells firing).
    Established in bins of 3s. Active threshold 1 SD. Additional rules: H_event only if previous bin no H_event. 
    L_Event only if previous no L and following bin no H. It also applies a savgol filter (3 order 15 frames)

        Parameters
    -------------
        dF_F: Fluorescent traces normalized (CellsxFrames) 
        bin_seconds: Size of each bin in second, default 3s
        fr: Frame rate, default 30Hz
        Print: To print number of events of each type

        Returns
    -------------
        Type_Event: Array of len number of bins and values 3 if H, 2 if L, 1 if S and 0 if No event.
                    BUT CAREFUL, it includes exclusions!!! So it is more like a onset event array.
        Number_Events: Dictionary with number of events of each type
        Array_Number_Events: np.array([H_events,L_events,S_events,No_events])
        Prob_active: Proportion of bins active each cell (vector of len number of cells)
        Prop_active: Proportion of cells active in each bin (vector of len bin)
    '''

    t = bin_seconds * fr # bin size in frames
    tmp = sps.savgol_filter(dF_F[:,:36000],15,3,axis=1) #Filter signal
    binary = np.zeros([tmp.shape[0],tmp.shape[1]])
    tmp2 = np.std(tmp,axis=1) #Calculate SD to use as activity threshold

    for i in range(tmp.shape[0]):
        binary[i,:] = tmp[i,:]>tmp2[i] #Calculate if cell active in each frame
    
    bins = round(len(tmp.T)/t) #Number of bins
    a = np.zeros([len(binary),bins])
    Type_Event = np.zeros([bins])
    Type_Event[Type_Event==0]=np.nan

    #Calculate if cell active in each bin, if active at any moment
    for ii in range(0,bins):
        for yy in range(len(binary)):
            a[yy,ii] = any(binary[yy,t*ii:t*(ii+1)]>0)

    
    H_events = 0
    L_events = 0
    S_events = 0
    No_events = 0

    for i in range(a.shape[1]-1): #Skipped last bin to be able to use a[i+1] as a rule
        
        #Calculate number of H events
        if np.mean(a[:,i])>=0.8 and np.mean(a[:,i-1])<0.8:
            H_events+=1
            Type_Event[i] = 3

        #Calculate number L events
        if np.mean(a[:,i])>=0.2 and np.mean(a[:,i])<0.8 and np.mean(a[:,i+1])<0.8 and np.mean(a[:,i-1])<0.2:
            L_events+=1
            Type_Event[i] = 2
        
        #Calculate number of S events
        if np.mean(a[:,i])<0.2 and np.mean(a[:,i])>=0.01:
            S_events += 1  #You made this one up to approximate
            Type_Event[i] = 1

        #Calculate number of NO events
        if np.mean(a[:,i])<0.01:
            No_events += 1
            Type_Event[i] = 0

    Number_Events = {'H_events':H_events,'L_events':L_events,'S_events':S_events,'No_events':No_events}
    Array_Number_Events = np.array([H_events,L_events,S_events,No_events])
    Prob_active = np.mean(a,1)        
    Prop_active = np.mean(a,0)
  
    if Print:
        print('H_events: ' + str(H_events))
        print('L_events: ' + str(L_events))
        print('S_events: ' + str(S_events))
        print('No_events: ' + str(No_events))
    
    return(Type_Event,Number_Events,Array_Number_Events,Prob_active,Prop_active)
    

def Event_characterizer(dF_F,t=6,FR=30,single_neuron_threshold='Mean'):
    '''
    Function to localize highly synchronous events based on the mean trace,
    and characterize each of them. Provides maximum dF_F of each event,
    frame location, duration, cell recruitment and start/end of the event.
    Takes ~1 min per 30 min recording at 30Hz.

        Parameters
    -------------
        dF_F: Fluorescent traces normalized (CellsxFrames)
        t: bin size in seconds, default is 6s
        FR: frame rate, default is 30 fr/sec
        single_neuron_threshold: threshold to establish that a single neurons is involved in a event
        

        Returns
    -------------
        Values: a numpy array (EventsxFeatures). Features are in order: max, argmax, duration (s), cell recruitment, 
                start (frame) and stop (frame)
    '''
    bin_size = int(np.round(t*FR))
    bins = (np.round(dF_F.shape[1]/(bin_size))).astype(int)
    count = 0
    Values = np.zeros([100,6])
    Values[Values==0] = np.nan
    threshold_event_edges = (np.mean(np.mean(dF_F,0)))
    if single_neuron_threshold == 'Mean':
        single_neuron_threshold = np.mean(dF_F,1)#+(4*np.std(dF_F,1))
    for bin in range(1,bins):
        mean_before = np.mean(dF_F,0)[(bin-1)*bin_size:bin*bin_size]
        mean_this = np.mean(dF_F,0)[bin*bin_size:(bin+1)*bin_size]
        mean_next = np.mean(dF_F,0)[(bin+1)*bin_size:(bin+2)*bin_size]
        
        if any(mean_this>5): 
            if (mean_before.size != 0) & (mean_next.size != 0):
                if np.max(mean_before)<np.max(mean_this)>np.max(mean_next): 
                    for start in range((bin)*bin_size+np.argmax(mean_this),0,-1): 
                        if np.mean(dF_F[:,start])<threshold_event_edges:
                            break
                    for stop in range((bin)*bin_size+np.argmax(mean_this),dF_F.shape[1]): 
                        if np.mean(dF_F[:,stop])<threshold_event_edges:
                            break
                    Values[count,0] = np.max(mean_this)
                    Values[count,1] = ((bin)*bin_size)+np.argmax(mean_this)
                    Values[count,2] = (stop - start)/FR # Duration in seconds
                    Values[count,3] = np.mean(np.mean(dF_F[:,10500:10700],axis=1)>single_neuron_threshold)#.any(axis =1)) # Participation in that event
                    Values[count,4] = start
                    Values[count,5] = stop
                    count += 1

    return Values[:np.sum(~np.isnan(Values[:,0])),:]  


def paq_read_function(file_path):
    '''
    Code from Paq2Py Lloyd Russel 2015 - https://github.com/llerussell/paq2py/blob/master/paq2py.py
    Reads in .paq file contents, uses data to define global variables, and returns html output
    Arguments: file_path (string) - binary .paq file contents
    Returns: html outputs
    '''

    # open file
    fid = open(file_path, 'rb')

    # get sample rate
    temprate = int(np.fromfile(fid, dtype='>f', count=1))

    # get number of channels
    tempnum_chans = int(np.fromfile(fid, dtype='>f', count=1))
    
    # get channel names
    tempchan_names = []
    for i in range(tempnum_chans):
        num_chars = int(np.fromfile(fid, dtype='>f', count=1))
        chan_name = ''
        for j in range(num_chars):
            chan_name = chan_name + chr(int(np.fromfile(fid, dtype='>f', count=1))) #adding extra integer typecast as numbers were seen as non-ints
        tempchan_names.append(chan_name)
    # get channel hardware lines
    hw_chans = []
    for i in range(tempnum_chans):
        num_chars = int(np.fromfile(fid, dtype='>f', count=1))
        hw_chan = ''
        for j in range(num_chars):
            hw_chan = hw_chan + chr(int(np.fromfile(fid, dtype='>f', count=1)))
        hw_chans.append(hw_chan)

    # get acquisition units
    units = []
    for i in range(tempnum_chans):
        num_chars = int(np.fromfile(fid, dtype='>f', count=1))
        unit = ''
        for j in range(num_chars):
            unit = unit + chr(int(np.fromfile(fid, dtype='>f', count=1)))
        units.append(unit)

    # get data
    temp_data = np.fromfile(fid, dtype='>f', count=-1)
    tempnum_datapoints = int(len(temp_data)/tempnum_chans)
    temp2_data = np.reshape(temp_data, [tempnum_datapoints, tempnum_chans]).transpose()
    
    #if any of the channel names are 'Frame_clock' replace by 'frame_clock'
    for i in range(len(tempchan_names)):
        if tempchan_names[i] == 'Frame_clock':
            tempchan_names[i] = 'frame_clock'
    # close file
    #fid.close()

    #return {"data": data,
     #       "chan_names": chan_names,
     #       "hw_chans": hw_chans,
     #       "units": units,
     #       "rate": rate}
    return{'data':temp2_data,
          'rate':temprate,
          'num_channels':tempnum_chans,
          'channels_names':tempchan_names,
          'hw_chans':hw_chans,
          'units': units}

def paq_extract(paq,channel_to_Extract, height=2.1, distance=500, distance_stim = 10000):
    '''
    Obtains the frames in which a stimulus was presented

    Parameters
    --------------------
    paq: dictionary with the paq data and channel names
    channel_to_Extract: string with the name of the channel to extract (e.g.,'Whisker_in' or 'Sound_in')
    height: height of the peak to detect (default = 2.1)
    distance: minimum distance between peaks for frame clock (default = 500)
    distance_stim: minimum distance between peaks for stims (default = 10000)

    Returns
    --------------------
    stim_frames: list with the frames in which the stimulus was presented

    '''
    stim_frames = []
    #if any of the channels is called frame_clock, then use that one
    index_frame_clock = np.where(np.array(paq['channels_names']) == 'frame_clock')[0][0]
    index_channel_to_Extract = np.where(np.array(paq['channels_names']) == channel_to_Extract)[0][0]
    frames_times =sps.find_peaks(paq['data'][index_frame_clock], height=height, distance=distance)[0]
    #print(frames_times.shape)
    stim_times = sps.find_peaks(paq['data'][index_channel_to_Extract], height=height, distance=distance_stim)[0]
    for i in range(len(stim_times)):
        #frame_stim = min(frames_times, key=lambda x:abs(x-stim_times))
        frame_stim = np.argmin(np.abs(frames_times-stim_times[i]))
        stim_frames.append(frame_stim)
    return stim_frames


def OLDpaq_extract(paq,channel_to_Extract, height=2.1, distance=500, distance_stim = 10000):
    '''
    Obtains the frames in which a stimulus was presented

    Parameters
    --------------------
    paq: dictionary with the paq data and channel names
    channel_to_Extract: string with the name of the channel to extract (e.g.,'Whisker_in' or 'Sound_in')
    height: height of the peak to detect (default = 2.1)
    distance: minimum distance between peaks for frame clock (default = 500)
    distance_stim: minimum distance between peaks for stims (default = 10000)

    Returns
    --------------------
    stim_frames: list with the frames in which the stimulus was presented

    '''
    stim_frames = []
    #if any of the channels is called frame_clock, then use that one
    index_frame_clock = np.where(np.array(paq['channels_names']) == 'frame_clock')[0][0]
    index_channel_to_Extract = np.where(np.array(paq['channels_names']) == channel_to_Extract)[0][0]
    frames_times =sps.find_peaks(paq['data'][index_frame_clock], height=height, distance=distance)[0]
    #print(frames_times.shape)
    stim_times = sps.find_peaks(paq['data'][index_channel_to_Extract], height=height, distance=distance_stim)[0]
    for i in range(len(stim_times)):
        #frame_stim = min(frames_times, key=lambda x:abs(x-stim_times))
        frame_stim = np.argmin(np.abs(frames_times-stim_times[i]))
        stim_frames.append(frame_stim)
    return stim_frames

def frame_count(paq_read, recording_length = 5,frame_rate=30.18):
    '''
    Provided the output of paq_read_function, it will calculate the number of the frame in which there was a frame captured by the scope
    
    Parameters
    --------------------
    paq_read: Output of paq_read_function (dict)
    recording_length: Length of the recording in minutes (int)


    Returns
    --------------------
    frame_count: Number of frames captured by the scope (int)
    '''
    frame_count = 0
    frame_count_2 = 0
    thresh = 0.2
    tmp_i = 0
    tmp = paq_read['data'][1,:]
    for i in range(paq_read['data'].shape[1]):
        if tmp[i]>thresh and tmp[i-1]<thresh: #Only works if the frame clock is the second channel and the signal is a step
            frame_count += 1
    print(frame_count)
    if frame_count > (recording_length*60*frame_rate+300):
        frame_count = 0
        print('WARNING: frame_count is longer than expected') # this is because packIO was left running after a scan was aborted and the frame clock was still running
        # iterate backwards through the frame clock to find the recording
        for i in range(paq_read['data'].shape[1]-1,0,-1):
            if tmp[i]>thresh and tmp[i-1]<thresh:
                frame_count +=1
                
                if frame_count != 1 and ((tmp_i-i) > ((paq_read['rate']/frame_rate)*5)):
                    true_i = tmp_i
                    paq_read['data'] = paq_read['data'][:,true_i:]
                    break
                tmp_i = i
        print(frame_count)
        return frame_count, paq_read
        
    else:
        return frame_count, paq_read


def PeriStimFrames(dF_F,Whisker_in_fn,minus_fr=2,plus_fr=6,hz=30):
    '''
    These functions obtains the frames peri-whisker stimulation. minus_fr (default 2s) establish the seconds pre stim to obtain,
    plus_fr (default 6s) establishes the number of frames post stim to take. The default values are thought for GCaMP data. HZ (default 30 frames) is the 
    recording frequency, i.e., frames per second. dF_F is the F-Fmean/Fmean array with ROIs and Whisker_in_fn is the output of paq_extract. It says whiskers but you can
    provide data from other modalities to calculate any kind of peri stim response

        Parameters
    -------------
        dF_F: CellsxFrames dF_F values
        Whisker_in_fn: Stim onset frame numbers
        minus_fr= default 2s, prestim trace to calculate
        plus_fr= default 6s, postim trace to calculate
        hz= recording frequency 30 by default in Rig3 at 512x
    
        Returns
    -------------
        Whisker_in_stim_resp: Numpy array with Cells x Stim_repeats x frames 
    '''

    whisker_in_stim_resp = np.zeros([len(dF_F),len(Whisker_in_fn),int(np.round((minus_fr)*hz))+int(np.round((plus_fr)*hz))])
    whisker_in_stim_resp[whisker_in_stim_resp==0]=np.nan
    #cont_resp = np.zeros([10,2400])


    for yy in range(len(Whisker_in_fn)):
        if Whisker_in_fn[yy]-(int(np.round(minus_fr*hz)))<0:
            #print('Skip first stim because not enough frames before stim')
            continue
        else: 
            whisker_in_stim_resp[:,yy,:] = dF_F[:,Whisker_in_fn[yy]-(int(np.round(minus_fr*hz))):Whisker_in_fn[yy]+(int(np.round(plus_fr*hz)))]
            #cont_resp[yy,:] = dF_Fmean[rnd[yy]-600:rnd[yy]+1800]

    return whisker_in_stim_resp



def MeanResponsivenessT(whisker_in_stim_resp, p_val = 0.05, bonferroning = True, PerTrial = False):
    '''
    Uses a t-test to calculate whether there is a significant difference in the signal 2 seconds prior to stimulation vs 2 seconds after stimulation.
    Parameters
    -------------
    whisker_in_stim_resp: Numpy array with Cells x Stim_repeats x frames - established using previous function
    p_val: significance level for t-test
    bonferroning: whether to correct for multiple comparisons
    PerTrial: whether to calculate the mean response across all stimulation trials, or calculate whether there is a significant response in each cell to each stimulation - when True can calculate the proportion of cells which reliably respond.
    Returns
    -------------
    Propor: proportion of cells which respond to stimulation
    or WresponsingYOO: p-values for each cells mean response
    '''
    meanarray = np.zeros((whisker_in_stim_resp.shape[0],whisker_in_stim_resp.shape[1],2))
    # Mean response ~2 seconds prior to stimulation
    meanarray[:,:,0] = np.nanmean(whisker_in_stim_resp[:,:,20:50],axis=2)
    # Mean response ~2 seconds after stimulation with a 10 frame delay
    meanarray[:,:,1] = np.nanmean(whisker_in_stim_resp[:,:,60:90],axis=2)
    if PerTrial == True:
        WresponsingYOO = np.zeros([len(whisker_in_stim_resp),whisker_in_stim_resp.shape[1]])
        #Wresp = np.zeros([len(WresponsingYOO)])
        for i in range(0, len(whisker_in_stim_resp)):
            for y in range(0,whisker_in_stim_resp.shape[1]):
                [a,b] = scipy.stats.ttest_rel(whisker_in_stim_resp[i,y,20:50],whisker_in_stim_resp[i,y,60:90])
                WresponsingYOO[i,y]= b
        if bonferroning:
            p_Values = np.reshape(WresponsingYOO,-1)
            pv= multipletests(p_Values[:], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
            p_Values_sorted = np.reshape(pv[1],(len(WresponsingYOO),len(WresponsingYOO.T)))
            tmp = p_Values_sorted<p_val
            Propor=sum(tmp.T)/(len(p_Values_sorted.T))
        #Sum(Wresp>0.5) #Gives you the number of neuron that are responsive to more than 50% of the stim trials
        return(Propor)
    else:
        WresponsingYOO = np.zeros([len(whisker_in_stim_resp)])
        for i in range(0, len(whisker_in_stim_resp)):
            mean1: np.nanmean(whisker_in_stim_resp[i,:,0:60],1)
            [a,b] = scipy.stats.ttest_rel(meanarray[i,:,0][~np.isnan(meanarray[i,:,0])],meanarray[i,:,1][~np.isnan(meanarray[i,:,1])])
            WresponsingYOO[i]= b
        if bonferroning:
            p_Values = WresponsingYOO
            pv= multipletests(p_Values[:], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
            tmp = pv[1]<p_val
            WresponsingYOO = pv[1]
            #Propor=sum(tmp.T)/(len(p_Values_sorted.T))
        #Sum(Wresp>0.5) #Gives you the number of neuron that are responsive to more than 50% of the stim trials
        return(WresponsingYOO)


def MeanResponsivenessU(whisker_in_stim_resp, p_val = 0.05, pre_window_1 = 20, pre_window_2 = 50, post_window_1 =  60, post_window_2 = 90, bonferroning = True, PerTrial = False):
    '''
    Uses a Mann-Whitnney U test to calculate whether there is a significant difference in the signal 2 seconds prior to stimulation vs 2 seconds after stimulation.
    Parameters
    -------------
    whisker_in_stim_resp: Numpy array with Cells x Stim_repeats x frames - established using previous function
    p_val: significance level for t-test
    bonferroning: whether to correct for multiple comparisons
    PerTrial: whether to calculate the mean response across all stimulation trials, or calculate whether there is a significant response in each cell to each stimulation - when True can calculate the proportion of cells which reliably respond.
    Returns
    -------------
    Propor: proportion of cells which respond to stimulation
    or WresponsingYOO: p-values for each cells mean response
    '''
    meanarray = np.zeros((whisker_in_stim_resp.shape[0],whisker_in_stim_resp.shape[1],2))
    # Mean response ~2 seconds prior to stimulation
    meanarray[:,:,0] = np.nanmean(whisker_in_stim_resp[:,:,pre_window_1:pre_window_2],axis=2)
    # Mean response ~2 seconds after stimulation with a 10 frame delay
    meanarray[:,:,1] = np.nanmean(whisker_in_stim_resp[:,:,post_window_1:post_window_2],axis=2)
    if PerTrial == True:
        WresponsingYOO = np.zeros([len(whisker_in_stim_resp),whisker_in_stim_resp.shape[1]])
        #Wresp = np.zeros([len(WresponsingYOO)])
        for i in range(0, len(whisker_in_stim_resp)):
            for y in range(0,whisker_in_stim_resp.shape[1]):
                [a,b] = scipy.stats.wilcoxon(whisker_in_stim_resp[i,y,pre_window_1:pre_window_2],whisker_in_stim_resp[i,y,post_window_1:post_window_2])
                WresponsingYOO[i,y]= b
        if bonferroning:
            p_Values = np.reshape(WresponsingYOO,-1)
            pv = multipletests(p_Values[:], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
            p_Values_sorted = np.reshape(pv[1],(len(WresponsingYOO),len(WresponsingYOO.T)))
            tmp = p_Values_sorted<p_val
            Propor=sum(tmp.T)/(len(p_Values_sorted.T))
        #Sum(Wresp>0.5) #Gives you the number of neuron that are responsive to more than 50% of the stim trials
        return(Propor)
    else:
        WresponsingYOO = np.zeros([len(whisker_in_stim_resp)])
        for i in range(0, len(whisker_in_stim_resp)):
            [a,b] = scipy.stats.wilcoxon(meanarray[i,:,0][~np.isnan(meanarray[i,:,0])],meanarray[i,:,1][~np.isnan(meanarray[i,:,1])])
            
            WresponsingYOO[i]= b
        response = np.mean(meanarray[:,:,1],1)-np.mean(meanarray[:,:,0],1)
        #Make all negative responses -1 and all positive responses 1
        response[response<0] = -1
        response[response>0] = 1
        if bonferroning:
            p_Values = WresponsingYOO
            pv= multipletests(p_Values[:], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
            #tmp = pv[1]<p_val
            WresponsingYOO = pv[1]
            #Propor=sum(tmp.T)/(len(p_Values_sorted.T))
        #Sum(Wresp>0.5) #Gives you the number of neuron that are responsive to more than 50% of the stim trials
        return(WresponsingYOO,response)


def PupMarkerMovements(path,marker_num):
    from numpy import genfromtxt
    '''
    Provide the path to the csv deeplabcut file and the number of markers, and it will output an array with the euclidean distances from t to t+1 for each marker,
    the average movement of each marker during that recording and the confidence that DLC has on that marker
    '''
    Pup = genfromtxt(path, delimiter=',')
    Markers = np.zeros([marker_num,len(Pup)-4]) #Change 'k' for a more intuitive name 
    for k in range(0,marker_num):
        tmp = 'Marker_' + str(k)
        movement = np.zeros([marker_num])
        confidence =  np.zeros([marker_num])
        globals()[tmp] = np.zeros([len(Pup)-4,1])
        for i in range(3,len(Pup)-1):
            (globals()[tmp])[i-3,0] = np.array(np.linalg.norm(Pup[i,1+3*k:2+3*k]-Pup[i+1,1+3*k:2+3*k]))
        Markers[k,:]= (globals()[tmp])[:,0]
        movement[k] = np.mean((globals()[tmp]))
        confidence[k] = np.mean(Pup[3:,3+k*3])
    return(Markers,movement,confidence)


def Sleep_Scorer(Marker,SD_scale = 0.5, mov_threshold = 0.5, bin_size = 6, FR = 30):
    '''
    Sleep scorer based on left forelimb movement. Rules: Removes movements < SD_scale*SDs, bins of bin_size sec,
    awake if moving for more than mov_threshold (sec) on that bin, Quiet sleep (QS) if not moving at all,
    Active sleep (AS) if moving but less than mov_threshold (sec). AS canot follow awake and bins of REM cannot 
    be separated by a single bin of QS, minimu two (bin_size*2 s)

        Parameters
    -------------
        Marker = Numpy vector with forelimb movement
        SD_scale = Scales S.D. to remove movements smaller than SD_scale*SD (default 0.5)
        mov_threshold = Time in seconds spent moving during that bin to be considered awake (default 0.5s)
        bin_size = Size of the bin in seconds (default 6s)
        FR = Frame rate recording (default 30)

        Returns
    -------------
        Sleep = numpy vector, with lenght number of bins, where 0 is awake, 1 QS and 2 AS
    '''  

    Marker[Marker<=(np.std(Marker)*SD_scale)]=0 #Denoise
    binary = Marker[:]>0 #Activity threshold established by eye
    t = bin_size * FR #Bin size in frames
    bins = round(len(Marker)/t) #Number of bins

    Awake = np.zeros([bins]) 
    AS = np.zeros([bins])
    QS = np.zeros([bins])
    
    for ii in range(0,bins):
        Awake[ii] = np.mean(binary[t*ii:t*(ii+1)])>= ((mov_threshold*FR)/t) 
        if Awake[ii]!=1 and any(binary[t*ii:t*(ii+1)]) and np.mean(binary[t*ii:t*(ii+1)])<0.08 and Awake[ii-1]!=1:
            AS[ii]=1
        if Awake[ii]!=1 and AS[ii]!=1:
            QS[ii]=1
        if AS[ii]==1 and AS[ii-2]==1:
            QS[ii-1]=0
            AS[ii-1]=1
            Awake[ii-1]=0

    Sleep = np.zeros([bins])
    Sleep[Sleep==0]=np.nan
    Sleep[Awake==1] = 0
    Sleep[QS==1] = 1
    Sleep[AS==1] = 2

    return Sleep
    

def tSNEing(Tmp,tSNE=True):
    '''Code to perform dimensionality reduction with initial PCA followed by tSNE on the 'n' features (.shape[1]) (e.g.,time points) 
       of a seires of rows (.shape[0]) (e.g., cells or mice).

       Parameters
    -------------
        Tmp: Rows x Features array, the second dimension gets reduced.
        tSNE: if True tSNE follows PCA
    
    Returns
    -------------
        SNE : values for each row element along its two dimensions of maximal variability
    '''
    resp_lowd = PCA(n_components=min(2, Tmp.shape[1]), random_state=0).fit_transform(Tmp)
    if tSNE:
        resp_lowd = TSNE(n_components=2, random_state=0).fit_transform(resp_lowd)
    x, y = resp_lowd[:, 0], resp_lowd[:, 1]
    SNE = np.array([x,y]) 
    return SNE


def Cluster(SNE,n_clusters):
    '''
    Classifies ROIs in clusters

        Parameters
    -------------
        SNE: 2D data with x and y coordinates from previous dimensionality reduction (e.g., with tSNE)
        n_clusters: Number of clusters 

    
        Returns
    -------------
        A scatter plot with every cell coloured by cluster (colours generated at random)
    '''
    distances = np.zeros([n_clusters])
    distances[distances==0]=np.nan
    clus = sc.k_means(SNE.T,n_clusters=n_clusters)[0]
    cluster= np.zeros([SNE.shape[1]])
    for i in range(0,SNE.shape[1]):
        for y in range(0,n_clusters):
            distances[y]=np.linalg.norm(SNE[:,i]-clus[y,:])
        cluster[i]=np.argmin(distances)

    return cluster


def LogReg(X_train, X_test,y):
    '''Code to perform Logistic regression for neuronal decoding

        Parameters
    -------------
        X_train: training data
        y: labels
    
    Returns
    -------------
        accuracy: proportion of rightly decoded for test data
    '''
    log_reg = LogisticRegression(penalty="l2", solver='saga',max_iter=5000)
    log_reg.fit(X_train, y)
    y_pred = log_reg.predict(X_test)
    accuracy = (y == y_pred).mean()
    return accuracy


def Entropy(Firing_Prob):
    '''Calculates entropy based on firing probability of each cell

        Parameters
    -------------
        Firing_Prob: A vector with the firing probability of each cell
    
    Returns
    -------------
        entropy: as per shannon entropy, measure of information
    '''
    counts, _ = np.histogram(Firing_Prob)
    pmf = counts / np.sum(counts)
    pmf = pmf[pmf > 0]
    entropy = np.abs(-np.sum(pmf * np.log2(pmf)))
    return entropy


def NeuropilCorrelation(path):
    '''
    Function to calculate mean correlation between each cell and its neuropil

    Parameters
    --------------
    path : string with the path to the data (suite2p folder)


    Returns
    --------------
    Mean Neuropil_corr : Mean correlation between each cell and its neuropil
    '''

    #Load iscell data, select first coloumn and convert to integer to use as index
    iscell = np.load(path + 'iscell.npy')[:,0].astype(int)
    #Load cell and neuropil fluorescence data
    Flu = np.load(path + 'F.npy')[iscell]
    Neuropil = np.load(path + 'Fneu.npy')[iscell]
    #Calculate correlation between each cell and its neuropil
    Neuropil_corr = np.zeros(Flu.shape[0])
    for i in range(Flu.shape[0]):
        Neuropil_corr[i] = np.corrcoef(Flu[i,:],Neuropil[i,:])[0,1]
    return np.mean(Neuropil_corr)


def FindLabelled_Cells(dF_F,lf=500,pct=99):
    '''
    This function finds the cells that are positive for tdTomato in the final frames of the recording.
        Parameters
    -------------
        dF_F: Neuron traces
        lf: number of frames from the end which contain the tdTomato signal
        pct: which percentile of the brightest cells shall be considered?
    
        Returns
    -------------
        tdtom_idx: index of the cells that are positive for tdTomato, an array of integers
    
    '''
    Histo = np.mean(dF_F[:,-lf:-1],1)
    tdtom_idx = np.where(Histo >= np.percentile(Histo,pct))[0]
    return tdtom_idx

def FindNOTLabelled_Cells(dF_F,lf=500,pct=1):
    '''
    This function finds the cells that are positive for tdTomato in the final frames of the recording.
        Parameters
    -------------
        dF_F: Neuron traces
        lf: number of frames from the end which contain the tdTomato signal
        pct: which percentile of the brightest cells shall be considered?
    
        Returns
    -------------
        tdtom_idx: index of the cells that are positive for tdTomato, an array of integers
    
    '''
    Histo = np.mean(dF_F[:,-lf:-1],1)
    NOtdtom_idx = np.where(Histo <= np.percentile(Histo,pct))[0]
    return NOtdtom_idx

def Population_entropy(dF,bin_size=30):
    '''
    Calculate the entropy of a population of neurons

    Parameters
    ---------------
    dF: array of fluorescence traces (np.array CellsxFrames)
    bin_size: size of the bin in frames

    Returns
    ---------------
    entropy: entropy of the population

    '''

    #Find the number of bins
    n_bin = int(np.floor(dF.shape[1]/bin_size))
    num_frames = n_bin*bin_size
    

    #Binning
    dF_reshaped = np.reshape(dF[:,:num_frames],(dF.shape[0],n_bin,bin_size))
    dF_binned = np.mean(dF_reshaped,axis=2)
    
    #assert np.all(dF.mean(1) == 0), f'Cells are not zero centered'
    #Binarize by 1 S.D.
    threshold = np.std(dF,1).reshape(dF.shape[0],1)
    dF_binned[dF_binned>=threshold]=1
    dF_binned[dF_binned<threshold]=0
    
    #Calculate state probability
    probability = np.unique(dF_binned,return_counts=True,axis=1)[1]/np.sum(np.unique(dF_binned,return_counts=True,axis=1)[1])

    #calculate entropy
    entropy = -np.sum(probability*np.log2(probability),axis=0)

    return entropy


def sliding_window(data, window_size = 50, step_size = 1):
    """
    Inputs:
        data - 1D array of data
        window_size - size of window to compute mean
        step_size - step size to move window
    Outputs:
        values - array of summed values for each window
        
    """
    values = []

    for i in range(0, len(data), step_size):
        if i+window_size < len(data):
            values.append(np.sum(data[i:i+window_size]))
            
        else:
            values.append(np.mean(data[i:]))
         
    return np.array(values)


def find_frame_length(path):
    '''
    Function to find the number of frames in PV backup .xml file

    Parameters
    ---------------
    path: path to the folder where the .xml file can be found

    Returns
    ---------------
    n_frames: number of frames in the recording

    '''

    path = path + path[53:-1] + '_BACKUP.xml'
    with open(path, 'r') as file:
        xmlstr = file.read()
    return int(xmlstr[xmlstr.rfind('<ExtraParameters')-25:xmlstr.rfind('<ExtraParameters')-19])

def find_frame_rate(path):
    '''
    Function to find the frame rate of the recording based on the PV .env file

    Parameters
    ---------------
    path: path to the folder where the .env file can be found

    Returns
    ---------------
    framerate: frame rate of the recording

    '''

    path = path + path[53:-1] + '.env'
    with open(path, 'r') as file:
        envstr = file.read()
    return float(envstr[envstr.find('framerate')+18:envstr.find('framerate')+31])


def mode(array):
    '''
    Function to find the mode of an array

    Parameters
    ---------------
    array: array of values

    Returns
    ---------------
    mode: mode of the array

    '''

    vals,counts = np.unique(array,return_counts=True)
    return vals[np.argmax(counts)]

import pandas as pd
import numpy as np
from scipy import stats

import scikit_posthocs as sp
def stating(df,parameter,genotypes,alpha = 0.05, Fisher = False):
    '''
    Function to perform statistical analysis on a dataframe with genotypes and a parameter
    with permutation testing if it's not normally distributed

    Parameters
    ---------------
    df: dataframe with genotypes and parameter
    parameter: parameter to be analysed
    genotypes: genotypes to be compared
    alpha: Value of FWER at which to calculate HSD, i.e., probability
            of making a false discovery due to multiple comparisons, default 0.05

    Returns
    ---------------
    prints the results of the statistical analysis

    '''

    df = df[df['genotype'].isin(genotypes)]
    print('Two way ANOVA results:')
    print(PdA.two_way_ANOVA(df,parameter))
    #selec only genotypes in genotypes
    shap = stats.shapiro(df[parameter])[1]
    print('Shapiro p-val:' +str(shap))
    if shap<=0.05:
        
        no_param = PdA.permutation_test(df,parameter)
        print('Permutation test results:')
        print(no_param)
        if no_param[1] < 0.05:
            #remove none and nan
            df = df[df[parameter].notnull()]
            if Fisher:
                print('Fisher LSD test results:')
                print(fishers_lsd_from_df(df, parameter, 'genotype', alpha=alpha))
                #perm_LSD = PdA.LSD_permutation_test(df, parameter,factor2 = 'genotype', num_permutations=1000, test_statistic_func = fishers_lsd_from_df)
                #print(perm_LSD)
            else:
                if shap>=0.05:
                    print('Tukey HSD test results:')
                    print(tukey.pairwise_tukeyhsd(endog = df[parameter], groups= df['genotype'],alpha=alpha))
                else:
                    print(sp.posthoc_dunn(df, val_col=parameter, group_col='genotype'))#, p_adjust= ‘fdr_by’ ))

    return 


def Peri_H_event(path,df,row):
    '''
    Function to get the peri H-event dF/F for a given row of the data frame

    Parameters
    --------------
    path: string
        Path to the data
    df: pandas dataframe
        Dataframe with the data
    row: int
        Row of the dataframe to plot
    
    Returns
    --------------
    periH: numpy array
        Array with the peri H-event dF/F for all the H-events in the row
    
    '''
    data = CaDataLoad(path)
    dF_F = CaldF_F_percentile(data)

    Peak_frames = df.iloc[row]['Events'][0][:,1]
    Peak_frames_H = Peak_frames[df.iloc[row]['Events'][0][:,3]>=0.8]

    #For each value in Peak_frames_H, obtain the 300 frames before and after 

    periH = np.zeros((len(Peak_frames_H),dF_F.shape[0],300))
    for i in range(len(Peak_frames_H)):
        periH[i,:] = dF_F[:,int(Peak_frames_H[i]-150):int(Peak_frames_H[i]+150)]
    
    return periH


'''------------------------------------------------------------------------------------------------------------------------------'''
'''                                                 FUNCTION ARCHIVE                                                             '''
'''------------------------------------------------------------------------------------------------------------------------------'''

def OLD_paq_extract(tmp, thresh = 0.2, only_AUD = False):
    '''
    Provided the output of paq_read_function, it will calculate the number of the frame in which there was a stim onset,
    as an array called channel name + _fn, not flexible btw stim types yet.

        Parameters
    --------------------
        Path_paq: Provides the path to the .paq file

        Returns
    --------------------
        Channel_name_fn: An array with frame numbers corresponding to stim onset.

    '''
    paq_data = tmp['data']
    paq_names = tmp['channels_names']
    paq_num_chan = tmp['num_channels']
    for y in range(paq_num_chan):
        tmp=paq_data[y]
        trig_times = []  #Maybe tmp.T_?
        tmp2 = 'trig_times_' + str(y)
        for i in range(len(paq_data[0])):
            if tmp[i]>thresh and tmp[i-1]<thresh:
                trig_times.append(i)
                globals()[tmp2] = trig_times
                
    #Finds The frames where whisker is stimulated! (1)
    tmp = 'trig_times_' + str(paq_names.index('frame_clock'))
    for y in range(0,1): #change for paq_num_cha
        if paq_names[y] != 'frame_clock' and paq_names[y] != 'Eye_tracking':
            tmp1 = 'trig_times_' + str(y)
            tmp2 = paq_names[y] + '_fn' #Frames Numbers
            tmp3=[]
            globals()[tmp2] = []
            for ii in range(len(globals()[tmp1])): #aa #globals is saying that this string is a variable, i'm amazed that this actually works!
                tmp4 = (globals()[tmp1])[ii]
                tmp5 = min((globals()[tmp]), key=lambda x:abs(x-tmp4)) #Frame Clock, this one shouldn't change
                tmp6 = (globals()[tmp]).index(tmp5)
                tmp3.append(tmp6)
                globals()[tmp2] = tmp3
            print(len((globals()[tmp1])))
    #Patch to make Sound stim work since it's not a step but an oscillation... You can probably do better BUT it works! 
    if tmp2 == 'Sound_in_fn':
        Sound_unique = np.unique(globals()[tmp2]) 
        #Sound_in_fn = np.zeros([20-len(Whisker_in_fn)]) #Good only for your normal whisker/aud paradigm
        Sound_in_fn = np.zeros([len(Sound_unique)])
        Sound_in_fn[Sound_in_fn==0] = np.nan
        count=0
        for i in range(0,len(Sound_unique)):
            if i==0:
                Sound_in_fn[count]=Sound_unique[i]
                count+=1
            else:
                if Sound_unique[i]-1 != Sound_unique[i-1]:
                    Sound_in_fn[count]=Sound_unique[i]
                    count+=1   
        Sound_in_fn = Sound_in_fn.astype(int)
        
        if only_AUD:
            return(Sound_in_fn)
        else:
            return(Whisker_in_fn, Sound_in_fn) #Output number of frames where stim onset, as name of channel _fn

    else:
        return Whisker_in_fn
    

def OLD_PupSleepScoring(Markers,t=90):
    '''
    Trinarizing sleep in pups: Event classifier into Sleep, Awake or Twitch with same data binning as for H,L,S event classifier.
    It requires the an array with the eucclidean distance between t and t+1 for all markers and t is binning size in frames, 
    default is 90 frames as in H, L, S event, ~3 seconds. It provides the classification based on each marker. Four arrays of
    MarkersxBins: Awake, twiching, sleeping non twitching and total sleep
    '''
    binary = Markers[:]>1 #Activity threshold established by eye
    t = 90 #Binning frames, event time window 30Hz
    bins = round(Markers.shape[1]/t)
    a = np.zeros([len(binary),bins]) 
    b = np.zeros([len(binary),bins])
    c = np.zeros([len(binary),bins])
    d = np.zeros([len(binary),bins])
    for ii in range(0,bins):
        for yy in range(len(binary)):
            a[yy,ii] = (sum(binary[yy,t*ii:t*(ii+1)])/t)>0.3
            if any(binary[yy,t*ii:t*(ii+1)]) and (sum(binary[yy,t*ii:t*(ii+1)])/t)<0.3:
                b[yy,ii]=1
            if a[yy,ii]!=1 and b[yy,ii]!=1:
                c[yy,ii]=1
            if b[yy,ii]==1 or c[yy,ii]==1:
                d[yy,ii]=1
    return{'Awake' : a, 
          'Twitching' : b,
          'Sleep_non_twitching' : c,
          'Sleep_total' : d}


def OLD_EventClassi(dF_F,t=90):
    '''
    Event classifier into High synchronicity, Low synchronicity or Sparce activity with data BINING (First two from Leighton et al. (2021), Last one I made it up). 
    It outputs, respectively, the proportion of cells firing in each data bin (to compare to behaviour), the number of H, L and S events and the average number of
    calcium events per cell in that recording
    '''
    binary = dF_F>2.2 #Activity threshold established by eye
    #t = 90 #Binning frames, event time window 30Hz
    bins = round(len(dF_F.T)/t)
    a = np.zeros([len(binary),bins])
    for ii in range(0,bins):
        for yy in range(len(binary)):
            a[yy,ii] = any(binary[yy,t*ii:t*(ii+1)]>0)
    Prob_firing = np.mean(a,1)        
    Prop_firing = sum(a)/len(a)
    H_events = sum((sum(a)/len(a))>=0.8) 
    L_events = sum(((sum(a)/len(a))>=0.2) & ((sum(a)/len(a))<0.8))
    S_events = sum(((sum(a)/len(a))<0.2)) #You made this one up to approximate 
    Events_cell = sum(sum(a))/len(binary) #Average number of events per cell, after binning
    #tmp_str = 'There are ' + str(H_events) + ' H events, ' + str(L_events) + ' L events, ' + str(S_events) + ' S events and the average number of events per cell is ' + str(round(Events_cell)) + '.'
    return(Prop_firing,H_events,L_events,S_events,Events_cell)