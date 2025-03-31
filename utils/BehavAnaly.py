# Behaviour analysis
import os
import glob
import numpy as np

def movement_extract(dates, animals, recordings, DLC_model_name, path = '//qnap-amp001.dpag.ox.ac.uk/gsantero/Data/Behavioural_Recordings/All_csvs/', print_confidence = False, confidence_filter = False, confidence_threshold = 0.9):
    '''
    Extracts the movement from the behavioural recordings of the GRAB experiment
    Parameters
    ---------------
    path: string
        Path to the folder containing the data
    dates: list
        List of the recording dates e.g., ['2023-07-15']
    animals: list
        List of strings containing the animal numbers e.g., ['440','441']
    recordings: list
        List of strings containing the recording numbers e.g., ['t_001','t_002']
    DLC_model_name: string
        Name of the DLC model used for the tracking (e.g., 'DLC_resnet50_Pup_Tracking_P7-11_SSRIs_2023-06-27Jun27shuffle1_500000.csv')
    print_confidence: bool
        If True prints the mean confidence of the tracking, default = False
    confidence_filter: bool
        If True uses last confident coordinates in low confince points, default = False

    Returns
    ---------------
    mov: array
        Vector containing the movement of the animal between frame t and t+1, of length n-1 where n is the number of frames
    '''

    for date in dates:
        for animal in animals:
            for recording in recordings:
                path_behav = path + date + '-GOS' + animal + '-' + recording + '*' 
                #Replace all _ with -
                #path_behav = path_behav.replace('_', '-')
     
                full_path = glob.glob(path_behav) #To find the recording without specifying the exact time of the recording
         
                #Replace all 
                full_path = [s for s in full_path if DLC_model_name in s][0] #To find the recording with the correct DLC model

                assert os.path.exists(full_path), "Something went worng, the path to the recording does not exist"
                #Load coordinates
                coords = np.genfromtxt(full_path, delimiter=',')[3:,:]
                #Print mean confidence and std of the tracking
                if print_confidence:
                    print('Mean confidence of' + date + '_' + animal + recording + ':' +  + str(np.mean(coords[:,3])) + ' +/- ' + str(np.std(coords[:,3])))
                #Filter out low confidence points
                if confidence_filter:
                    coords[0,3] = 1 #To avoid the first point to be replaced, since there is no previous point and initial position is not important
                    for i in range(len(coords[:,3])):
                        if coords[i,3] < confidence_threshold:
                            coords[i,1:3] = coords[i-1,1:3] # Not worth vectorizing it beause if low confidence points are next to each other it will not work
                #Calculate movement
                mov = np.linalg.norm(coords[:-1,1:3]-coords[1:,1:3],axis=1)
                
                return mov
            
def movement_for_ANN(dates, animals, recordings, DLC_model_name, path = '//qnap-amp001.dpag.ox.ac.uk/gsantero/Data/Behavioural_Recordings/GOSB17_3/', weight_marker = False, print_confidence = False, confidence_filter = False, confidence_threshold = 0.8):
    '''
    Extracts the movement from the behavioural recordings of the GRAB experiment
    Parameters
    ---------------
    path: string
        Path to the folder containing the data
    dates: list
        List of the recording dates e.g., ['2023-07-15']
    animals: list
        List of strings containing the animal numbers e.g., ['440','441']
    recordings: list
        List of strings containing the recording numbers e.g., ['t_001','t_002']
    DLC_model_name: string
        Name of the DLC model used for the tracking (e.g., 'DLC_resnet50_Pup_Tracking_P7-11_SSRIs_2023-06-27Jun27shuffle1_500000.csv')
    print_confidence: bool
        If True prints the mean confidence of the tracking, default = False
    confidence_filter: bool
        If True uses last confident coordinates in low confince points, default = False

    Returns
    ---------------
    mov: array
        Vector containing the movement of the animal between frame t and t+1, of length n-1 where n is the number of frames
    '''

    for date in dates:
        for animal in animals:
            for recording in recordings:
                path_behav = path + date + '-GOS' + animal + '-' + recording + '*' 
                #print(path_behav)
                full_path = glob.glob(path_behav) #To find the recording without specifying the exact time of the recording
                #Replace all the backslashes with forward slashes
                #full_path = [s.replace('\\', '/') for s in full_path]
                #print(full_path)
                full_path = [s for s in full_path if DLC_model_name in s][0] #To find the recording with the correct DLC model
                #print(full_path)
                assert os.path.exists(full_path), "Something went worng, the path to the recording does not exist"
                #Load coordinates
                coords = np.genfromtxt(full_path, delimiter=',')[3:,:]
                #Print mean confidence and std of the tracking
                if print_confidence:
                    print('Mean confidence of' + date + '_' + animal + recording + ':' +  + str(np.mean(coords[:,3])) + ' +/- ' + str(np.std(coords[:,3])))
                #Filter out low confidence points
                # (i.e., if unsure where the marker is, presume it didn't move)
                if confidence_filter:
                    coords[0,3] = 1 #To avoid the first point to be replaced, since there is no previous point and initial position is not important
                    coords[0,6] = 1 #To avoid the first point to be replaced, since there is no previous point and initial position is not important
                    coords[0,9] = 1 #To avoid the first point to be replaced, since there is no previous point and initial position is not important
                    for i in range(len(coords[:,3])):
                        if coords[i,3] < confidence_threshold:
                            coords[i,1:3] = coords[i-1,1:3] # Not worth vectorizing it beause if low confidence points are next to each other it will not work
                            coords[i,4:6] = coords[i-1,4:6] # Not worth vectorizing it beause if low confidence points are next to each other it will not work
                            coords[i,7:9] = coords[i-1,7:9] # Not worth vectorizing it beause if low confidence points are next to each other it will not work
                #Calculate movement
                mov1 = np.linalg.norm(coords[:-1,1:3]-coords[1:,1:3],axis=1) # left FL
                mov2 = np.linalg.norm(coords[:-1,4:6]-coords[1:,4:6],axis=1) # right FL
                mov3 = np.linalg.norm(coords[:-1,7:9]-coords[1:,7:9],axis=1)  # nose
                confindece1 = coords[:,3]
                confindece2 = coords[:,6]
                confindece3 = coords[:,9]
                
                if weight_marker:
                    mov1 = mov1 * 2
                    mov2 = mov2 * 1
                    mov3 = mov3 * 1
                return np.vstack((mov1, mov2, mov3)), np.vstack((confindece1[:-1], confindece2[:-1], confindece3[:-1]))


def sleep_broadcast(sleep, frames):
    '''     
    Broadcasts the sleep array to match the length of the movement array

    Parameters
    ---------------
    sleep: array
        Array containing the sleep score for each epoch
    frames: array
        Array containing the frame number at which each epoch ends
    
    Returns
    ---------------
    sleep_array: array
        Array containing the sleep score for each frame
    '''

    sleep_array = np.zeros(frames[-1])
    for i in range(len(frames)):
        if i == 0:
            sleep_array[0:frames[i]] = sleep[i]
        else:
            sleep_array[frames[i-1]:frames[i]] = sleep[i]
    return sleep_array

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
    #Normalized marker by mean
    Marker = Marker/np.nanmean(Marker)
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
    