#Functions for adult analysis

import numpy as np
import utils.CaAnaly as CaaN
import utils.BehavAnaly as Ba
import utils.GRABAnaly as Gr
import utils.Pandas_analysis as PdA
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import scipy.stats as stats
import scikit_posthocs as sp

def stim_index(row,stim):
    '''
    Given the row of the adult data frame and the stim name
    it returns the index of the stim in the row

    Parameters
    -----------------
    row: row of the adult data frame
    stim: string with the name of the stim

    Returns
    -----------------
    index: int
        index of the stim in the row
    '''
    #subselect stim order info
    stim_order = row[7:14]
    #find index of stim
    return int(stim_order[stim_order == stim].index[0][-1])-1

def periFing(row,interneurons = False,responsive = False):
    '''
    Very specific function to extract peri-stimulus fluorescence from adult data
    with the problematic ideosyncracies of this particular data set

    Parameters
    -----------------
    row: row of the adult data frame
    interneurons: bool, if True, it returns the peri-stimulus fluorescence of the interneurons, default False
    responsive: bool, if True, it returns the peri-stimulus fluorescence of the responsive cells, default False

    Returns
    -----------------
    PeriF: array
        Peri-stimulus fluorescence of the adult data

    '''
    path = row['filepath']
    stims = ['Single whisker','Baseline','Rough','Multiwhisker','Smooth','Air puff','Sound']
    #Load data single animal
    #print(path)
    data = CaaN.CaDataLoad(path+'1/suite2p/plane0/')
    dF_F = CaaN.CaldF_F(data)
    PeriF = np.zeros([7,dF_F.shape[0],10,670])
    if responsive:
        PeriF = np.zeros([7,2,dF_F.shape[0],10,670])
    PeriF[PeriF == 0] = np.nan
    
    
    #Extract number of frames per recording
    num_frames = []
    for t in range(1,8):
        path_t = path + str(t) + '/'
        try:
            num_frames.append(CaaN.find_frame_length(path_t))
        except:
            num_frames.append(8963) #The duration of most recordings
    #substract one to all the frames to make them start at 0
    num_frames = np.array(num_frames) - 1
    #print(num_frames)

    for n in range(7):
        stim = stims[n]
        index = stim_index(row,stim) #Which recording was this stim
        #path = '//qnap-amp001.dpag.ox.ac.uk/gsantero/Data/ + 2023-05-30/2023-05-30_GOS346_t-00' + str(n) + '/'
        path_N = path + str(index+1) +'/'  #'1'#/suite2p/plane0/'
        
        
        Rec = path_N[-6:-1]
        #replace the last '-' with a '_'
        Rec = Rec.replace('-', '_')
        path_paq = path_N[:53] + path_N[53:-6] + Rec + '.paq'
        #print(path_paq)
        if (stim == 'Baseline') and (os.path.isfile(path_paq) == False) or path_paq == '//qnap-amp001.dpag.ox.ac.uk/gsantero/Data/2023-06-22/2023-06-22_GOS377_t_002.paq':
            Rec = path_N[-6:-2] + '1'
            Rec = Rec.replace('-', '_')
            path_paq = path_paq = path_N[:53] + path_N[53:-6] + Rec + '.paq'
        #print(path_paq)
        #print(path_paq)
        paq_read = CaaN.paq_read_function(path_paq)
        #print(n)
        if stim == 'Sound':
            try:
                times = CaaN.paq_extract(paq_read,'Sound_in')
            except:
                print('At ' + stim + ' from ' + row['animal'] + ', the sound channel was off -.-')
                continue
        else:
            times = CaaN.paq_extract(paq_read,'Whisker_in')

        try:
            hz = CaaN.find_frame_rate(path_N)
        except:
            hz= 30 #The rounded frame rate
        
        minus_fr = 2
        plus_fr = 20
        #print(stim)
        #print(times)
        #print(index)
        #print(n)
        #print(dF_F.shape)
        #print(dF_F[:,np.sum(np.array(num_frames[:index])):].shape)
        #print(dF_F[:,np.sum(np.array(num_frames[:index])):])
        #print(hz)
        #print(num_frames)

        #if times empty continue
        if len(times) == 0:
            print('At ' + stim + ' from ' + row['animal'] + ', there are no trials')
            continue
        
        #if the recording is shorter than the length of this trial, continue and print a warning
        if np.any(dF_F.shape[1] < (np.sum(np.array(num_frames[:index+1]))-1800)):
            print('At ' + stim + ' from ' + row['animal'] + ' stopped, because the times are bigger than the number of frames,\n maybe you forgot to run a trial in suite2p?')
            continue

        resp = CaaN.PeriStimFrames(dF_F[:,np.sum(np.array(num_frames[:index])):],times,hz=hz,minus_fr=minus_fr,plus_fr=plus_fr)
        #print(path_paq)
        #Find peri-stim frames
        if responsive:
            
            rep, sign = CaaN.MeanResponsivenessU(resp,PerTrial=False,pre_window_1 = 30, pre_window_2 = 45,post_window_1 = 75, post_window_2 = 90)
            try:
                positive_resp = resp[rep<0.05,:,:][sign[rep<0.05]==1,:,:],axis=(1)
                negative_resp = resp[rep<0.05,:,:][sign[rep<0.05]==-1,:,:],axis=(1)
                PeriF[n,0,:positive_resp.shape[0],:(resp[:,:10,:]).shape[1],:resp.shape[2]] = positive_resp[:,:10,:]
                PeriF[n,1,:negative_resp.shape[0],:(resp[:,:10,:]).shape[1],:resp.shape[2]] = negative_resp[:,:10,:]
            except:
                pass
        else:
            PeriF[n,:,:(resp[:,:10,:]).shape[1],:resp.shape[2]] = resp[:,:10,:]
        #remove raws with all nans
        #trial_number= [0,1,2,3,4,5,6][:]
        #target_labels = [trial_number[i] for i in range(len(trial_number)) for j in range(10)]
        #target_labels = target_labels[~np.isnan(PeriF).all(axis=(1,2,3))]
        #print(PeriF.shape)
        #PeriF = PeriF[~np.isnan(PeriF).all(axis=(3))]
        #print(PeriF.shape)
    if interneurons:
        int_index = CaaN.FindLabelled_Cells(dF_F,pct = 95)
        no_int_index = CaaN.FindNOTLabelled_Cells(dF_F,pct = 5)

        return (PeriF[:,int_index,:,:],PeriF[:,no_int_index,:,:])
        
        
    return PeriF

#def combined(path):
    #try:
     #   PeriF = periFing(path)
      #  #print(PeriF)
       # PC1 = PCAing(PeriF[:,:,:,:300])
        #accuracy = Stim_decoding(PC1)
        #return accuracy
    #except:
    #    print('Something went wrong with ' + path)
    #    return np.nan





def folder_iteration(df_data,new_data_label,function):
    '''
    Iterates through your data frame and extracts fluorescence data, then applies function2 to that data
    and it includes it in a new coloumn of your data framed called the value of new_data_label

    Parameters
    ------------------------
    df_data: data frame with one recording day per row
    new_data label: string with the name of your new data extracted (e.g., dF_F fluorescent trace)
    function2: a function with everything that is to be done to extract this new data (e.g. CaaN.CaldF_F())
    funtion1: by default it extracts the fluorescent trace (i.e., CaaN.DataLoad), but it could load other types 
                of data like behavioural if the function is changed

    Returns
    ------------------------
    df_data: data frame with a new coloumn, labelled with the value of 'new_data_label, that contains the new data extracted

    '''

    df_data[new_data_label] = None
    for i, row in df_data.iterrows():
        if os.path.exists(row['filepath']+'1/suite2p/plane0/') and os.path.isfile(row['filepath']+'1/suite2p/plane0/'+'F.npy') and os.path.isfile(row['filepath']+'1/suite2p/plane0/'+'iscell.npy'):
            #try:
                #print(i)
            data = function(row)
            df_data.at[i,new_data_label] = data
            #except:
                #df_data.at[i,new_data_label] = np.nan

        else:
            print(f'Data from animal {row["animal"]} on {row["Date"]} not found.')

    return df_data


def split_by_mouse(responses_all,geno_list):

    '''
    Splits data in training and testing with subject wise cross validation, using a 60% training and 40% testing split.

    Parameters
    -----------------
    responses_all: array
        Array containing the responses of all the mice, shape (n_mice, n_stimuli, n_trials, n_neurons, n_frames)
    geno_list: list
        List containing the genotype of each mouse

    Returns
    -----------------
    X_train: array
        Array containing the responses of the mice used for training, shape (n_mice, n_stimuli, n_trials, n_neurons, n_frames)
    X_test: array
        Array containing the responses of the mice used for testing, shape (n_mice, n_stimuli, n_trials, n_neurons, n_frames)
    y_train: array
        Array containing the labels of the mice used for training, shape (n_mice,)
    y_test: array
        Array containing the labels of the mice used for testing, shape (n_mice,)
    
    '''
    
    
    # Randomly choose 3/5 of the mice for training and 2/5 for testing of each genotype
    # Create a list of the indices of each genotype
    WT_idx = [i for i, x in enumerate(geno_list) if x == 'WT']
    HET_idx = [i for i, x in enumerate(geno_list) if x == 'HET']
    KO_idx = [i for i, x in enumerate(geno_list) if x == 'KO']
    Control_idx = [i for i, x in enumerate(geno_list) if x == 'Control']
    SSRI_idx = [i for i, x in enumerate(geno_list) if x == 'SSRI']

    # Randomly choose 3/5 of the mice for training and 2/5 for testing of each genotype
    WT_train = np.random.choice(WT_idx, int(len(WT_idx)*0.6), replace=False)
    WT_test = [x for x in WT_idx if x not in WT_train]
    HET_train = np.random.choice(HET_idx, int(len(HET_idx)*0.6), replace=False)
    HET_test = [x for x in HET_idx if x not in HET_train]
    KO_train = np.random.choice(KO_idx, int(len(KO_idx)*0.6), replace=False)
    KO_test = [x for x in KO_idx if x not in KO_train]
    Control_train = np.random.choice(Control_idx, int(len(Control_idx)*0.6), replace=False)
    Control_test = [x for x in Control_idx if x not in Control_train]
    SSRI_train = np.random.choice(SSRI_idx, int(len(SSRI_idx)*0.6), replace=False)
    SSRI_test = [x for x in SSRI_idx if x not in SSRI_train]

    # Concatenate all the training and testing indices
    train_idx = np.concatenate((WT_train,HET_train,KO_train,Control_train,SSRI_train))
    #Make labels by repeating each genotype the length of the train_idx
    y_train = np.repeat('WT',len(WT_train))
    y_train = np.concatenate((y_train,np.repeat('HET',len(HET_train))))
    y_train = np.concatenate((y_train,np.repeat('KO',len(KO_train))))
    y_train = np.concatenate((y_train,np.repeat('Control',len(Control_train))))
    y_train = np.concatenate((y_train,np.repeat('SSRI',len(SSRI_train))))

    test_idx = np.concatenate((WT_test,HET_test,KO_test,Control_test,SSRI_test))
    #Make labels by repeating each genotype the length of the train_idx
    y_test = np.repeat('WT',len(WT_test))
    y_test = np.concatenate((y_test,np.repeat('HET',len(HET_test))))
    y_test = np.concatenate((y_test,np.repeat('KO',len(KO_test))))
    y_test = np.concatenate((y_test,np.repeat('Control',len(Control_test))))
    y_test = np.concatenate((y_test,np.repeat('SSRI',len(SSRI_test))))

    X = responses_all[1:].copy()
    X_train = X[train_idx,:,:,:,:]
    X_test = X[test_idx,:,:,:,:]

    return X_train, X_test, y_train, y_test




def decoding(X,y,test_size=0.25,max_iter=100):
    '''
    Stimulus classifier based on neuron activity using logistic regression

    Parameters
    -----------------
    X: array
        Array containing the features, shape (n_sampes, n_features)
    y: array
        Array containing the labels, shape (n_samples,)
    test_size: float, default 0.25
        Fraction of the data to be used for testing
    max_iter: int, default 100
        Maximum number of iterations for the logistic regression

    Returns
    -----------------
    accuracy: float
        Accuracy of the classifier on the test set
    '''

    # X is a feature matrix of shape (n_samples, 2000)
    # y is a target vector of shape (n_samples,)

    # Split data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Create an instance of LogisticRegression
    log_reg = LogisticRegression(max_iter=max_iter, penalty='l2', C = 0.1, multi_class='ovr')

    # Fit the model on the training data
    log_reg.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred = log_reg.predict(X_test)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    #print(f'Accuracy: {accuracy}')

    return accuracy

def bootstrapping(responses_all, iters = 1000):
    '''
    Performs bootstrapping on the data to calculate an accuracy distribution, structured for
    genotype classification. Prints every 100 iterations, to check progress. 

    Parameters
    -----------------
    responses_all: array
        Array containing the responses of all the mice, shape (n_mice, n_stimuli, n_trials, n_neurons, n_frames)
    iters: int, default 1000
        Number of iterations for the bootstrapping (Multiplied by 7 as we keep some stimulus structure
        to confirm randomization, i.e., there shouldn't be major differences between the accuracies of
        the different stimuli, otherwise (1) shuffling is problematic or (2) the number of iterations is
        too low)

    Returns
    -----------------
    accuracies: array
        Array containing the accuracy distribution, shape (n_stimuli, iters)

    '''
    accuracies = np.zeros([7, 1000])
    for y in range(iters):
        # Randomly permute the cells across different mice
        responses_ALL = responses_all[:,:,:,:,:].copy()
        
        # Reshape the array for easier shuffling
        reshaped_array = responses_ALL.reshape(-1, responses_all.shape[2])

        # Shuffle each slice independently
        for i in range(reshaped_array.shape[0]):
            np.random.shuffle(reshaped_array[i])

        # Reshape back to the original shape
        shuffled_responses_ALL = reshaped_array.reshape(responses_ALL.shape)


        
        #Choose random number between 0 and 40
        
        for s in range(7): #stim types
            t = np.random.randint(0,41)
            # Split the data into a training set and a test set
            X_train, X_test, y_train, y_test = split_by_mouse(shuffled_responses_ALL,geno_list)

            X_train = np.reshape(X_train[:,s,:,:,t], (X_train.shape[0]*10,500))
            X_test = np.reshape(X_test[:,s,:,:,t], (X_test.shape[0]*10,500))

            # Repeat each value in y_train and y_test 10 times
            y_train = np.repeat(y_train,10)
            y_test = np.repeat(y_test,10)

                # Remove nans
            y_train = y_train[~np.isnan(X_train).any(axis=1)]
            X_train = X_train[~np.isnan(X_train).any(axis=1)]

            y_test = y_test[~np.isnan(X_test).any(axis=1)]
            X_test = X_test[~np.isnan(X_test).any(axis=1)]

            # Create an instance of LogisticRegression
            log_reg = LogisticRegression(max_iter=1000, penalty='l2',C = 0.1, multi_class='ovr')

            # Fit the model on the training data
            log_reg.fit(X_train, y_train)

            # Predict the labels for the test set
            y_pred = log_reg.predict(X_test)

            # Calculate the accuracy
            accuracy = accuracy_score(y_test, y_pred)

            accuracies[s,y] = accuracy
            #print(f'Accuracy: {accuracy}')
        if y%100 == 0:
            print('Iteration ' +str(y) + ' done.')
    return accuracies


def genotype_decoding(responses_all,geno_list, genotypes_to_decode,n_stims=7, n_bins=41):
    '''
    Performs decoding of the genotype of the mice based on the neural responses to the stimuli

    Parameters
    -----------------
    responses_all: array
        Array containing the responses of all the mice, shape (n_mice, n_stimuli, n_trials, n_neurons, n_frames)
    geno_list: list
        List containing the genotype of each mouse, shape (n_mice,)
    genotypes_to_decode: list
        List containing the genotypes to decode, shape (n_genotypes,)

    Returns
    -----------------
    accuracies: array
        Array containing the accuracy distribution, shape (n_stimuli, iters)

    '''
        
    accuracies = np.zeros([n_stims, n_bins])
    all_genotypes = ['WT','HET','KO','Control','SSRI']
    #Make a list with the genotypes that NOT to decode
    genotypes_not_to_decode = [x for x in all_genotypes if x not in genotypes_to_decode]


    for i in range(n_bins): #time bins
        for y in range(n_stims): #stim types
        
            # Split the data into a training set and a test set
            X_train, X_test, y_train, y_test = split_by_mouse(responses_all,geno_list)

            X_train = np.reshape(X_train[:,y,:,:,i], (X_train.shape[0]*10,500))
            X_test = np.reshape(X_test[:,y,:,:,i], (X_test.shape[0]*10,500))

            # Repeat each value in y_train and y_test 10 times
            y_train = np.repeat(y_train,10)
            y_test = np.repeat(y_test,10)

            # Remove nans
            y_train = y_train[~np.isnan(X_train).any(axis=1)]
            X_train = X_train[~np.isnan(X_train).any(axis=1)]

            y_test = y_test[~np.isnan(X_test).any(axis=1)]
            X_test = X_test[~np.isnan(X_test).any(axis=1)]

            #Select only genotypes to compare
            for geno in genotypes_not_to_decode:
                X_train = X_train[y_train != geno]
                y_train = y_train[y_train != geno]

                X_test = X_test[y_test != geno]
                y_test = y_test[y_test != geno]

            # Create an instance of LogisticRegression
            log_reg = LogisticRegression(max_iter=1000, penalty='l2',C = 0.1, multi_class='ovr')

            # Fit the model on the training data
            log_reg.fit(X_train, y_train)

            # Predict the labels for the test set
            y_pred = log_reg.predict(X_test)

            # Calculate the accuracy
            accuracy = accuracy_score(y_test, y_pred)

            accuracies[y,i] = accuracy
            #print(f'Accuracy: {accuracy}')

    return accuracies

def responsiveness(stim_resp):
    
    mean_responsive = np.zeros([7,13])
    mean_responsive[mean_responsive == 0] = np.nan
    for i in range(7):
        try:
            # Check  if stim_resp[i] is empty
            if np.all(np.isnan(stim_resp[i])):
                continue
            stim_respo = stim_resp[i]
            #remove in the second dimension values with all nans
            stim_respo = stim_respo[~np.isnan(stim_respo).all(axis=(1,2))]
            #Remove if all 0
            stim_respo = stim_respo[~np.all(stim_respo == 0, axis=(1,2))]
            rep = CaaN.MeanResponsivenessU(stim_respo[:,1:,:],PerTrial=True,pre_window_1 = 30, pre_window_2 = 45,post_window_1 = 75, post_window_2 = 90, bonferroning=True)

            mean_responsive[i,:10] = [np.mean(rep>0.1),np.mean(rep>0.2),np.mean(rep>0.3),np.mean(rep>0.4),np.mean(rep>0.5),np.mean(rep>0.6),np.mean(rep>0.7),np.mean(rep>0.8),np.mean(rep>0.9),np.mean(rep==1)]
                    
                        
            #Obtain the #Number responsive
            rep, sign = CaaN.MeanResponsivenessU(stim_respo[:,1:,:],PerTrial=False,pre_window_1 = 30, pre_window_2 = 45,post_window_1 = 75, post_window_2 = 90, bonferroning=False)
            #Append the number of responsive cells
            mean_responsive[i,10] = np.mean(rep<0.05)
            #Append the number of excited cells
            mean_responsive[i,11] = np.sum(sign[rep<0.05]==1)
            #Append the number of inhibited cells
            mean_responsive[i,12] = np.sum(sign[rep<0.05]==-1)
        except:
            continue
    return mean_responsive



def HISTO_plot_layers(df,parameter,genotypes,color,y_label='Number of cells',spacing = 1,fontsize=20,size=50,two_way_ANOVA = True, layer_stats=True,layers=[1,2,4,5,6]):
    '''
    Plots a vertical violin plot of the parameter in the different layers for histology across
    cortical columns,

    
    '''
    if two_way_ANOVA:
        df_tmp = df.copy()
        #REMOVE nones and nans in parameter
        df_tmp =df_tmp[df_tmp[parameter].notnull()]
        df_tmp = df_tmp.reset_index()
        #Rename column layer as 'age'
        df_tmp = df_tmp.rename(columns={'layer':'age'}) #The original function was written for development -.-"
        #RENAME 'Num Detecions as 'Num_detections', some python function don't like spaces
        df_tmp = df_tmp.rename(columns={parameter:parameter.replace(' ','_')})
        #Select only WT/HET/KO
        df_tmp = df_tmp[df_tmp['genotype'].isin(genotypes)]
        #Make wt 1, het 2, ko 3
        df_tmp['genotype'] = df_tmp['genotype'].replace(genotypes,np.arange(1,len(genotypes)+1))
        

        #Make age float instead of string
        df_tmp['age'] = df_tmp['age'].replace(['1','2','4','5','6'],[1,2,4,5,6])
        #Group by animal
        #set column parameter as float
        df_tmp[parameter.replace(' ','_')] = df_tmp[parameter.replace(' ','_')].astype(float)
        df_tmp = df_tmp.groupby(['animal','age']).mean()

        #Uncollapse age
        df_tmp = df_tmp.reset_index()
        #print(df_tmp)

        CaaN.stating(df_tmp,parameter.replace(' ','_'),genotypes=np.arange(1,len(genotypes)+1))
   
    top=False
    for i,layer in enumerate(layers):
        df_tmp = df.copy()
        df_tmp = df_tmp.reset_index()   
        df_tmp = df_tmp[df_tmp['layer']==str(layer)]
        #Remove nones
        df_tmp =df_tmp[df_tmp[parameter].notnull()]
        #remove nans
        df_tmp =df_tmp[df_tmp['genotype'].notnull()]
        
        if layer==2:
            top = True
        PdA.violin_plot(df_tmp,parameter,genotypes = genotypes,color = color,y_label=y_label,vert=False,plot_on_top=top,previous_pos=i*(len(genotypes)+spacing),fontsize=fontsize,size=size, layer_index= i)

        if layer_stats:
            #Select only WT/HET/KO
            df_tmp = df_tmp[df_tmp['genotype'].isin(genotypes)]
            #Make wt 1, het 2, ko 3
            df_tmp['genotype'] = df_tmp['genotype'].replace(genotypes,np.arange(1,len(genotypes)+1))
            #Group by animal
            #Set column parameter as float
            df_tmp[parameter] = df_tmp[parameter].astype(float)
            df_tmp = df_tmp.groupby(['animal']).mean()
            #print layer:
            print('layer: ' +str(layer))

            a,b = stats.shapiro(np.concatenate((df_tmp[df_tmp['genotype']==1][parameter],df_tmp[df_tmp['genotype']==2][parameter],df_tmp[df_tmp['genotype']==3][parameter])))
            print('Shapiro: stat: ' + str(a) + ', p-val:' + str(b))
        
            if b>0.05:
            #Tukey
                #import tukey pairwise
                from statsmodels.stats.multicomp import pairwise_tukeyhsd

                if len(genotypes)<3:
                    print(stats.ttest_ind(df_tmp[df_tmp['genotype']==1][parameter],df_tmp[df_tmp['genotype']==2][parameter]))
                
                else:
                    #print(pairwise_tukeyhsd(df_tmp[parameter],df_tmp['genotype']))
                    print(CaaN.fishers_lsd_from_df(df_tmp,parameter=parameter,group_col='genotype'))
            else:
                if len(genotypes)<3:
                    print(stats.mannwhitneyu(df_tmp[df_tmp['genotype']==1][parameter],df_tmp[df_tmp['genotype']==2][parameter]))
                else:
                    print(sp.posthoc_dunn(df_tmp, val_col=parameter, group_col='genotype'))
    #So that layer 1 is on top
    plt.gca().invert_yaxis()

            