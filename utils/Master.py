
#Master functions to iterate thorugh folders and create Data Frames

import numpy as np
import pandas as pd
import os
import utils.CaAnaly as CaaN
import gspread



def next_dates(date,n=11,include_first = False,format_ = False):
    '''
    Function that generates a series of dates starting from a given date and returns them as a list of strings.
    
    Parameters:
    ---------------------
    date: str with format YYYY_MM_DD
    n: number of dates to generate (default = 11)
    include_first: if True, the first date is included in the list (default = False)
    format_: if True, the date is returned as a string with format YYYY_MM_DD, instead of YYYY-MM-DD (default = True)

    Returns:
    ---------------------
    dates: list of strings with format YYYY-MM-DD or YYYY_MM_DD

    '''
    date = date.replace('_', '-')
    date = np.datetime64(date)
    dates = np.arange(date, date + np.timedelta64(n, 'D'), np.timedelta64(1, 'D'))
    dates = np.array([str(i) for i in dates])
    if format_ == True:
        dates = [i.replace('-','_') for i in dates]
    else:
        dates = list(dates)
    if include_first == False:
        return dates[1:]
    if include_first == True:
        return dates
    
def create_dictionaries(Google_doc_name = 'Gabi_Mice_Record', Google_sheet_name = 'DPhil-Surgery'):
    '''
    Creates dictionaries with the information from the surgery sheet in the google doc. 
    The keys are the litter number and the values are the dates, animals, age and genotype of the litter.
    The litter number is the order in which the litters were recorded in the surgery sheet.
    The dates are the dates of the second surgery (imaging is just +1 so next_dates doesn't include the first one)
    and the age is the age of the animal at the second surgery + 1. T

    Parameters
    --------------------
    Google_doc_name: name of the google doc where the surgery sheet is stored (default = 'Gabi_Mice_Record')
    Google_sheet_name: name of the surgery sheet (default = 'DPhil-Surgery')

    Returns
    --------------------
    dict_dates: dictionary where the keys are the litter number and the values are the dates of the second surgery
    dict_animals: dictionary where the keys are the litter number and the values are the animal IDs
    dict_start_age: dictionary where the keys are the litter number and the values are the age of the animal at the second surgery + 1 (first day imaged)
    dict_genotype: dictionary where the keys are the litter number and the values are the genotype of the litter

    '''
    #Initialize dictionaries
    dict_dates = {}
    dict_animals = {}
    dict_start_age = {}
    dict_genotype = {}
    dict_interneuron_genotype = {}
    dict_sex = {}

    #Load you surgery spreadsheet
    gc = gspread.oauth()
    Mice = gc.open(Google_doc_name)
    surgery = Mice.worksheet(Google_sheet_name)

    #Extract information from surgery sheet (gspread) and put it in a list, first find the right column then extract it

    #Extract animal
    Animals = surgery.col_values(surgery.row_values(3).index('Pseudo ID')+1) #+1 because for gspread the first column is 1 and not 0 like python 

    # Extract dates of second surgery (imaging is just +1)
    date_second_surgery = [i for i, e in enumerate(surgery.row_values(3)) if e == 'Date'][1] # First procedure is always the injection and you want the window (-1 imaging)
    Dates = surgery.col_values(date_second_surgery+1) 

    # Extract genotype 
    Genotype = surgery.col_values(surgery.row_values(3).index('Genotype')+1) # Genotype column
    Interneuron_genotype = surgery.col_values(surgery.row_values(3).index('Interneuron Genotype')+1)

    # Extract sex
    Sex =  surgery.col_values(surgery.row_values(3).index('Sex')+1)

    # Extract age
    Age_second_surgery = [i for i, e in enumerate(surgery.row_values(3)) if e == 'Age'][1] 
    Age = surgery.col_values(Age_second_surgery+1) 
    Age = [int(i[0]) if i is not '' else 0 for i in Age[3:]] #Extract age as int and replace empty cells with 0
    #Add 1 to the age to get the age at the first day imaged
    Age = [i+1 for i in Age]
    Age = [0,0,'Age'] + Age # Add the 2 empty cells and the header

    #Extract if imaged
    Imaged = surgery.col_values(surgery.row_values(3).index('Imaged')+1) # Imaged column
    
    #Extract imaged dates and animals
    Dates_imaged=[e for i, e in enumerate(Dates) if i in [i for i, e in enumerate(Imaged) if e == 'yes']]
    #List unique values on Dates_imaged (since it is the same across litters)
    Dates_imaged_unique = list(dict.fromkeys(Dates_imaged))
    #Animals imaged
    Animals_imaged=[e for i, e in enumerate(Animals) if i in [i for i, e in enumerate(Imaged) if e == 'yes']]

    #Create dictionaries
    for i in range(len(Dates_imaged_unique)):
        dict_dates[i] =  next_dates(Dates_imaged_unique[i],n=23) #23 days will obtain dates until P30, to cover the P20 day imaged, but will generate lots of non-existing data, only worth it if curated after
        dict_animals[i] = [e for y, e in enumerate(Animals_imaged) if y in [y for y, e in enumerate(Dates_imaged) if e == Dates_imaged_unique[i]]]
        dict_start_age[i] = Age[Animals.index(dict_animals[i][0])]
        dict_genotype[i] = [e for y, e in enumerate(Genotype) if y in [y for y, e in enumerate(Animals) if e in dict_animals[i]]]
        dict_interneuron_genotype[i] = [e for y, e in enumerate(Interneuron_genotype) if y in [y for y, e in enumerate(Animals) if e in dict_animals[i]]]
        dict_sex[i] = [e for y, e in enumerate(Sex) if y in [y for y, e in enumerate(Animals) if e in dict_animals[i]]]

    return dict_dates, dict_animals, dict_start_age, dict_genotype, dict_interneuron_genotype, dict_sex





def create_data_frame(dict_animals,dict_dates,dict_start_age, dict_genotype, dict_interneuron_genotype, dict_sex, Path = '/home/gsantero/mnt/qnap/Data/'):
    '''
    Produces a data frame with experiment set, date, animal ids, animal age and data paths, provided this data
    in three dictionaries, made with help from all powerful This! (JAN 2023)

    Parameters
    --------------------
    dict_animals: dictionary where the keys are the experiment (i.e. litter) the values the animal IDs
    dict_dates: dictionary where the keys are the experiment (i.e. litter) the values the dates recorded
    dict_start_dates: dictionary where the keys are the experiment (i.e. litter) the values the starting imaging date for that litter
    dict_genotype: dictionary where the keys are the experiment (i.e. litter) the values the genotype of the litter
    dict_interneuron_genotype: dictionary where the keys are the experiment (i.e. litter) the values the interneuron labelled
    dict_sex: dictionary where the keys are the experiment (i.e. litter) and the values the sex of the animals
    Path: path to the data (default = '/home/gsantero/mnt/qnap/Data/')

    Returns
    --------------------
    df_data: pandas data frame with this data
    
    '''
    for i_ds in dict_animals.keys():
            ## Load the data:
            list_animals = dict_animals[i_ds]
            list_dates = dict_dates[i_ds]
            list_dates_ = [a.replace('-','_') for a in list_dates] #To solve your stupid '_' mistake
            list_genotypes = dict_genotype[i_ds]
            list_interneuron_genotypes = dict_interneuron_genotype[i_ds]
            list_sex = dict_sex[i_ds]
            
            ## Now create lists of repeated animals & dates. Different orders, see comments:
            list_repeated_animals = np.ravel([[a] * len(list_dates) for a in list_animals])  # a1, a1, a1, a2, a2, a2, a3, a3, a3 etc.
            list_repeated_genotypes = np.ravel([[b] * len(list_dates) for b in list_genotypes])  # g1, g1, g1, g2, g2, g2, g3, g3, g3 etc.
            list_repeated_interneuron_genotypes = np.ravel([[c] * len(list_dates) for c in list_interneuron_genotypes])
            list_repeated_sex = np.ravel([[d] * len(list_dates) for d in list_sex])   
            list_repeated_dates = list_dates * len(list_animals)  # d1, d2, d3, d1, d2, d3, d1, d2, d3 etc.
            list_repeated_ages = [d for d in range(dict_start_age[i_ds], dict_start_age[i_ds] + (len(list_dates)))] * len(list_animals)  # same sequence as dates
            assert len(list_repeated_dates) == len(list_repeated_animals)  # these are the same length now (both n_animals * n_dates)

           
            ## Create temporary DF for this experiment:
            tmp_df = pd.DataFrame({'animal': list_repeated_animals,'date': list_repeated_dates, 'age': list_repeated_ages,'genotype':list_repeated_genotypes,'interneuron':list_repeated_interneuron_genotypes, 'sex': list_repeated_sex})  # put in dataframe # 
            tmp_df['experiment_id'] = i_ds  # add id of experiment set by setting same value for all rows

            ## Add file paths using lambda method (to combine columns of str into a new column)
            tmp_df['filepath'] = tmp_df.apply(lambda df: Path + df['date'] + '/' + df['date'] + '_' + df['animal'] + '_' + 't-001' + '/suite2p/plane0/', axis=1)
                        
            ## Add temporary DF to main DF:
            if i_ds == 0:  # first iteration, create df:
                df_data = tmp_df
            else:  # next iterations, add df 
                df_data = pd.concat((df_data, tmp_df))

    return df_data

        
def curate(Data_Frame):
    '''
    Curates your data frame by removing the rows with non existing paths

    Parameters
    --------------------
    Data_Frame: data frame with one recording day per row

    Returns
    --------------------
    df_data: pandas data frame with curated data

    '''
    List = []
    for i, row in Data_Frame.iterrows():
        if (os.path.exists(row['filepath'])): # Save the index on the list if the path does not exist
            continue
        else:
            List.append(i)
            if os.path.exists(row['filepath'][:-15]):
                print(row['filepath'][-39:-16] +': Exists but there is no suite2p folder')
    df_data = Data_Frame.drop(List) # Drop the rows with the index in the list
    return df_data,List
        
    
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
        if os.path.exists(row['filepath']) and os.path.isfile(row['filepath']+'F.npy') and os.path.isfile(row['filepath']+'iscell.npy'):
            data = function(row['filepath'])
            df_data.at[i,new_data_label] = data

        else:
            print(f'Data from animal {row["animal"]} on {row["date"]} not found.')

    return df_data