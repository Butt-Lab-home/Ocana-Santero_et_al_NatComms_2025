
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import statsmodels
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sklearn.cluster as sc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import scipy.signal as sps
import utils.Plotting as Pl
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scikit_posthocs as sp




def extract_across_ages(df,parameter,min_age=7, max_age=30):
    '''
    Extracts the mean and standard deviation of a variable stored in a coloumn across ages
    
    Parameters
    ----------------
    df: pandas dataframe
    parameter: string, name of the coloumn to be extracted
    min_age: int, minimum age to be considered
    max_age: int, maximum age to be considered

    Returns
    ----------------
    data: numpy array, mean and standard deviation of the variable across ages
    
    '''
    data = np.zeros([2,max_age-min_age +1])
    data[:] = np.nan
    for a in range(min_age,max_age):
        df_tmp = df[df['age']==a]
        data[0,a-min_age] = df_tmp[parameter].mean()
        data[1,a-min_age] = df_tmp[parameter].std()
    return data

def sample_size_across_ages(df,min_age=7, max_age=30):
    '''
    Extracts the sample size across ages

    Parameters
    ----------------
    df: pandas dataframe
    min_age: int, minimum age to be considered
    max_age: int, maximum age to be considered

    Returns
    ----------------
    total_count: numpy array, sample size across ages

    '''

    total_count = np.zeros([24])
    total_count[:] = np.nan
    for a in range(min_age,max_age):
        total_count[a-7] = len(df[df['age']==a])
    return total_count

def extract_to_separate_coloumn(df,parameter,index,new_label):
    '''
    Extracts a sub-variable stored in a coloumn to a new coloumn
    
    Parameters
    ----------------
    df: pandas dataframe
    parameter: string, name of the coloumn to be extracted
    index: int, index of the sub-variable to be extracted
    new_label: string, name of the new coloumn
    
    Returns
    ----------------
    df: pandas dataframe, with the new coloumn
    
    '''
    df[new_label] = np.zeros(len(df))
    df[new_label] = np.nan

    for i, row in df.iterrows():
        try:
            df.at[i,new_label] = row[parameter][index]
        except:
            row[new_label] = np.nan
    return df

def developmental_plot(df,parameter,y_label,min_age=7, max_age=30, x_label ='Age (days)', genotype = None, color = ['k'], fontsize=30, legend_line_size=10, legend_font_size=30):
    '''
    Plots the mean and standard deviation of a variable across ages, with the option to separate by genotype

    Parameters
    ----------------
    df: pandas dataframe
    parameter: string, name of the coloumn to be extracted
    y_label: string, y axis label
    min_age: int, minimum age to be considered, default = 7
    max_age: int, maximum age to be considered, default = 30
    x_label: string, x axis label, default = 'Age (days)'
    genotype: list of strings, list of genotypes to be considered, default = None
    color: list of strings, list of colors to be used for each genotype, default = ['k']
    fontsize: int, fontsize of the y axis label, default = 30
    legend_line_size: int, line width of the legend, default = 10
    legend_font_size: int, fontsize of the legend, default = 30

    Returns
    ----------------
    Plot of the mean with the standard error shaded across ages
    
    '''
    plt.figure(figsize=(30,10))
    plt.xticks(np.arange(0,24,2),np.arange(7,30,2))
    plt.xlim([0,10])
    plt.xlabel('Age (days)',fontsize=30)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.ylabel(y_label,fontsize=fontsize)
    plt.xlabel(x_label,fontsize=fontsize)

    if genotype is None:
        data = extract_across_ages(df,parameter,min_age, max_age)
        n_ = sample_size_across_ages(df,min_age, max_age)
        sqrt_n = np.sqrt(n_)
        plt.plot(data[0,:],color='k')
        plt.fill_between(np.arange(0,24,1),data[0,:]-data[1,:]/sqrt_n,data[0,:]+data[1,:]/sqrt_n,alpha=0.5,color=color)
    else:
        count = 0
        n_count = np.zeros([24])
        n_count[:] = np.nan
        for g in genotype:
            data = extract_across_ages(df[df['genotype']==g],parameter,min_age, max_age)
            n_ = sample_size_across_ages(df[df['genotype']==g],min_age, max_age)
            sqrt_n = np.sqrt(n_)
            plt.plot(data[0,:],label = g, color= color[count])
            plt.fill_between(np.arange(0,24,1),data[0,:]-data[1,:]/sqrt_n,data[0,:]+data[1,:]/sqrt_n,alpha=0.5,color=color[count])
            count +=1
    leg = plt.legend(fontsize = legend_font_size)
    for legobj in leg.legendHandles: 
        legobj.set_linewidth(legend_line_size)
    #set xticks to be the ages
    return

def developmental_plot_errorbars(df,parameter,y_label,min_age=7, max_age=30, x_label ='Age (days)', genotype = None, color = 'k', fontsize=30, legend_line_size=10, legend_font_size=30,LW=8):

    '''
    Plots the mean and standard deviation of a variable across ages, with the option to separate by genotype

    Parameters
    ----------------
    df: pandas dataframe
    parameter: string, name of the coloumn to be extracted
    y_label: string, y axis label
    min_age: int, minimum age to be considered, default = 7
    max_age: int, maximum age to be considered, default = 30
    x_label: string, x axis label, default = 'Age (days)'
    genotype: list of strings, list of genotypes to be considered, default = None
    color: list of strings, list of colors to be used for each genotype, default = ['k']
    fontsize: int, fontsize of the y axis label, default = 30
    legend_line_size: int, line width of the legend, default = 10
    legend_font_size: int, fontsize of the legend, default = 30

    Returns
    ----------------
    Plot of the mean with the standard error shaded across ages
    
    '''



    plt.xticks(np.arange(0,24,2),np.arange(7,30,2))
    plt.xlim([0,10])
    plt.xlabel('Age (days)',fontsize=30)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.ylabel(y_label,fontsize=fontsize)
    plt.xlabel(x_label,fontsize=fontsize)
    ax.spines['bottom'].set_linewidth(8)
    ax.spines['left'].set_linewidth(8)
    #ticks length and width
    ax.tick_params(axis='both', which='major', length=15, width=8)

    if genotype is None:
        data = extract_across_ages(df,parameter,min_age, max_age)
        n_ = sample_size_across_ages(df,min_age, max_age)
        n_[n_==0] = np.nan
        sqrt_n = np.sqrt(n_)
        plt.plot(data[0,:],color='k',lw=LW)
        plt.errorbar(np.arange(0,24,1),data[0,:],yerr=data[1,:]/sqrt_n,fmt='o',color=color,lw=LW,markersize=4, capsize=8, capthick = 8)
    else:
        count = 0
        n_count = np.zeros([max_age-min_age+1])
        n_count[:] = np.nan
        for g in genotype:
            data = extract_across_ages(df[df['genotype']==g],parameter,min_age, max_age)
            n_ = sample_size_across_ages(df[df['genotype']==g],min_age, max_age)
            n_[n_==0] = np.nan
            sqrt_n = np.sqrt(n_)
            plt.plot(data[0,:],label = g, color= color[count],lw=LW)
            plt.errorbar(np.arange(0,data[1,:].shape[0],1),data[0,:],yerr=data[1,:]/sqrt_n,fmt='o',color=color[count],lw=LW,markersize=5, capsize=10, capthick=4)
            
            path = 'C:/Users/gabrielos/Downloads/'
            np.savetxt(path + str(count) + 'resp.txt', data[0,:], delimiter=",")
            count +=1
    leg = plt.legend(fontsize = legend_font_size)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(legend_line_size)




    
    return

def bar_scatter(df,parameter, genotypes, y_label, color = ['k','c','m'], fontsize=30, ANOVA = True, Tukey = True, Tukey_alpha = 0.05, percentage =  False) :
    '''
    Plots a bar plot with the points scattered on top, for a genotype (averaging across all ages present in the df),
    with the option to perform ANOVA and Tukey's HSD test
    
    Parameters
    ----------------
    df: pandas dataframe
    parameter: string, name of the coloumn to be extracted
    genotypes: list of strings, list of genotypes to be considered (e.g. ['WT','HET', 'KO'] or ['Control','SSRI'])]
    y_label: string, y axis label
    color: list of strings, list of colors to be used for each genotype, default = ['k','c','m']
    fontsize: int, fontsize of the y axis label, default = 30
    ANOVA: boolean, whether to perform a one way ANOVA, default = True
    Tukey: boolean, whether to perform Tukey's HSD test, default = True
    Tukey_alpha: float, alpha value for Tukey's HSD test, default = 0.05
    percentage: boolean, whether to plot the data as a percentage of the control, default = False

    Returns
    ----------------
    Bar plot with the points scattered on top

    '''


    #Define figure settings
    plt.figure(figsize=(30,10))
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.ylabel(y_label,fontsize=fontsize)
    geno_counter = 0
    #Iterate through the genotypes
    for g in genotypes:

        df_gen = df[df['genotype'] == g]
        animal_ids = df_gen['animal'].unique()

        #Iterate through animals with that genotype
        globals()[g] = np.zeros((len(animal_ids))) #Create an array to store the data with the same length as the number of animals with this genotype
        globals()[g][globals()[g]==0] = np.nan
        count = 0
        #Obtain the mean value of the parameter for each animal, across the range of ages provided
        for animal_id in animal_ids:
           globals()[g][count] = np.nanmean(df_gen[(df_gen['animal'] == animal_id)][parameter].dropna())
           count += 1

        #Plot the mean and standard error of the mean
        if percentage == True: #To plot the % value
            data = data*100

        #bar plot
        Pl.bar_scatter(g,globals()[g],y_label=y_label, bar_number=geno_counter,colour = color[geno_counter])
        geno_counter += 1

    if ANOVA == True:
        if len(genotypes) == 2:
            print('ANOVA: ' + str(scipy.stats.f_oneway(globals()[genotypes[0]][~np.isnan(globals()[genotypes[0]])],globals()[genotypes[1]][~np.isnan(globals()[genotypes[1]])])))
        if len(genotypes) == 3:
            print('ANOVA: ' + str(scipy.stats.f_oneway(globals()[genotypes[0]][~np.isnan(globals()[genotypes[0]])],globals()[genotypes[1]][~np.isnan(globals()[genotypes[1]])],globals()[genotypes[2]][~np.isnan(globals()[genotypes[2]])])))

    #Run Tukey test on genotypes

    
    if Tukey == True:
        if len(genotypes)==2: #For Control/SSRIs
            #Generate a list of the genotypes for each animal
            genotypes_rep = [genotypes[0]]*len(globals()[genotypes[0]][~np.isnan(globals()[genotypes[0]])]) + [genotypes[1]]*len(globals()[genotypes[1]][~np.isnan(globals()[genotypes[1]])]) 
            #Run Tukey test
            tukey = pairwise_tukeyhsd(endog=np.concatenate((globals()[genotypes[0]][~np.isnan(globals()[genotypes[0]])],globals()[genotypes[1]][~np.isnan(globals()[genotypes[1]])])),
                                    groups = genotypes_rep, alpha = Tukey_alpha)
            
        if len(genotypes)==3: #For WT/SERTKO-HET/SERTKO-HOM
            #Generate a list of the genotypes for each animal
            genotypes_rep = [genotypes[0]]*len(globals()[genotypes[0]][~np.isnan(globals()[genotypes[0]])]) + [genotypes[1]]*len(globals()[genotypes[1]][~np.isnan(globals()[genotypes[1]])]) + [genotypes[2]]*len(globals()[genotypes[2]][~np.isnan(globals()[genotypes[2]])])
            #Run Tukey test
            tukey = pairwise_tukeyhsd(endog=np.concatenate((globals()[genotypes[0]][~np.isnan(globals()[genotypes[0]])],globals()[genotypes[1]][~np.isnan(globals()[genotypes[1]])],globals()[genotypes[2]][~np.isnan(globals()[genotypes[2]])])),
                                        groups = genotypes_rep, alpha = Tukey_alpha)
            print(tukey)

    return

def violin_plot(df,parameter, genotypes, y_label = None, color = ['k','c','m'],title = None, fontsize = 70,var=0.03,size=100,vert=True,previous_pos=0,plot_on_top=False, layer_index= None):

    '''
    Plot the data as a violin plot

    Parameters
    ----------------
    df: dataframe, dataframe containing the data
    parameter: string, name of the column containing the data to be plotted
    genotypes: list of strings, list of the genotypes to be plotted (e.g. ['WT','HET','KO'])
    y_label: string, label of the y axis
    color: list of strings, list of the colors to be used for each genotype
    title: string, title of the plot
    fontsize: int, fontsize of the x and y axis labels
    var: float, variance of the jitter
    size: int, size of the scatter points
    
    Returns
    ----------------
    Violin plot

    '''

    #Define figure settings
    #plt.subplot(sp_row,sp_colomn,sp_number)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tick_params(axis='both', which='major', labelsize=fontsize,width=10,length=10)
    #plt.xlabel('Age (weeks)',fontsize=fontsize)

    if y_label == None:
        if vert:
            plt.ylabel(parameter,fontsize=fontsize)
        else:
            plt.xlabel(parameter,fontsize=fontsize)
    else:
        if vert:
            plt.ylabel(y_label,fontsize=fontsize)
        else:
            plt.xlabel(y_label,fontsize=fontsize)
        
    #plt.ylim(0,df[parameter].max())
    #plt.xticks(np.arange(0,x_lim+1,10))
    #plt.yticks(np.arange(0,df[parameter].max()+1,10))
    plt.title(title,fontsize=fontsize)
   

    counter = 0
    #Iterate through the genotypes
    for g in genotypes:

        df_gen = df[df['genotype'] == g]
        animal_ids = df_gen['animal'].unique()

        #Iterate through animals with that genotype
        globals()[g] = np.zeros((len(animal_ids))) #Create an array to store the
        animal_counter = 0
        for a in animal_ids:
            df_animal = df_gen[df_gen['animal'] == a]
            globals()[g][animal_counter] = df_animal[parameter].mean()
            animal_counter += 1
        

        #Plot the data with the appropiate color
        violin_parts = plt.violinplot(globals()[g],positions=[genotypes.index(g)+1+previous_pos],showmeans=True,showextrema=True,showmedians=False,widths=0.5,vert=vert)
        for i,violin_part in enumerate(violin_parts['bodies']):
            violin_part.set_facecolor(color[genotypes.index(g)])
            violin_part.set_edgecolor('black')
            violin_part.set_alpha(0.5)

        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
            vp = violin_parts[partname]
            if vp is not None:
                vp.set_edgecolor('black')
                vp.set_linewidth(1.2)  # Adjust the line width if needed
                #Alpha 0.5
        
        path = 'C:/Users/gabrielos/Downloads/'
        if layer_index != None:
            np.savetxt(path + str(layer_index) + '_' + str(counter) + 'data.txt', globals()[g], delimiter=",")
        else:
            np.savetxt(path + str(counter) + 'data.txt', globals()[g], delimiter=",")
        counter += 1
        
        if vert:
            plt.scatter(np.ones((len(animal_ids)))*(genotypes.index(g)+1)+previous_pos+np.random.normal(0,var,len(animal_ids)),globals()[g],color=color[genotypes.index(g)],s=size,edgecolors='k',zorder=3)
        else:
            plt.scatter(globals()[g],np.ones((len(animal_ids)))*(genotypes.index(g)+1)+previous_pos+np.random.normal(0,var,len(animal_ids)),color=color[genotypes.index(g)],s=size,edgecolors='k',zorder=3)    
        #X axis labels as genotype names
    if vert:
        plt.xticks(np.linspace(1+previous_pos,len(genotypes)+previous_pos,len(genotypes)),genotypes,fontsize=fontsize)
    else:
        if plot_on_top:
            # Get current x-ticks
            # Get current tick positions and labels
            current_tick_positions = plt.yticks()[0]
            current_tick_labels = [tick.get_text() for tick in plt.gca().get_yticklabels()]

            # Define new ticks to add
            new_ticks = np.linspace(1+previous_pos,len(genotypes)+previous_pos,len(genotypes))
            new_labels = genotypes

            # Combine current and new ticks
            all_tick_positions = np.concatenate((current_tick_positions, new_ticks))
            all_tick_labels = current_tick_labels + new_labels
            
            # Set the combined x-ticks
            plt.yticks(all_tick_positions,all_tick_labels,fontsize=fontsize)
        else:
            plt.yticks(np.linspace(1+previous_pos,len(genotypes)+previous_pos,len(genotypes)),genotypes,fontsize=fontsize)
    #set color of the first violin plot to black, the second to cyan and the third to magenta


    #Set axis width
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(10)

    return()

def cum_sum(df,parameter, genotypes, ages, color = ['k','c','m'],title = None, sp_row=1,sp_colomn=4,sp_number=1, fontsize = 70, cumsum_range = np.linspace(-2,70,199), vertical_line=False, vertical_line_val = 2, legend_line_size = 35, legend_font_size = 60, x_lim = 70):
    '''
    Plot the cumulative sum of the data for each genotype

    Parameters
    ----------------
    df: dataframe, dataframe containing the data
    parameter: string, name of the column containing the cumulative sum data to be plotted
    genotypes: list of strings, list of the genotypes to be plotted (e.g. ['WT','HET','KO'])
    ages: list of integers, list of the ages to be included in the plot (e.g. [7,8,9,10)
    color: list of strings, list of the colors to be used for each genotype
    title: string, title of the plot
    sp_row: integer, number of rows in the subplot
    sp_colomn: integer, number of colomns in the subplot
    sp_number: integer, number of the subplot
    fontsize: integer, fontsize of the axis labels
    cumsum_range: array, range of values to be included in the cumulative sum plot
    vertical_line: boolean, whether to plot a vertical line at a specific value
    vertical_line_val: integer, value at which to plot the vertical line
    legend_line_size: integer, size of the line in the legend
    legend_font_size: integer, size of the font in the legend
    x_lim: integer, maximum value of the x axis

    Returns
    ----------------
    Plot of the cumulative sum of the data for each genotype

    '''

    ax = plt.subplot(sp_row,sp_colomn,sp_number)

    plt.rc('xtick',labelsize=50)
    plt.rc('ytick',labelsize=50)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # change all spines
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(9)
    ax.tick_params(width=13,length=30)
    plt.ylabel('Cummulative\nprobability',fontsize=70)
    plt.xlabel('ΔF/F',fontsize=fontsize)
    if title != None:
        plt.title(title,fontsize=fontsize+10)

    #Make a new dataframe with only the data for the ages of interest

    df_data = df[df['age'] == ages[0]]
    for age in ages[1:]:
        df_data = df_data.append(df[df['age'] == age])

    #Obtain the number of animals in each genotype
    geno_counter = 0
    for g in genotypes:
        df_gen = df_data[df_data['genotype'] == g]
        animal_ids = df_gen['animal'].unique()

        globals()[g] = np.zeros((len(animal_ids),199)) #Create an array to store the data with the same length as the number of animals with this genotype
        globals()[g][globals()[g]==0] = np.nan

        count = 0

        for animal_id in animal_ids:
            #boolo = ((df_data_4[(df_data_4['animal'] == animal_id)]['Events_features']).reset_index(drop=True))[0][0][:,0]>0.80
            globals()[g][count] = np.array((df_gen[(df_gen['animal'] == animal_id)][parameter]).mean(axis=0))
            count += 1

        n_sqrt = np.sqrt(len(animal_ids))

        plt.plot(cumsum_range,np.nanmean(globals()[g],axis=0)/np.nanmean(globals()[g],axis=0)[-1], lw =4 ,label = g, color = color[geno_counter])
        plt.fill_between(cumsum_range,np.nanmean(globals()[g],axis=0)/np.nanmean(globals()[g],axis=0)[-1]-(np.nanstd(globals()[g],axis=0)/n_sqrt)/np.nanmean(globals()[g],axis=0)[-1],np.nanmean(globals()[g],axis=0)/np.nanmean(globals()[g],axis=0)[-1]+(np.nanstd(globals()[g],axis=0)/n_sqrt)/np.nanmean(globals()[g],axis=0)[-1],alpha=0.8, color = color[geno_counter], label = '_nolegend_')
        geno_counter += 1


    #plot vertical threshold line
    if vertical_line == True:
        plt.axvline(x=vertical_line_val, color='k', linestyle='--',linewidth=5)

    plt.xlim([-2,x_lim])

    leg = plt.legend(fontsize = legend_font_size)
    for legobj in leg.legendHandles: 
        legobj.set_linewidth(legend_line_size)
    plt.tight_layout()


def two_way_ANOVA(df,parameter,var_1 = 'age',var_2 = 'genotype'):
    '''
    Perform a two way ANOVA on a parameter of interest in the dataframe 

    Parameters
    ----------------
    df: dataframe, dataframe containing the data
    parameter: string, name of the column containing the data to be analyzed
    var_1: string, name of the column containing the first variable to be analyzed, default = 'age'
    var_2: string, name of the column containing the second variable to be analyzed, default = 'genotype'

    Returns
    ----------------
    F value and p value for the two way ANOVA
    '''

    #Make a copy of the dataframe including only the coloumns 'genotype','age' and 'AUC'
    df_tmp = df[[var_1,var_2,parameter]].copy()
    #remove rows with nan values
    df_tmp = df_tmp.dropna()
    #convert parameter to numeric
    

    df_tmp[parameter] = pd.to_numeric(df_tmp[parameter])
    #two way ANOVA
    model = ols(parameter + ' ~ C(' + var_1 + ') + C(' + var_2 + ') + C(' + var_1 + '):C(' + var_2 + ')', data=df_tmp).fit()
    return sm.stats.anova_lm(model, typ=2)



def violin_plot_several_ages(df,parameter, genotypes, ages,y_label = None, color = ['k','c','m'],title = None, fontsize = 70,vert=True):

    '''
    Plot the data as a violin plot

    Parameters
    ----------------
    df: dataframe, dataframe containing the data
    parameter: string, name of the column containing the data to be plotted
    genotypes: list of strings, list of the genotypes to be plotted (e.g. ['WT','HET','KO'])
    y_label: string, label of the y axis
    color: list of strings, list of the colors to be used for each genotype
    title: string, title of the plot
    fontsize: int, fontsize of the x and y axis labels
    
    Returns
    ----------------
    Violin plot

    '''

    #Define figure settings
    #plt.subplot(sp_row,sp_colomn,sp_number)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tick_params(axis='both', which='major', labelsize=fontsize,width=10,length=10)
    #plt.xlabel('Age (weeks)',fontsize=fontsize)
    if y_label == None:
        plt.ylabel(parameter,fontsize=fontsize)
    else:
        plt.ylabel(y_label,fontsize=fontsize)
    #plt.ylim(0,df[parameter].max())
    #plt.xticks(np.arange(0,x_lim+1,10))
    #plt.yticks(np.arange(0,df[parameter].max()+1,10))
    plt.title(title,fontsize=fontsize)

    #Iterate through the genotypes
    for age in ages:
        df_gen = df[df['age']==age]
        for g in genotypes:
            df_gen = df_gen[df_gen['genotype'] == g]
            animal_ids = df_gen['animal'].unique()
            #Iterate through animals with that genotype
            globals()[g] = np.zeros((len(animal_ids))) #Create an array to store the
            animal_counter = 0
            for a in animal_ids:
                df_animal = df_gen[df_gen['animal'] == a]
                globals()[g][animal_counter] = df_animal[parameter].mean()
                animal_counter += 1

            #Plot the data
            plt.violinplot(globals()[g],positions=[ages.index(age)+genotypes.index(g)+1],showmeans=True,showextrema=True,showmedians=False,widths=0.5,vert=vert)
            if vert:
                plt.scatter(np.ones((len(animal_ids)))*(ages.index(age)+genotypes.index(g)+1),globals()[g],color=color[genotypes.index(g)],s=100,edgecolors='k',zorder=3)
            else:
                plt.scatter(globals()[g],np.ones((len(animal_ids)))*(ages.index(age)+genotypes.index(g)+1),color=color[genotypes.index(g)],s=100,edgecolors='k',zorder=3)
            #X axis labels as genotype names

    if vert:
        plt.xticks(np.linspace(1,len(genotypes),len(genotypes)),genotypes,fontsize=fontsize)
    else:
        plt.yticks(np.linspace(1,len(genotypes),len(genotypes)),genotypes,fontsize=fontsize)

    #Set axis width
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(10)

    return()


def Response(df,parameter, genotypes, ages = None, length=660, y_label = 'Response',color = ['k','c','m'],title = None, sp_row=1,sp_colomn=4,sp_number=1, fontsize = 70, cumsum_range = np.linspace(-60,600,660), vertical_line=False, vertical_line_val = 2, legend_line_size = 10, legend_font_size = 50, x_lim = 70,stat=np.nanmean,num_cells=False,cells=None):
    '''
    Plot the cumulative sum of the data for each genotype

    Parameters
    ----------------
    df: dataframe, dataframe containing the data
    parameter: string, name of the column containing the cumulative sum data to be plotted
    genotypes: list of strings, list of the genotypes to be plotted (e.g. ['WT','HET','KO'])
    ages: list of integers, list of the ages to be included in the plot (e.g. [7,8,9,10)
    color: list of strings, list of the colors to be used for each genotype
    title: string, title of the plot
    sp_row: integer, number of rows in the subplot
    sp_colomn: integer, number of colomns in the subplot
    sp_number: integer, number of the subplot
    fontsize: integer, fontsize of the axis labels
    cumsum_range: array, range of values to be included in the cumulative sum plot
    vertical_line: boolean, whether to plot a vertical line at a specific value
    vertical_line_val: integer, value at which to plot the vertical line
    legend_line_size: integer, size of the line in the legend
    legend_font_size: integer, size of the font in the legend
    x_lim: integer, maximum value of the x axis

    Returns
    ----------------
    Plot of the cumulative sum of the data for each genotype

    '''

    ax = plt.subplot(sp_row,sp_colomn,sp_number)

    plt.rc('xtick',labelsize=50)
    plt.rc('ytick',labelsize=50)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # change all spines
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(9)
    ax.tick_params(width=13,length=30)
    plt.ylabel(y_label,fontsize=70)
    plt.xlabel('ΔF/F',fontsize=fontsize)
    if title != None:
        plt.title(title,fontsize=fontsize+10)

    #Make a new dataframe with only the data for the ages of interest
    if ages != None:
        df_data = df[df['age'] == ages[0]]
        for age in ages[1:]:
            df_data = df_data.append(df[df['age'] == age])
    else:
        df_data = df

    #Obtain the number of animals in each genotype
    geno_counter = 0
    for g in genotypes:
        df_gen = df_data[df_data['genotype'] == g]
        animal_ids = df_gen['animal'].unique()

        globals()[g] = np.zeros((len(animal_ids),length)) #Create an array to store the data with the same length as the number of animals with this genotype
        globals()[g][globals()[g]==0] = np.nan

        count = 0

        for animal_id in animal_ids:
            #boolo = ((df_data_4[(df_data_4['animal'] == animal_id)]['Events_features']).reset_index(drop=True))[0][0][:,0]>0.80
            globals()[g][count] = np.array((df_gen[(df_gen['animal'] == animal_id)][parameter]).mean(axis=0))
            count += 1

        n_sqrt = np.sqrt(len(animal_ids))
        #NORMALIZE with num cells
        if num_cells:
            n_sqrt = cells[genotypes.index(g)]
        baseline = np.nanmean(globals()[g][:,:60],axis=1)
        globals()[g] = (globals()[g].T-baseline).T
        plt.plot(cumsum_range,np.nanmean(globals()[g],axis=0), label = g, color = color[geno_counter],lw=5)
        plt.fill_between(cumsum_range,np.nanmean(globals()[g],axis=0)-(np.nanstd(globals()[g],axis=0)/n_sqrt),np.nanmean(globals()[g],axis=0)+(np.nanstd(globals()[g],axis=0)/n_sqrt),alpha=0.5, color = color[geno_counter])
        path = 'C:/Users/gabrielos/Downloads/'
        np.savetxt(path + str(geno_counter) + 'respON.txt', np.nanmean(globals()[g],axis=0), delimiter=",")
            
        geno_counter += 1

    #Run tukey on mean response [:60]
    #Calculate mean
    for g in genotypes:
        globals()[g] = stat(globals()[g][:,60:],axis=1)
        #remove nans
        globals()[g] = globals()[g][~np.isnan(globals()[g])]
    #Stats

    
    if len(genotypes) == 2:
        concatenate = np.concatenate((globals()[genotypes[0]],globals()[genotypes[1]]))
        a,b = scipy.stats.shapiro(concatenate)
        if b < 0.05:
            print('Mann Whitney U test: ' + str(scipy.stats.mannwhitneyu(globals()[genotypes[0]],globals()[genotypes[1]])))
        else:
            print('T-test: ' + str(scipy.stats.ttest_ind(globals()[genotypes[0]],globals()[genotypes[1]])))
    #Run ANOVA and Tukey
    
    if len(genotypes) == 3:
        concatenate = np.concatenate((globals()[genotypes[0]],globals()[genotypes[1]],globals()[genotypes[2]])) 
        a,b = scipy.stats.shapiro(concatenate)
        if b < 0.05:
            print('Kruskal Wallis test: ' + str(scipy.stats.kruskal(globals()[genotypes[0]],globals()[genotypes[1]],globals()[genotypes[2]])))
            print('Tukey: ' + str(pairwise_tukeyhsd(endog=np.concatenate((globals()[genotypes[0]],globals()[genotypes[1]],globals()[genotypes[2]])),
                                        groups = np.concatenate((np.ones((len(globals()[genotypes[0]]))),np.ones((len(globals()[genotypes[1]])))*2,np.ones((len(globals()[genotypes[2]])))*3)), alpha = 0.05)))

        else:
            #Anova
            print('ANOVA: ' + str(scipy.stats.f_oneway(globals()[genotypes[0]],globals()[genotypes[1]],globals()[genotypes[2]])))
            #Tukey
            print('Tukey: ' + str(pairwise_tukeyhsd(endog=np.concatenate((globals()[genotypes[0]],globals()[genotypes[1]],globals()[genotypes[2]])),
                                        groups = np.concatenate((np.ones((len(globals()[genotypes[0]]))),np.ones((len(globals()[genotypes[1]])))*2,np.ones((len(globals()[genotypes[2]])))*3)), alpha = 0.05)))
            
        


    #plot vertical threshold line
    if vertical_line == True:
        plt.axvline(x=vertical_line_val, color='k', linestyle='--',linewidth=5)

    #plt.xlim([-2,x_lim])

    leg = plt.legend(fontsize = legend_font_size)
    for legobj in leg.legendHandles: 
        legobj.set_linewidth(legend_line_size)
    plt.tight_layout()


def Corr_distance_plot(df,parameter, ages, genotypes, color = ['k','c','m'],title = None, sp_row=1,sp_colomn=4,sp_number=1, fontsize = 70, cumsum_range = np.arange(0,400,41), vertical_line=False, vertical_line_val = 2, legend_line_size = 10, legend_font_size = 50, x_lim = 70,LW=10):
    '''
    Plot the cumulative sum of the data for each genotype

    Parameters
    ----------------
    df: dataframe, dataframe containing the data
    parameter: string, name of the column containing the cumulative sum data to be plotted
    genotypes: list of strings, list of the genotypes to be plotted (e.g. ['WT','HET','KO'])
    ages: list of integers, list of the ages to be included in the plot (e.g. [7,8,9,10)
    color: list of strings, list of the colors to be used for each genotype
    title: string, title of the plot
    sp_row: integer, number of rows in the subplot
    sp_colomn: integer, number of colomns in the subplot
    sp_number: integer, number of the subplot
    fontsize: integer, fontsize of the axis labels
    cumsum_range: array, range of values to be included in the cumulative sum plot
    vertical_line: boolean, whether to plot a vertical line at a specific value
    vertical_line_val: integer, value at which to plot the vertical line
    legend_line_size: integer, size of the line in the legend
    legend_font_size: integer, size of the font in the legend
    x_lim: integer, maximum value of the x axis

    Returns
    ----------------
    Plot of the cumulative sum of the data for each genotype

    '''

    ax = plt.subplot(sp_row,sp_colomn,sp_number)

    plt.rc('xtick',labelsize=50)
    plt.rc('ytick',labelsize=50)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # change all spines
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(9)
    ax.tick_params(width=13,length=30)
    plt.ylabel('Pearson correlation coefficient',fontsize=70)
    plt.xlabel('Distance (µm)',fontsize=fontsize)
    if title != None:
        plt.title(title,fontsize=fontsize+10)

    #Make a new dataframe with only the data for the ages of interest
 
    df_data = df[df['age'] == ages[0]]
    for age in ages[1:]:
        df_data = df_data.append(df[df['age'] == age])

    #Obtain the number of animals in each genotype
    geno_counter = 0
    for g in genotypes:
        df_gen = df_data[df_data['genotype'] == g]
        animal_ids = df_gen['animal'].unique()

        globals()[g] = np.zeros((len(animal_ids),10)) #Create an array to store the data with the same length as the number of animals with this genotype
        globals()[g][globals()[g]==0] = np.nan

        count = 0

        for animal_id in animal_ids:
            #boolo = ((df_data_4[(df_data_4['animal'] == animal_id)]['Events_features']).reset_index(drop=True))[0][0][:,0]>0.80
            globals()[g][count] = np.array((df_gen[(df_gen['animal'] == animal_id)][parameter]).mean(axis=0))
            count += 1

        n_sqrt = np.sqrt(len(animal_ids))
        baseline = 0
        #plor error bars
        plt.errorbar(np.arange(0,10,1),np.nanmean(globals()[g],axis=0), yerr=(np.nanstd(globals()[g],axis=0)/n_sqrt), label = g, color = color[geno_counter],lw=LW, capsize=10, capthick=5)
        path = 'C:/Users/gabrielos/Downloads/'
        np.savetxt(path + str(geno_counter) + 'corr.txt', np.nanstd(globals()[g],axis=0)/n_sqrt, delimiter=",")

        geno_counter += 1


    #plot vertical threshold line
    if vertical_line == True:
        plt.axvline(x=vertical_line_val, color='k', linestyle='--',linewidth=5)

    #plt.xlim([-2,x_lim])

    leg = plt.legend(fontsize = legend_font_size)
    for legobj in leg.legendHandles: 
        legobj.set_linewidth(legend_line_size)
    plt.tight_layout()

    return None


def permutation_test(df, parameter,factor1 = 'age', factor2 = 'genotype', num_permutations=1000, test_statistic_func = two_way_ANOVA):
    '''
    Perform a permutation test for a two-factor comparison.

    Parameters:
    - data: A pandas DataFrame or NumPy array containing the data for the two factors.
    - factor1: The first factor or group labels (categorical variable).
    - factor2: The second factor or group labels (categorical variable).
    - num_permutations: Number of permutations to perform.
    - test_statistic_func: A function to compute the test statistic (default: two_way_ANOVA)

    Returns:
    - p_values: The p-values for the permutation test.
    '''

    #remove rows with nan or nones
    #Make a copy of the dataframe including only the coloumns 'genotype','age' and 'AUC'
    df_tmp = df[[factor1,factor2,parameter]].copy()
    #remove rows with nan values
    df_tmp = df_tmp.dropna()

    # Calculate the original test statistic
    original_statistic = test_statistic_func(df_tmp,parameter = parameter,var_1 = factor1,var_2 =factor2)['PR(>F)'].iloc[0:3]

    # Create an array to store permuted test statistics
    permuted_statistics = np.zeros([3,num_permutations]) 
    #create new dataframe with the same shape but all nones
    
    for i in range(num_permutations):
        # Make a new data frame with values from factor1 and factor2 randomly permuted
        df_tmp[factor1] = np.random.permutation(df_tmp[factor1])
        df_tmp[factor2] = np.random.permutation(df_tmp[factor2])

        # Calculate the permuted test statistic
        permuted_statistics[0,i] = test_statistic_func(df_tmp,parameter,factor1,factor2)['PR(>F)'].iloc[0]
        permuted_statistics[1,i] = test_statistic_func(df_tmp,parameter,factor1,factor2)['PR(>F)'].iloc[1]
        permuted_statistics[2,i] = test_statistic_func(df_tmp,parameter,factor1,factor2)['PR(>F)'].iloc[2]


    # Calculate the p-value
    p_value_age = (np.sum(permuted_statistics[0,:] <= original_statistic[0])) / (num_permutations)
    p_value_genotype = (np.sum(permuted_statistics[1,:] <= original_statistic[1])) / (num_permutations)
    p_value_interaction = (np.sum(permuted_statistics[2,:] <= original_statistic[2])) / (num_permutations)


    return p_value_age, p_value_genotype, p_value_interaction