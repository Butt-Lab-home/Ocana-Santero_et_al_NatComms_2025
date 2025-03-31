
import numpy as np
import matplotlib.pyplot as plt
import utils.CaAnaly as CaaN
import sklearn.cluster as sc
import matplotlib.gridspec as gridspec


def jitter(arr, stdev = 0.12):
    '''
    Jitters values with a specific stdev for plotting scatters
    on top of var plots

    Parameters
    -------------
        arr = values to transform in jitter, generally an array with the same integer repeated several times (e.g., [1,1,1,1])
        stdev = value of the stdev to jitter the new values
    
    Returns
    -------------
        arr = initial arr with each value randomly shifted/jittered
    '''
    return arr + np.random.randn(len(arr)) * stdev


def plot_1D(y, ax=None, colour='k',LW=5, x_axis = 'Seconds', fr=30,num_lab=10,
            ylabel='ΔF/F',min_age = 7, subplot_x=1,subplot_y=1,subplot_num=1, 
            fsize = (30,10), errorbar = False, two_d = False, x = any):
    '''
     Plots with some default parameters that you generally use.

    Parameters
    -------------
        y: y_Data
        colour: color of the trace, by default black 
        LW: thickness of the trace, by default 5
        x_axis: To establish ticks and x label, by default 'Seconds', but it can also be 'Minutes' and 'Days. Careful because you plot integers and
                so you might round wrongly if the total number of frames is not a multiple of 30 (s) ot 180 (minutes)
        fr: frame rate, by default 30
        num_lab: number of x_labels, by default 10
        ylabel: y-axis label, by default is ΔF/F
        fsize: determines figure size, by default (30,10).
        errorbar: to plot errorbars, by default 'False', if true you need to provide a two dimensional 'y' and it will plot the mean of the rows and plot
                  the SEM as errorbars in each data point.
        two_d: Plots a graph with 'x' values provided. Default 'False', if True you need to provide the x values.    
                
    Returns
    -------------
        figure with plt.plot(y)
            if errobar = True: plt.errorbar(x,y,yerr)
            if two_d == True: plt.plot(x,y)
    '''
    if ax is None:
        if subplot_num==1:
            plt.figure(figsize=fsize)
    
    ax = plt.subplot(subplot_x,subplot_y,subplot_num)
    plt.rc('xtick',labelsize=50)
    plt.rc('ytick',labelsize=50)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(width=7,length=15)
    size= y.shape[0]
    if errorbar==False:
        plt.xlim([0,size])

# change all spines   REPLACE PLT FOR AX HANDLE!!!! You can also return the AX to get the handle directly
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(9)
    if x_axis == 'Frames':
        plt.xticks(np.linspace(0,size,num_lab+1),np.linspace(0,size,num_lab+1).astype(int), fontsize = 40)
        plt.xlabel('Frames',fontsize=50)
    if x_axis == 'Seconds':
        plt.xticks(np.linspace(0,size,num_lab+1),np.linspace(0,int(size/fr),num_lab+1).astype(int), fontsize = 40)
        plt.xlabel('Seconds',fontsize=50)
    if x_axis == 'Minutes':
        plt.xticks(np.linspace(0,size,num_lab+1),np.linspace(0,int(size/(fr*60)),num_lab+1).astype(int), fontsize = 40)
        plt.xlabel('Minutes',fontsize=50)
    if x_axis == 'Days':
        plt.xticks(np.linspace(0,size,num_lab+1),np.linspace(min_age,size+min_age,num_lab+1).astype(int))
        plt.xlabel('Age (days)',fontsize=50)
    
    plt.ylabel(ylabel, fontsize=50)

    if errorbar:
        size= y.shape[1]
        for a in range(0,y.shape[0]):
            for d in range(min_age,min_age + y.shape[1]):
                    plt.scatter(d,y[a,d-7],color=colour,s=700,alpha=0.7,label='_nolegend_')
        yerr = (np.nanstd(y,axis=(0)))/(np.sqrt(np.sum(~np.isnan(y),0)))
        plt.errorbar(np.linspace(min_age,min_age + y.shape[1]-1,y.shape[1]),np.nanmean(y,axis=(0)),yerr=yerr,color=colour,lw=10, capsize=15, capthick=8)
        #plt.xlim([min_age,size])
        plt.xticks(np.linspace(min_age,size+min_age,num_lab+1),np.linspace(min_age,size+min_age,num_lab+1).astype(int))
        return

    if two_d:
        return plt.plot(x,y,color=colour,lw=LW)
    else:
        return plt.plot(y,color=colour,lw=LW)


def plot_scatter(x,y,S=80,M='o',C='k',colour=False,*,CMAP=0,VMAX=0, VMIN=0, LABEL=0,fig=0):
    '''
    Generates scatter plot with some default parameters that you generally use for 1D data

    Parameters
    -------------
        x: x_Data
        y: y_Data
        C: color of the trace, by default black 
        S: thickness of the points, by default 80
        M: Marker, by default 'o'
        colour: if True there is a color map with the following setting that need definition (default False)
        CMAP: color map (e.g., 'copper')
        VMAX: Max value associated to max color
        VMIN: Min value associated to min color
        LABEL: Label for the colorbar
        fig: just do =fig or the name of your current figure

    
    Returns
    -------------
        scatter with plt.scatter(x,y)
     '''
    
    
    ax = plt.subplot(1,1,1)
    plt.rc('xtick',labelsize=50)
    plt.rc('ytick',labelsize=50)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # change all spines
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(6)
    ax.tick_params(width=10,length=25)
    if colour:
        pts = plt.scatter(x,y,s=S,c=C,cmap= CMAP, vmax=VMAX, vmin=VMIN)
        return plt.scatter(x,y,s=S,c=C,cmap= CMAP, vmax=VMAX, vmin=VMIN), fig.colorbar(pts, ticks=np.linspace(VMIN, VMAX, 5), label=LABEL)
    else:
        return plt.scatter(x,y,s=S,c=C)


def bar_scatter(label,values,y_label,colour= 'k',bar_number=0):
    '''
    Plots with some default parameters that you generally use for plotting a bar plot (mean +- SEM error bars)
    with single values scatered on top of the bar and jittered.

    Parameters
    -------------
        label = x axis label of the bar
        values = values of which to plot
        colour = color of the bar, by default black 
        bar_number = each aditional bar += 1 (default 0) 
    
    Returns
    -------------
        figure bar and scattered points
    '''
    
    plt.rc('xtick',labelsize=50)
    plt.rc('ytick',labelsize=50)
    ax = plt.subplot(111)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # change all spines
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(9)
    ax.tick_params(width=13,length=30)

    yerr = (np.nanstd(values))/(np.sqrt(5))
        
    plt.bar(label,[np.nanmean(values)], yerr =yerr,color=colour,lw=4,width=0.8,align='center',alpha=0.7,error_kw=dict(lw=7, capsize=15, capthick=7))#,fill=False

    plt.scatter(jitter(np.ones(values.shape)*bar_number), values,color = 'k',s=3000)

    plt.ylabel(y_label,fontsize=100) 
    

def legendary(legend,line_size=10.0,fontsize=40):
    '''
    Generates legends with line widths not offensively small

        Parameters
    -------------
        legend: list with relevant strings
        line_size: float number that determines the size of the line, by default 10.0
        fontsize: size of font used in the legend, by default 40

    
        Returns
    -------------
        legend 
    '''

    leg = plt.legend(legend,fontsize=fontsize) 
    for legobj in leg.legendHandles: 
        legobj.set_linewidth(line_size)
    return leg


def SNE_Cluster_Plot(SNE,n_clusters):
    '''
    Plot clusters generated by K-means

        Parameters
    -------------
        SNE: 2D data with x and y coordinates from previous dimensionality reduction (e.g., with tSNE)
        n_clusters: Number of clusters 

    
        Returns
    -------------
        A scatter plot with every cell coloured by cluster (colours generated at random)
    '''
    color = np.array(np.random.choice(range(256), size=(3,n_clusters))/256)
    distances = np.zeros([n_clusters])
    distances[distances==0]=np.nan
    clus = sc.k_means(SNE.T,n_clusters=n_clusters)[0]
    plot_scatter(SNE[0,:],SNE[1,:])
    plot_scatter(clus[:,0],clus[:,1],C='cyan',M='x',S=400)
    for i in range(0,SNE.shape[1]):
        for y in range(0,n_clusters):
            distances[y]=np.linalg.norm(SNE[:,i]-clus[y,:])
        plot_scatter(SNE[0,i],SNE[1,i],C=color[:,np.argmin(distances)])
    return clus


def Plot_Cluster_Traces(dF_F,SNE,n_clusters,save=False,name='tSNE_Clustering.pdf',num_lab = 17,fr=30):

    '''
    Plot tSNE with clusters generated by K-means and the average trace of each cluster

        Parameters
    -------------
        dF_F: dF_F traces of all cells in that recording
        SNE: 2D data with x and y coordinates from previous dimensionality reduction (e.g., with tSNE)
        n_clusters: Number of clusters 
        num_lab: number of labels in the x-axis
        fr: frame rate, by default 30

    
        Returns
    -------------
        A scatter plot with every cell coloured by cluster (colours generated at random) and the trace of each cluster
    '''
    fig = plt.figure(figsize=(60,35))
    plt.rc('xtick',labelsize=50)
    plt.rc('ytick',labelsize=50)

    gs0 = gridspec.GridSpec(n_clusters,2,figure=fig)#figure=fig,width_ratios=(1,1))


    # the following syntax does the same as the GridSpecFromSubplotSpec call above:
    #gs01 = gs0[1].subgridspec(3, 3)
    
    #plt.rc('xtick',labelsize=50)
    #plt.rc('ytick',labelsize=50)

    ax = fig.add_subplot(gs0[:,0])
    color = np.array(np.random.choice(range(256), size=(3,n_clusters))/256)
    distances = np.zeros([n_clusters,SNE.shape[1]])
    Clu_idx= np.zeros([SNE.shape[1]])
    distances[distances==0]=np.nan
    clus = sc.k_means(SNE.T,n_clusters=n_clusters)[0]
    plt.scatter(SNE[0,:],SNE[1,:])
    plt.xlabel('PCA 1',fontsize=80)
    plt.ylabel('PCA 2',fontsize=80)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(width=7,length=15)
    plt.scatter(clus[:,0],clus[:,1],color='k',s=9000,marker='x')
    for i in range(0,SNE.shape[1]):
        for y in range(0,n_clusters):
            distances[y,i]=np.linalg.norm(SNE[:,i]-clus[y,:])
        Clu_idx[i]=np.argmin(distances[:,i])
        plt.scatter(SNE[0,i],SNE[1,i],color=color[:,np.argmin(distances[:,i])], s=300)
    #MAKE AXIS THICKER
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(9)
   

    for i in range(0,n_clusters):
        ax = fig.add_subplot(gs0[i,1])
        plt.plot(np.mean(dF_F[Clu_idx==i,:],0),color=color[:,i], lw=10)
        #plt.rc('xtick',labelsize=20)
        #plt.rc('ytick',labelsize=20)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.tick_params(width=7,length=15)
        plt.xlim([0,dF_F.shape[1]])
        size = dF_F.shape[1]
        plt.xticks(np.linspace(0,size,num_lab),np.linspace(0,int(size/(fr*60)),num_lab).astype(int))
        plt.ylabel('ΔF/F',fontsize=50)
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(9)
        #plt.rc('xtick',labelsize=30)
        #plt.rc('ytick',labelsize=30)
    plt.xlabel('Minutes',fontsize=80)

    #plt.suptitle("GridSpec Inside GridSpec")
    plt.tight_layout()
    if save:
        plt.savefig(name,bbox_inches = 'tight',dpi=300)
    plt.show()

    return distances

def Heat_Map(dF,cmap='hot',num_lab=11,fr=30):
    '''
    Plot Heat Map 

        Parameters
    -------------
        dF: dF_F traces of all cells in that recording (cellsxFrames)
        cmap: color scheme, default 'hot'
        num_lab: number of labels in the x-axis, by default 11
        fr: frame rate, by default 30

    
        Returns
    -------------
        A Heat map
    '''
    fig= plt.figure(figsize=(30,10))
    hm = plt.imshow(dF, cmap=cmap, interpolation='nearest',aspect='auto')
    fig.colorbar(hm, ticks=np.linspace(0, np.max(dF), 5),orientation="horizontal",label='ΔF/F',location='top',aspect=200, pad=0.009,fraction=0.2)
    size = dF.shape[1]
    plt.xticks(np.linspace(0,size,num_lab),np.linspace(0,int(size/(fr*60)),num_lab).astype(int))
    plt.ylabel('Neurons',fontsize=40)
    plt.xlabel('Minutes',fontsize=40)


def Plot_Sleep_Scoring(Marker,Sleep,t=180):
    '''
    Plot sleep scoring outcome

        Parameters
    -------------
        Marker: Numpy vector with forelimb movement
        Sleep: Numpy vector, with lenght number of bins, where 0 is awake, 1 QS and 2 AS
        t= bin size in frames (default 180, 6s at 30Hz)

        Returns
    -------------
        A plot with the Marker trace and labeled bins based on sleep state
    '''
    
    plot_1D(Marker, x_axis = 'Minutes', ylabel='Forelimb Movement (a.u.)',LW=1)
    for i in range(0,len(Sleep)):
        if Sleep[i]==0:
            plt.plot(range(i*t,(i+1)*t),np.ones(t)*-8,'darkblue', lw=5)
        if Sleep[i]==1:
            plt.plot(range(i*t,(i+1)*t),np.ones(t)*-13,'darkcyan',lw=5)
        if Sleep[i]==2:
            plt.plot(range(i*t,(i+1)*t),np.ones(t)*-18,'slategray',lw=5)
    plt.text(25000,400,'Awake',fontsize=30,color='darkblue')
    plt.text(25000,360,'Quiet Sleep',fontsize=30,color='darkcyan')
    plt.text(25000,320,'Active Sleep',fontsize=30,color='slategray')
    
    return

def Plot_periH_heat_map(periH,num_cells=2000,vmin=-2,vmax=6):
    '''
    Function to plot a heat map of the peri H-event dF/F for a given recording

    Parameters
    --------------
    periH: numpy array
        Array with the peri H-event dF/F for all the H-events in the row
    num_cells: int
        Number of cells to plot
    vmin: int
        Minimum value for the heat map
    vmax: int
        Maximum value for the heat map

    Returns
    --------------
    Plot of the heat map

    '''

    Av = np.nanmedian(periH,axis=0)
    #Arrange Av rows in order of max value
    Av = Av[np.argsort(np.nanmax(Av,axis=1))[::-1],:]
    #Plot heat map
    plt.imshow(Av[50:num_cells+50],aspect='auto',cmap='jet',vmin=vmin,vmax=vmax)
    #plt.colorbar()
    #Place at 20, 150, 300, 450, 600 - 5, -2.5, 0, 2.5, 5
    plt.xticks([0,75,150,225,300],[-5,-2.5,0,2.5,5])
    plt.xlabel('Peri H-Event time (s)',fontsize=50)
    plt.yticks([0,500,1000,1500,2000],[0,500,1000,1500,2000])
    plt.ylabel('Neurons',fontsize=60)