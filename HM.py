import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from utils import *
import warnings
warnings.filterwarnings("ignore")


def parseElements(df, A1feats, A2feats, A1name, A2name):
    df1 = df.groupby(A1name)[A1feats].mean()
    df2 = df.groupby(A2name)[A2feats].mean()
    I = df[[A1name, A2name]]
    return df1, df2, I

def Constringency(Grp1,Grp2,I,A1name,A2name,numC1,numC2):
    
    #Kmeans for PreClustering
    I = pd.merge(I, Grp1, on=A1name)
    I = pd.merge(I, Grp2, on=A2name)
   
    #Calculate Forward Constringency
    cf = 0 
    for i in np.arange(numC1):
        subWeight = (Grp1['Grp1']==i).sum()/ len(Grp1)
        subclass = I[I['Grp1'] == i]
        maxratio = subclass.groupby('Grp2')['Grp2'].count().max() / len(subclass)
        cf += subWeight * maxratio
    #Calculate Backward Constringency
    cb = 0        
    for i in np.arange(numC2):
        subWeight = (Grp2['Grp2']==i).sum()/ len(Grp2)
        subclass = I[I['Grp2'] == i]
        maxratio = subclass.groupby('Grp1')['Grp1'].count().max() / len(subclass)
        cb += subWeight * maxratio       

    #Calculate Baseline
    bf = len(Grp2[Grp2['Grp2']==Grp2['Grp2'].mode()[0]])/len(Grp2)
    bb = len(Grp1[Grp1['Grp1']==Grp1['Grp1'].mode()[0]])/len(Grp1)
    #Relative Error
    #print(cf,bf,cb,bb)
    Rcf = np.abs(cf-bf)/bf
    Rcb = np.abs(cb-bb)/bb
      
    return Rcf,Rcb

def findBestSolution(A1df, A2df,I, A1name, A2name, maxi, maxj):
    clist = []
    idxlist = []
    for i in np.arange(maxi-1)+2:
        Cluster1 = KMeans(n_clusters=i, random_state=0).fit(A1df.values).labels_
        Grp1 = pd.DataFrame({A1name:A1df.index, 'Grp1':Cluster1})
        for j in np.arange(maxj-1)+2:
            Cluster2 = KMeans(n_clusters=j, random_state=0).fit(A2df.values).labels_
            Grp2 = pd.DataFrame({A2name:A2df.index, 'Grp2':Cluster2})
            #Forward Constringency
            cf,cb = Constringency(Grp1, Grp2, I, A1name, A2name, i, j)
            c_mean = (cf+cb)/2.0
            print("Relative Constringency for {}-{} solution is {}".format(i,j,c_mean))
            clist += [c_mean]
            idxlist += [[i,j]]
    
    print("The best solution is:")
    bestidx = np.argmax(clist)
    besti = idxlist[bestidx][0]
    bestj =  idxlist[bestidx][1]
    print("{}-{} solution, Relative Constringency = {}".format(besti,bestj,clist[bestidx]))
    
    Cluster1 = KMeans(n_clusters=besti, random_state=0).fit(A1df.values).labels_
    Grp1 = pd.DataFrame({A1name:A1df.index, 'Grp1':Cluster1})
    Cluster2 = KMeans(n_clusters=bestj, random_state=0).fit(A2df.values).labels_
    Grp2 = pd.DataFrame({A2name:A2df.index, 'Grp2':Cluster2})    
    
    return Grp1, Grp2, clist
            

