import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from HM import *
from utils import *

#%% Generate 50 agents and 5000 interactions
df = pd.DataFrame();
for i in range(50):
    mean = np.random.randint(100)
    sd = np.random.rand()*4
    tmp = pd.DataFrame(sd*np.random.randn(100,10)+mean)
    tmp.columns = list(np.arange(9))+['target1']    
    tmp['Agent0'] = i
    df = df.append(tmp)

df = df.sample(frac=1).reset_index(drop=True)

df2 = pd.DataFrame();
for i in range(25):
    mean = np.random.randint(20)
    sd = np.random.rand()*8
    tmp = pd.DataFrame(sd*np.random.randn(200,10)+mean)
    tmp.columns = list(np.arange(9)+10)+['target2']    
    tmp['Agent1'] = i
    df2 = df2.append(tmp)

df2 = df2.sample(frac=1).reset_index(drop=True)

df = pd.concat([df,df2], axis=1)
df['target'] = df['target1'] + df['target2']
df.drop(['target1', 'target2'], axis=1, inplace=True)

    
#%%
A1, A2, I = parseElements(df, np.arange(0,9), np.arange(10,19), 'Agent0', 'Agent1')
Grp1, Grp2, Clst = findBestSolution(A1,A2, I, 'Agent0','Agent1', 10,10)
Scores_df = pd.DataFrame([])
Scores_df['Models'] = ['LinearRegression', 'RidgeRegression', 'KNN', 'RandomForest','GBDT']

#%%
## Label Encoding Result
df_LE = df.copy()
LE_scores = []
features = [c for c in df.columns if c not in ['Agent0','Agent1','target']]

LE_oof1, LE_score1 = kFoldLinear(df_LE, 'target'); LE_scores += [LE_score1]
LE_oof2, LE_score2 = kFoldRidge(df_LE, 'target'); LE_scores += [LE_score2]
LE_oof3, LE_score3 = kFoldNeighbors(df_LE, 'target'); LE_scores += [LE_score3]
#LE_oof4, LE_score4 = kFoldSVM(df_LE, 'target'); LE_scores += [LE_score4]
LE_oof5, LE_score5 = kFoldRF(df_LE, 'target'); LE_scores += [LE_score5]
LE_oof6, LE_score6 = kfold_lightgbm(df_LE,target='target'); LE_scores += [LE_score6]

Scores_df['LabelEncoding'] = LE_scores


#%%
## Constringency Encoding Result
df_CE = df.copy()
df_CE = pd.merge(df_CE, Grp1, how='left', on='Agent0')
df_CE = pd.merge(df_CE, Grp2, how='left', on='Agent1')
df_CE['Agent0'] = df_CE['Grp1']
df_CE['Agent1'] = df_CE['Grp2']
df_CE.drop(['Grp1','Grp2'], axis=1, inplace=True)
df_CE = pd.get_dummies(df_CE, prefix_sep="_", columns=['Agent0','Agent1'])

CE_scores = []
CE_oof1, CE_score1 = kFoldLinear(df_CE, 'target'); CE_scores += [CE_score1]
CE_oof2, CE_score2 = kFoldRidge(df_CE, 'target'); CE_scores += [CE_score2]
CE_oof3, CE_score3 = kFoldNeighbors(df_CE, 'target'); CE_scores += [CE_score3]
#CE_oof4, CE_score4 = kFoldSVM(df_CE, 'target'); CE_scores += [CE_score4]
CE_oof5, CE_score5 = kFoldRF(df_CE, 'target'); CE_scores += [CE_score5]
CE_oof6, CE_score6 = kfold_lightgbm(df_CE,target='target'); CE_scores += [CE_score6]

Scores_df['ConstringencyEncoding'] = CE_scores
    

#%%
#Constringency Feature
df_CF = df.copy()
df_CF = pd.merge(df_CF, Grp1, how='left', on = 'Agent0')
df_CF = pd.merge(df_CF, Grp2, how='left', on = 'Agent1')

CF_scores = []
CF_oof1, CF_score1 = kFoldLinear(df_CF, 'target'); CF_scores += [CF_score1]
CF_oof2, CF_score2 = kFoldRidge(df_CF, 'target'); CF_scores += [CF_score2]
CF_oof3, CF_score3 = kFoldNeighbors(df_CF, 'target'); CF_scores += [CF_score3]
#CF_oof4, CF_score4 = kFoldSVM(df_CF, 'target'); CF_scores += [CF_score4]
CF_oof5, CF_score5 = kFoldRF(df_CF, 'target'); CF_scores += [CF_score5]
CF_oof6, CF_score6 = kfold_lightgbm(df_CF,target='target'); CF_scores += [CF_score6]

Scores_df['ConstringencyFeature'] = CF_scores

#%%
#Stratified Modeling 
df_SM = df.copy()
df_SM = pd.merge(df_SM, Grp1, how='left', on='Agent0')
df_SM = pd.merge(df_SM, Grp2, how='left', on='Agent1')
scores1 = StratifiedModeling(df_SM, 'target', 'Agent0', 'Agent1',Grp1,'Grp1')
scores2 = StratifiedModeling(df_SM, 'target', 'Agent0', 'Agent1', Grp2, 'Grp2')
tmp=[]
for i in range(len[scores1]):
    tmp += (scores1[i]+scores2[i])/2
Scores_df['StratifiedModeling'] = tmp




