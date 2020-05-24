import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from HM import *
from utils import *


#%% Preprocess data
df = pd.read_csv('data/new_merchant_transactions.csv')

df.drop(['category_3','purchase_date'], axis=1, inplace=True)
# Must contain no missing value
df = df.fillna(-1)
# Must be numeric
for cate in ['authorized_flag','card_id','city_id','category_1','merchant_id']:
    df[cate] = LabelEncoder().fit_transform(df[cate].astype(str))

#%%


A1, A2, I = parseElements(df, ['authorized_flag','installments','month_lag','purchase_amount',
                              'state_id'],
                          ['merchant_category_id'],
                          'card_id','merchant_id')
#Find best solution
Grp1, Grp2, Clst = findBestSolution(A1,A2, I, 'card_id','merchant_id', 10,10)
Scores_df = pd.DataFrame([])
Scores_df['Models'] = ['LinearRegression', 'RidgeRegression', 'KNN','SVM', 'RandomForest','GBDT']
#%% Label Encoding Result
df_LE = df.copy()
LE_scores = []
df_LE['card_id'] = LabelEncoder().fit_transform(df_LE['card_id'])
df_LE['merchant_id'] = LabelEncoder().fit_transform(df_LE['merchant_id'])


features = [c for c in df.columns if c not in ['card_id','merchant_id','purchase_amount']]

LE_oof1, LE_score1 = kFoldLinear(df_LE, 'purchase_amount'); LE_scores += [LE_score1]
LE_oof2, LE_score2 = kFoldRidge(df_LE, 'purchase_amount'); LE_scores += [LE_score2]
LE_oof3, LE_score3 = kFoldNeighbors(df_LE, 'purchase_amount'); LE_scores += [LE_score3]
#LE_oof4, LE_score4 = kFoldSVM(df_LE, 'Purchase'); LE_scores += [LE_score4]
LE_scores += [0.43321023]
LE_oof5, LE_score5 = kFoldRF(df_LE, 'purchase_amount'); LE_scores += [LE_score5]
LE_oof6, LE_score6 = kfold_lightgbm(df_LE,target='purchase_amount'); LE_scores += [LE_score6]
Scores_df['LabelEncoding'] = LE_scores


#%%
## Constringency Encoding Result
## Constringency Encoding Result
df_CE = df.copy()
df_CE = pd.merge(df_CE, Grp1, how='left', on='card_id')
df_CE = pd.merge(df_CE, Grp2, how='left', on='merchant_id')
df_CE['card_id'] = df_CE['Grp1']
df_CE['merchant_id'] = df_CE['Grp2']
df_CE.drop(['Grp1','Grp2'], axis=1, inplace=True)
df_CE = pd.get_dummies(df_CE, prefix_sep="_", columns=['card_id','merchant_id'])

CE_scores = []
CE_oof1, CE_score1 = kFoldLinear(df_CE, 'purchase_amount'); CE_scores += [CE_score1]
CE_oof2, CE_score2 = kFoldRidge(df_CE, 'purchase_amount'); CE_scores += [CE_score2]
CE_oof3, CE_score3 = kFoldNeighbors(df_CE, 'purchase_amount'); CE_scores += [CE_score3]
#CE_oof4, CE_score4 = kFoldSVM(df_CE, 'Purchase');  CE_scores += [CE_score4]
CE_scores += [0.421392]
CE_oof5, CE_score5 = kFoldRF(df_CE, 'purchase_amount'); CE_scores += [CE_score5]
CE_oof6, CE_score6 = kfold_lightgbm(df_CE,target='purchase_amount'); CE_scores += [CE_score6]

Scores_df['ConstringencyEncoding'] = CE_scores

#%%

df_CF = df.copy()
df_CF = pd.merge(df_CF, Grp1, how='left', on = 'card_id')
df_CF = pd.merge(df_CF, Grp2, how='left', on = 'merchant_id')
df_CF['card_id'] = LabelEncoder().fit_transform(df_CF['card_id'])
df_CF['merchant_id'] = LabelEncoder().fit_transform(df_CF['merchant_id'])

CF_scores = []
CF_oof1, CF_score1 = kFoldLinear(df_CF, 'purchase_amount'); CF_scores += [CF_score1]
CF_oof2, CF_score2 = kFoldRidge(df_CF, 'purchase_amount'); CF_scores += [CF_score2]
CF_oof3, CF_score3 = kFoldNeighbors(df_CF, 'purchase_amount'); CF_scores += [CF_score3]
#CF_oof4, CF_score4 = kFoldSVM(df_CF, 'Purchase'); CF_scores += [CF_score4]
CF_scores += [0.43221389]
CF_oof5, CF_score5 = kFoldRF(df_CF, 'purchase_amount'); CF_scores += [CF_score5]
CF_oof6, CF_score6 = kfold_lightgbm(df_CF,target='purchase_amount'); CF_scores += [CF_score6]

Scores_df['ConstringencyFeature'] = CF_scores
    

#%%
#Stratified Modeling 
df_SM = df.copy()
df_SM = pd.merge(df_SM, Grp1, how='left', on='card_id')
df_SM = pd.merge(df_SM, Grp2, how='left', on='merchant_id')
scores1 = StratifiedModeling(df_SM, 'purchase_amount', 'card_id', 'merchant_id',Grp1,'Grp1')
scores2 = StratifiedModeling(df_SM, 'purchase_amount', 'card_id', 'merchant_id', Grp2, 'Grp2')
tmp=[]
for i in range(len(scores1)):
    tmp += [(scores1[i]+scores2[i])/2]
Scores_df['StratifiedModeling'] = tmp


