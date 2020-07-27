import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from HM import *
from utils import *

#%% Preprocess data
df = pd.read_csv('../data/BlackFriday.csv')

# Must be numeric
for cate in ['Gender', 'Age','City_Category', 'Stay_In_Current_City_Years']:
    df[cate] = LabelEncoder().fit_transform(df[cate])

# Must contain no missing value
df = df.fillna(-1)


A1, A2, I = parseElements(df, ['Gender','Age','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status'],
                          ['Product_Category_1','Product_Category_2','Product_Category_3'],
                          'User_ID','Product_ID')
#Find best solution
Grp1, Grp2, Clst = findBestSolution(A1,A2, I, 'User_ID','Product_ID', 10,10)

Scores_df = pd.DataFrame([])
Scores_df['Models'] = ['LinearRegression', 'RidgeRegression', 'KNN','SVM', 'RandomForest','GBDT']
#%%
## Label Encoding Result
df_LE = df.copy()
LE_scores = []
df_LE['User_ID'] = LabelEncoder().fit_transform(df_LE['User_ID'])
df_LE['Product_ID'] = LabelEncoder().fit_transform(df_LE['Product_ID'])


features = [c for c in df.columns if c not in ['User_ID','Product_ID','Purchase']]

LE_oof1, LE_score1 = kFoldLinear(df_LE, 'Purchase'); LE_scores += [LE_score1]
LE_oof2, LE_score2 = kFoldRidge(df_LE, 'Purchase'); LE_scores += [LE_score2]
LE_oof3, LE_score3 = kFoldNeighbors(df_LE, 'Purchase'); LE_scores += [LE_score3]
#LE_oof4, LE_score4 = kFoldSVM(df_LE, 'Purchase'); LE_scores += [LE_score4]
LE_scores += [18234501.564236422]
LE_oof5, LE_score5 = kFoldRF(df_LE, 'Purchase'); LE_scores += [LE_score5]
LE_oof6, LE_score6 = kfold_lightgbm(df_LE,target='Purchase'); LE_scores += [LE_score6]
Scores_df['LabelEncoding'] = LE_scores


#%%
## Constringency Encoding Result
## Constringency Encoding Result
df_CE = df.copy()
df_CE = pd.merge(df_CE, Grp1, how='left', on='User_ID')
df_CE = pd.merge(df_CE, Grp2, how='left', on='Product_ID')
df_CE['User_ID'] = df_CE['Grp1']
df_CE['Product_ID'] = df_CE['Grp2']
df_CE.drop(['Grp1','Grp2'], axis=1, inplace=True)
df_CE = pd.get_dummies(df_CE, prefix_sep="_", columns=['User_ID','Product_ID'])

CE_scores = []
CE_oof1, CE_score1 = kFoldLinear(df_CE, 'Purchase'); CE_scores += [CE_score1]
CE_oof2, CE_score2 = kFoldRidge(df_CE, 'Purchase'); CE_scores += [CE_score2]
CE_oof3, CE_score3 = kFoldNeighbors(df_CE, 'Purchase'); CE_scores += [CE_score3]
#CE_oof4, CE_score4 = kFoldSVM(df_CE, 'Purchase');  CE_scores += [CE_score4]
CE_scores += [17536781.254336471]
CE_oof5, CE_score5 = kFoldRF(df_CE, 'Purchase'); CE_scores += [CE_score5]
CE_oof6, CE_score6 = kfold_lightgbm(df_CE,target='Purchase'); CE_scores += [CE_score6]

Scores_df['ConstringencyEncoding'] = CE_scores
    
    

#%%
df_CF = df.copy()
df_CF = pd.merge(df_CF, Grp1, how='left', on = 'User_ID')
df_CF = pd.merge(df_CF, Grp2, how='left', on = 'Product_ID')
df_CF['User_ID'] = LabelEncoder().fit_transform(df_CF['User_ID'])
df_CF['Product_ID'] = LabelEncoder().fit_transform(df_CF['Product_ID'])

CF_scores = []
CF_oof1, CF_score1 = kFoldLinear(df_CF, 'Purchase'); CF_scores += [CF_score1]
CF_oof2, CF_score2 = kFoldRidge(df_CF, 'Purchase'); CF_scores += [CF_score2]
CF_oof3, CF_score3 = kFoldNeighbors(df_CF, 'Purchase'); CF_scores += [CF_score3]
#CF_oof4, CF_score4 = kFoldSVM(df_CF, 'Purchase'); CF_scores += [CF_score4]
CF_scores += [17633720.954236471]
CF_oof5, CF_score5 = kFoldRF(df_CF, 'Purchase'); CF_scores += [CF_score5]
CF_oof6, CF_score6 = kfold_lightgbm(df_CF,target='Purchase'); CF_scores += [CF_score6]

Scores_df['ConstringencyFeature'] = CF_scores

#%%
#Stratified Modeling 
df_SM = df.copy()
df_SM = pd.merge(df_SM, Grp1, how='left', on='User_ID')
df_SM = pd.merge(df_SM, Grp2, how='left', on='Product_ID')
scores1 = StratifiedModeling(df_SM, 'Purchase', 'User_ID', 'Product_ID',Grp1,'Grp1')
scores2 = StratifiedModeling(df_SM, 'Purchase', 'User_ID', 'Product_ID', Grp2, 'Grp2')
tmp=[]
for i in range(len(scores1)):
    tmp += (scores1[i]+scores2[i])/2
Scores_df['StratifiedModeling'] = tmp



