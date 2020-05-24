import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import RidgeClassifier, Ridge, LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import KFold, StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC, SVR
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.spatial import distance_matrix
import seaborn as sns
import lightgbm as lgb
from sklearn.covariance import ShrunkCovariance, GraphicalLasso, LedoitWolf, OAS, MinCovDet
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


from sklearn.metrics import recall_score

#%%

def kFoldRF(train,target, nfold=5, classification=False):
    oof = np.zeros(len(train))
    X = train.drop(target, axis=1)
    y = train[target]

    kf = KFold(n_splits=nfold,shuffle=True, random_state=42)
    if classification:
        for trn_idx, val_idx in kf.split(X,y):
            clf = RandomForestClassifier()
            clf.fit(X.iloc[trn_idx], y.iloc[trn_idx])
            oof[val_idx] = clf.predict(X.iloc[val_idx])
        print('RF oof acc score: {}'.format(accuracy_score(oof, train[target])))
        return oof,accuracy_score(oof, train[target])
    else:
        for trn_idx, val_idx in kf.split(X,y):
            clf = RandomForestRegressor()
            clf.fit(X.iloc[trn_idx], y.iloc[trn_idx])
            oof[val_idx] = clf.predict(X.iloc[val_idx])
        print('RF oof mse: {}'.format(mean_squared_error(oof, train[target])))
        return oof, mean_squared_error(oof, train[target])


def kFoldLinear(train,target, nfold=5, classification=False):
    oof = np.zeros(len(train))
    X = train.drop(target, axis=1)
    y = train[target]

    if classification:
        kf = KFold(n_splits=nfold,shuffle=True,  random_state=42)
        for trn_idx, val_idx in kf.split(X,y):
            clf = LogisticRegression()
            clf.fit(X.iloc[trn_idx], y.iloc[trn_idx])
            oof[val_idx] = clf.predict(X.iloc[val_idx])
        print('LR oof acc score: {}'.format(accuracy_score(oof, train[target])))
        return oof, accuracy_score(oof, train[target])
    else:
        kf = KFold(n_splits=nfold,shuffle=True,  random_state=42)
        for trn_idx, val_idx in kf.split(X,y):
            clf = LinearRegression()
            clf.fit(X.iloc[trn_idx], y.iloc[trn_idx])
            oof[val_idx] = clf.predict(X.iloc[val_idx])
        print('LR oof mse: {}'.format(mean_squared_error(oof, train[target])))    
        return oof, mean_squared_error(oof, train[target])
    
def kFoldNeighbors(train, target, nfold=5, classification=False):
    oof = np.zeros(len(train))
    X = train.drop(target, axis=1)
    y = train[target]
    
    if classification:
        kf = KFold(n_splits=nfold, shuffle=True, random_state=42)
        for trn_idx,val_idx in kf.split(X,y):
            clf = KNeighborsClassifier()
            clf.fit(X.iloc[trn_idx], y.iloc[trn_idx])
            oof[val_idx] = clf.predict(X.iloc[val_idx])
        print('kNN oof acc score: {}'.format(accuracy_score(oof, train[target])))
        return oof, accuracy_score(oof, train[target])
    else:
        kf = KFold(n_splits=nfold,shuffle=True,  random_state=42)
        for trn_idx, val_idx in kf.split(X,y):
            clf = KNeighborsRegressor()
            clf.fit(X.iloc[trn_idx], y.iloc[trn_idx])
            oof[val_idx] = clf.predict(X.iloc[val_idx])
        print('kNN oof mse: {}'.format(mean_squared_error(oof, train[target])))    
        return oof, mean_squared_error(oof, train[target])
    
    
def kFoldLDA(train, target, nfold=5):
    oof = np.zeros(len(train))
    X = train.drop(target, axis=1)
    y = train[target]
    kf = KFold(n_splits=nfold, shuffle=True, random_state=42)
    for trn_idx,val_idx in kf.split(X,y):
        clf = LinearDiscriminantAnalysis()
        clf.fit(X.iloc[trn_idx], y.iloc[trn_idx])
        oof[val_idx] = clf.predict(X.iloc[val_idx])
    print('LDA oof acc score: {}'.format(accuracy_score(oof, train[target])))
    return oof, accuracy_score(oof, train[target])

    
def kFoldQDA(train, target, nfold=5):
    oof = np.zeros(len(train))
    X = train.drop(target, axis=1)
    y = train[target]
    
    kf = KFold(n_splits=nfold, shuffle=True, random_state=42)
    for trn_idx,val_idx in kf.split(X,y):
        clf = QuadraticDiscriminantAnalysis()
        clf.fit(X.iloc[trn_idx], y.iloc[trn_idx])
        oof[val_idx] = clf.predict(X.iloc[val_idx])
    print('QDA oof acc score: {}'.format(accuracy_score(oof, train[target])))
    return oof, accuracy_score(oof, train[target])

    

def kFoldRidge(train,target, nfold=5, classification=False):
    oof = np.zeros(len(train))
    X = train.drop(target, axis=1)
    y = train[target]

    if classification:
        kf = KFold(n_splits=nfold,shuffle=True,  random_state=42)
        for trn_idx, val_idx in kf.split(X,y):
            clf = RidgeClassifier(alpha=5.0)
            clf.fit(X.iloc[trn_idx], y.iloc[trn_idx])
            oof[val_idx] = clf.predict(X.iloc[val_idx])
        print('Ridge oof acc score: {}'.format(accuracy_score(oof, train[target])))
        return oof, accuracy_score(oof, train[target])
    else:
        kf = KFold(n_splits=nfold,shuffle=True,  random_state=42)
        for trn_idx, val_idx in kf.split(X,y):
            clf = Ridge(alpha=2.0)
            clf.fit(X.iloc[trn_idx], y.iloc[trn_idx])
            oof[val_idx] = clf.predict(X.iloc[val_idx])
        print('Ridge oof mse: {}'.format(mean_squared_error(oof, train[target])))    
        return oof, mean_squared_error(oof, train[target])

def kFoldSVM(train,target, nfold=5, classification=False):
    oof = np.zeros(len(train))
    X = train.drop(target, axis=1)
    y = train[target]
    if classification:
        kf = KFold(n_splits=nfold,shuffle=True, random_state=42)
        for trn_idx, val_idx in kf.split(X,y):
            clf = SVC()
            clf.fit(X.iloc[trn_idx], y.iloc[trn_idx])
            oof[val_idx] = clf.predict(X.iloc[val_idx])
        print('SVC oof acc score: {}'.format(accuracy_score(oof, train[target])))
        return accuracy_score(oof, train[target])
    else:
        kf = KFold(n_splits=nfold,shuffle=True,  random_state=42)
        for trn_idx, val_idx in kf.split(X,y):
            clf = SVR()
            clf.fit(X.iloc[trn_idx], y.iloc[trn_idx])
            oof[val_idx] = clf.predict(X.iloc[val_idx])
        print('SVR oof mse: {}'.format(mean_squared_error(oof, train[target])))   
        return oof, mean_squared_error(oof, train[target])



def display_importances(feature_importance_df_):
    feature_importance_df_ = feature_importance_df_.reset_index()
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(20, 10))
    sns.barplot(x="importance", y="feature", data=best_features, orient = 'h', order = best_features.feature)
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('FeatureImportance.png')
    return best_features

def kfold_lightgbm(train_df, num_folds=5, feat=None, target=None, classification=False):

    folds = KFold(n_splits= num_folds, shuffle=True, random_state=326)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    feature_importance_df = pd.DataFrame()
    
    if feat is not None:
        feats = [f for f in feat if f not in [target]]
    else:
        feats = [f for f in train_df.columns if f not in [target]]

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df[target])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df[target].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df[target].iloc[valid_idx]

        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)
        
        if not classification:  
            params = {'num_leaves': 32,
             'objective':'regression',
             'max_depth': -1,
             'learning_rate': 0.05,
             "boosting": "gbdt",
             "metric": 'mse',
             "verbosity": -1,
             "random_state": 2019}
        
            
            reg = lgb.train(
                            params,
                            lgb_train,
                            valid_sets=[lgb_train, lgb_test],
                            valid_names=['train', 'test'],
                            num_boost_round=10000,
                            early_stopping_rounds= 200,
                            verbose_eval=-1
                            )
            
        else:
            params = {'num_leaves': 32,
             'objective':'multiclass',
             'max_depth': -1,
             'learning_rate': 0.05,
             "boosting": "gbdt",
             "verbosity": -1,
             "random_state": 2019}
        
            
            reg = lgb.train(
                            params,
                            lgb_train,
                            valid_sets=[lgb_train, lgb_test],
                            valid_names=['train', 'test'],
                            num_boost_round=10000,
                            early_stopping_rounds= 200,
                            verbose_eval=-1
                            )           
            

        oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)
            
            
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = np.log1p(reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))
        fold_importance_df["fold"] = n_fold + 1
        
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        if classification:
            print('Fold {} accuracy : {}'.format( n_fold + 1, accuracy_score(valid_y, oof_preds[valid_idx])))
        else:
            print('Fold {} mse : {}'.format(n_fold + 1, mean_squared_error(valid_y, oof_preds[valid_idx])))
        del reg, train_x, train_y, valid_x, valid_y

    # display importances

    feature_importance_df = feature_importance_df.groupby('feature').agg({'importance':['mean']})
    feature_importance_df.columns = ['importance']
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    display_importances(feature_importance_df)
    
    if classification:
        acc = accuracy_score(oof_preds, train_df[target])
        print('LGBM oof accuracy: {}'.format(accuracy_score(oof_preds, train_df[target])))
    else:
        acc = mean_squared_error(oof_preds, train_df[target])
        print('LGBM oof mse: {}'.format(mean_squared_error(oof_preds, train_df[target])))
    return oof_preds, acc

def StratifiedModeling(df, target,agent1, agent2, Grp, GrpName):
    print("target is", target)
    df_Scores = []
    pred1 = np.zeros(len(df))
    pred2 = np.zeros(len(df)) 
    pred3 = np.zeros(len(df))
    pred5 = np.zeros(len(df))
    pred6 = np.zeros(len(df))
    for i in Grp[GrpName].unique():
        currentIdx = df[df[GrpName] == i].index
        subtrain = df.iloc[currentIdx]
        oof1,_= kFoldLinear(subtrain, target); pred1[currentIdx] = oof1
        oof2,_ = kFoldRidge(subtrain, target); pred2[currentIdx] = oof2
        oof3,_= kFoldNeighbors(subtrain, target); pred3[currentIdx] = oof3
        CF_oof4, CF_score4 = kFoldSVM(df_CF, target); CF_scores += [CF_score4]
        oof5,_= kFoldRF(subtrain, target); pred5[currentIdx] = oof5
        oof6,_= kfold_lightgbm(subtrain,target=target); pred6[currentIdx] = oof6
    df_Scores += [mean_squared_error(df[target], pred1)]
    df_Scores += [mean_squared_error(df[target], pred2)]
    df_Scores += [mean_squared_error(df[target], pred3)]
    df_Scores += [mean_squared_error(df[target], pred5)]
    df_Scores += [mean_squared_error(df[target], pred6)]
    return df_Scores
        