from sklearn.linear_model import BayesianRidge
from sklearn.svm import LinearSVR
from sklearn.model_selection import StratifiedKFold
import pandas as pd

def cv_bayesianRidge(trn_df ,val_df):
    k = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
    X = trn_df[trn_df.ridge_preds.notnull()][['id', 'd','y_pred','ridge_preds', 'lasso_preds']]
    y = trn_df[trn_df.ridge_preds.notnull()]['TARGET']
    X['br_pred'] = 0
    val_df['br_pred'] = 0
    X['br_std'] = 0
    val_df['br_std'] = 0
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    for trn_indx, val_indx in k.split(X,y=y):
        
        br = BayesianRidge()
        br.fit(X.loc[trn_indx,['y_pred','ridge_preds', 'lasso_preds']], y.loc[trn_indx])
        p, _std= br.predict(X.loc[val_indx,['y_pred','ridge_preds', 'lasso_preds']], return_std=True)
        X.loc[val_indx, 'br_pred'] = p 
        X.loc[val_indx, 'br_std'] = _std
        p, _std= br.predict(val_df[['y_pred','ridge_preds', 'lasso_preds']], return_std=True)
        val_df['br_pred'] += p/5
        val_df['br_std'] += _std/5
    trn_df = pd.merge(trn_df, X[['id', 'd','br_pred', 'br_std']], how='outer', on=['id', 'd'])
    return trn_df, val_df

def cv_linearSVR(trn_df ,val_df):
    k = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
    X = trn_df[trn_df.ridge_preds.notnull()][['id','d','y_pred','ridge_preds', 'lasso_preds']]
    y = trn_df[trn_df.ridge_preds.notnull()]['TARGET']
    X['svm_pred'] = 0
    val_df['svm_pred'] = 0
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    for trn_indx, val_indx in k.split(X,y=y):
        
        svm = LinearSVR(random_state =2020)
        svm.fit(X.loc[trn_indx,['y_pred','ridge_preds', 'lasso_preds']], y.loc[trn_indx])
        p= svm.predict(X.loc[val_indx,['y_pred','ridge_preds', 'lasso_preds']])
        X.loc[val_indx, 'br_pred'] = p
        p = svm.predict(val_df[['y_pred','ridge_preds', 'lasso_preds']])
        val_df['br_pred'] += p/5
    trn_df = pd.merge(trn_df, X[['id', 'd','svm_pred']], how='outer', on=['id', 'd'])
    return trn_df, val_df