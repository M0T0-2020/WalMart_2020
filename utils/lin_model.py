import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
import gc, pickle
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, log_loss
from sklearn.linear_model import Ridge,Lasso

def to_onehot_data(data):
    data.drop(columns=['item_id','state_id'], inplace=True)
    category = ['cat_id', 'dept_id', 'store_id', 'month', 'wday']
    for cat in category:
        if cat!='dept_id':
            data = pd.concat([
                data.drop(cat, axis=1),
                pd.get_dummies(data[cat]).rename(columns={i:f'{cat}_{int(i)}' for i in data[cat].unique()})
            ], axis=1)
        else:
            data = pd.concat([
                data,
                pd.get_dummies(data[cat]).rename(columns={i:f'{cat}_{int(i)}' for i in data[cat].unique()})
            ], axis=1)
    return data

def data_split_lin(data, trn_days, val_days):
    data.dropna(0, inplace=True)
    ids = data[data.d==trn_days[0]].id.unique().tolist()
    train = data[data.d.isin(trn_days)]
    train = train[train.id.isin(ids)]
    val = data[data.d.isin(val_days)]
    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    trn_df = train[['id', 'd', 'TARGET']]
    val_df = val[['id', 'd', 'TARGET']]
    return train, val, trn_df, val_df

def linear_cv(data, trn_df):
    k = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
    models={}
    models['ridge'] = []
    models['lasso'] = []
    data.reset_index(drop=True, inplace=True)
    X = data.drop(columns=['dept_id','id', 'd', 'TARGET'])
    y = data['TARGET']
    data['ridge_preds'] = 0
    data['lasso_preds'] = 0
    for trn_indx, val_indx in k.split(data[['dept_id']],y=y):
        
        ridge = Ridge()
        lasso = Lasso()
    
        ridge.fit(X.loc[trn_indx,:],y.loc[trn_indx])
        lasso.fit(X.loc[trn_indx,:],y.loc[trn_indx])
        models['ridge'].append(ridge)
        models['lasso'].append(lasso)
        
        trn_df.loc[val_indx, 'ridge_preds'] = ridge.predict(X.loc[val_indx,:])
        trn_df.loc[val_indx, 'lasso_preds'] = lasso.predict(X.loc[val_indx,:])
    
    return models, trn_df

def cv_predict_lin(data, models):
    preds = np.zeros(len(data))
    for model in models:
        preds+=model.predict(data.drop(columns=['dept_id','id', 'd', 'TARGET'])) /len(models)
    return preds
    
def linear_predict(models, X, val_df):
    for name, _models in models.items():
        val_df[f'{name}_preds'] = cv_predict_lin(X, _models)
    return val_df

def train_lin(data, trn_days, val_days):
    data = to_onehot_data(data)
    train, val, trn_df, val_df = data_split_lin(data, trn_days, val_days)
    models, trn_df = linear_cv(train, trn_df)
    val_df = linear_predict(models, val, val_df)
    return val_df, trn_df

def train_lin_sub(data, for_predict):
    predict_day = data.d.unique()[-28:][for_predict-1]
    predict_sub_df = data[data.d==predict_day]
    data = to_onehot_data(data)
    data.dropna(0, inplace=True)
    data.reset_index(drop=True, inplace=True)
    predict_sub_df.reset_index(drop=True, inplace=True)

    trn_df = data[['id', 'd', 'TARGET']]
    val_df = predict_sub_df[['id', 'd', 'TARGET']]

    models, trn_df = linear_cv(data, trn_df)
    val_df = linear_predict(models, predict_sub_df, val_df)
    return val_df, trn_df