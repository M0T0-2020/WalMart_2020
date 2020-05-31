import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
import datetime
from catboost import CatBoostClassifier
import lightgbm as lgb
from time import time
from tqdm import tqdm
from collections import Counter
from scipy import stats
import gc, pickle
import ast

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold,TimeSeriesSplit, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, log_loss
from sklearn.linear_model import Ridge,Lasso, BayesianRidge
from sklearn.svm import LinearSVR
from sklearn.preprocessing import minmax_scale


from preprocessnig import create_sale_feature

PARAMS = {
    'boosting_type': 'gbdt',
    'objective': 'tweedie',
    'tweedie_variance_power': 1.1,
    'metric': 'rmse',
    'subsample': 0.5,
    'subsample_freq': 1,
    'learning_rate': 0.03,
    'num_leaves': 2**11-1,
    'min_data_in_leaf': 2**12-1,
    'feature_fraction': 0.5,
    'max_bin': 100,
    'n_estimators': 1400,
    'boost_from_average': False,
    'verbose': 1
    } 


def plot_importance(models, col, name):
    importances = np.zeros(len(col))
    for model in models:
        importances+=model.feature_importance(importance_type='gain')
    importance = pd.DataFrame()
    importance[f'importance_{name}'] = importances
    importance[f'importance_{name}'] = minmax_scale(importance.importance)
    importance['col'] = col
    importance.to_csv(f'importance.csv',index=False)

def predict_cv(x_val, models):
    preds = np.zeros(len(x_val))
    for model in models:
        preds+=model.predict(x_val)/len(models)
    return preds

def split_data_for_sub(data):
    data = data[data.TARGET.notnull()]
    data = data[data.shift_4.notnull()]
    data = data[data.diff_std_7.notnull()]
    trn_df = data[['id', 'd', 'TARGET']]
    y = np.log1p(data['TARGET']).astype(float)
    X = data.drop(columns=['id','d', 'TARGET','state_id']).astype(float)
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    trn_df.reset_index(drop=True, inplace=True)
    return X, y, trn_df

def run_cv_for_sub(X, y, trn_df, params=PARAMS):
    models = []
    k = GroupKFold(n_splits=5)
    trn_df['y_pred'] = 0
    
    for trn_indx, val_indx in k.split(X[['dept_id']],groups=X['dept_id']):
        train_set = lgb.Dataset(X.loc[trn_indx,:], y.loc[trn_indx])
        val_set = lgb.Dataset(X.loc[val_indx,:], y.loc[val_indx])
        
        categories = ['cat_id', 'dept_id', 'store_id']
        
        model = lgb.train(
            train_set=train_set, 
            valid_sets=[train_set, val_set],
            params=params, num_boost_round=3000, early_stopping_rounds=100, verbose_eval=500,
            categorical_feature=categories+['wday', 'month']
        )
        
        models.append(model)
        trn_df.loc[val_indx, 'y_pred']=np.e**(model.predict(X.loc[val_indx,:]))-1
        gc.collect()
        
    return models, trn_df

def train_sub_predict_direct(data, for_predict):
    train_d_cols = data.d.unique().tolist()
    predict_day=train_d_cols[-28:][for_predict-1]
    sub_predict_data = data[data.d==predict_day]
    X, y, trn_df = split_data_for_sub(data)
    print(X.shape)
    models, trn_df = run_cv_for_sub(X, y, trn_df)
    preds = predict_cv(sub_predict_data[X.columns], models)
    
    sub_df = sub_predict_data[['id', 'd', 'TARGET']]
    sub_df[f'y_pred'] = preds
    return trn_df, sub_df

def train_sub_predict_recursive(data, params=PARAMS):
    data.reset_index(drop=True, inplace=True)
    trn_df = data[['id', 'd', 'TARGET']]
    trn_df['y_pred'] = np.nan
    models = []
    k = GroupKFold(n_splits=5)
    categories = ['cat_id', 'dept_id', 'store_id']

    y = data['TARGET']
    data = data.drop('TARGET', axis=1)
    cols = data.columns.tolist()

    for trn_indx, val_indx in k.split(data[['dept_id']],groups=data['dept_id']):
        train_set = lgb.Dataset(data.loc[trn_indx,:], y.loc[trn_indx])
        val_set = lgb.Dataset(data.loc[val_indx,:], y.loc[val_indx])

        
        model = lgb.train(
            train_set=train_set, 
            valid_sets=[train_set, val_set],
            params=params, num_boost_round=3000, early_stopping_rounds=100, verbose_eval=500,
            categorical_feature=categories+['wday']
        )
        
        models.append(model)
        trn_df.loc[val_indx, 'y_pred']=np.e**(model.predict(data.loc[val_indx,:]))-1
        gc.collect()
    return models, trn_df, cols

def predict_recursive(data, predict_d, cols, del_cols, models):
    input_X = data[data.d==predict_d]
    input_X = input_X[cols]
    preds = predict_cv(input_X, models)
    data.loc[data.d==predict_d,'TARGET'] = np.e**(preds) -1
    data = data.drop(columns=del_cols)
    for _ in range(1,28):
        data = create_sale_feature(data)
        predict_d += 1
        input_X = data[data.d==predict_d]
        input_X = input_X[cols]
        preds = predict_cv(input_X, models)
        data.loc[data.d==predict_d,'TARGET'] = np.e**(preds) -1
        data = data.drop(columns=del_cols)
    return data