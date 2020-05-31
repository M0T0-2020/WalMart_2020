import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from time import time
import gc
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, log_loss
from sklearn.linear_model import Ridge,Lasso
from sklearn.preprocessing import minmax_scale


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


def plot_importance(models, col):
    importances = np.zeros(len(col))
    for model in models:
        importances+=model.feature_importance(importance_type='gain')
    importance = pd.DataFrame()
    importance['importance'] = importances
    importance['importance'] = minmax_scale(importance.importance)
    importance['col'] = col
    importance.to_csv(f'importance.csv',index=False)
    
def run_cv(x_train, y_train, trn_df, params=PARAMS):
    models = []
    k = GroupKFold(n_splits=5)
    trn_df['y_pred'] = 0
    
    for trn_indx, val_indx in k.split(x_train[['dept_id']],groups=x_train['dept_id']):
        train_set = lgb.Dataset(x_train.loc[trn_indx,:], y_train.loc[trn_indx])
        val_set = lgb.Dataset(x_train.loc[val_indx,:], y_train.loc[val_indx])
        
        categories = ['cat_id', 'dept_id', 'store_id']
        
        model = lgb.train(
            train_set=train_set, 
            valid_sets=[train_set, val_set],
            params=params, num_boost_round=3000, early_stopping_rounds=100, verbose_eval=500,
            categorical_feature=categories+['wday', 'month']
        )
        
        models.append(model)
        trn_df.loc[val_indx, 'y_pred']=np.e**(model.predict(x_train.loc[val_indx,:]))-1
        gc.collect()
        
    return models, trn_df

def predict_cv(x_val, models):
    preds = np.zeros(len(x_val))
    for model in models:
        preds+=model.predict(x_val)/len(models)
    return preds

def show_eval_score(preds, val_df):
    preds = np.e**(preds)-1
    val_df['y_pred'] = preds
    score= np.sqrt(mean_squared_error(val_df['TARGET'], preds))
    print("EVALUATION SCORE : ", score)
    return val_df

def split_data(data, trn_day, val_day):
    data = data[data.shift_4.notnull()]
    
    y = data[['d', 'id', 'TARGET']]
    X = data.drop(columns=['id',  'TARGET','state_id']).astype(float)
    
    x_train, x_val = X[X.d.isin(trn_day)], X[X.d.isin(val_day)]
    y_train, y_val = y[y.d.isin(trn_day)], y[y.d.isin(val_day)]
    
    x_train.reset_index(drop=True,inplace=True)
    x_val.reset_index(drop=True,inplace=True)
    y_train.reset_index(drop=True,inplace=True)
    y_val.reset_index(drop=True,inplace=True)
    trn_df = y_train[['id', 'd', 'TARGET']]
    val_df = y_val[['id', 'd', 'TARGET']]
    y_train['TARGET'] = np.log1p(y_train['TARGET'])
    
    x_train.drop('d', axis=1, inplace=True)
    x_val.drop('d', axis=1, inplace=True)
    y_train = y_train['TARGET'].astype(float)
    return x_train, x_val, y_train, trn_df, val_df

def train(data):
    split=28
    data = data[data.TARGET.notnull()]
    d_cols = sorted(data.d.unique())
    trn_day = d_cols[:-split]
    val_day = d_cols[-split:]

    x_train, x_val, y_train, trn_df, val_df = split_data(data, trn_day, val_day)
    print(x_train.shape, x_val.shape)
    models, trn_df = run_cv(x_train, y_train, trn_df)
    preds = predict_cv(x_val, models)
    val_df = show_eval_score(preds, val_df)
    plot_importance(models, x_train.columns)
    return models, val_df, trn_df

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

def train_sub_predict(data, for_predict):
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