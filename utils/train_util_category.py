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


def split_data_for_sub(data):
    data = data[data.TARGET.notnull()]
    data = data[data.shift_4.notnull()]
    data = data[data.diff_std_7.notnull()]
    trn_df = data[['id', 'd', 'TARGET']]
    y = np.log1p(data['TARGET']).astype(float)
    X = data.drop(columns=['id','d', 'TARGET','state_id']).astype(float)
    X.reset_index(drop=True, inplace=True)
    trn_df.reset_index(drop=True, inplace=True)
    return X, trn_df

def category_cv(X, trn_df, val_X, val_df, category):
    trn_df[f'{category}_pred'] = 0
    val_df[f'{category}_pred'] = 0
    category_cols = ['cat_id', 'dept_id', 'store_id']
    category_cols = [col for col in category_cols if col!=category]

    for cat in val_X[category].unique():
        tmp_train_X = X[X[category]==cat].drop(category, axis=1)
        tmp_val_X = val_X[val_X[category]==cat].drop(category, axis=1)
        tmp_train_X.reset_index(inplace=True, drop=True)
        tmp_val_X.reset_index(inplace=True, drop=True)

        k = GroupKFold(n_splits=5)
        for trn_indx, val_indx in k.split(X[['dept_id']], groups=X['dept_id']):
            train_set = lgb.Dataset(
                tmp_train_X.drop('TARGET', axis=1).loc[trn_indx,:],
                tmp_train_X['TARGET'].loc[trn_indx]
                )
            val_set = lgb.Dataset(
                tmp_train_X.drop('TARGET', axis=1).loc[val_indx,:],
                tmp_train_X['TARGET'].loc[val_indx]
                 )

            model = lgb.train(
                train_set=train_set, 
                valid_sets=[train_set, val_set],
                params=params, num_boost_round=3000, early_stopping_rounds=100, verbose_eval=500,
                categorical_feature=categories+['wday']
                )

            val_df[] = np.e**(model.predict(tmp_val_X.drop('TARGET', axis=1)))-1

        


