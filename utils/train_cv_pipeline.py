import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from time import time
import gc, pickle
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, log_loss
from sklearn.linear_model import Ridge,Lasso

try:
    from lin_model import *
    from train_utils import *
except:
    pass

def train_cv_pipeline(data):
    models, val_df, trn_df = train(data)
    trn_days, val_days = trn_df.d.unique().tolist(), val_df.d.unique().tolist()
    val_df_lin, trn_df_lin = train_lin(data, trn_days, val_days)

    return val_df, trn_df, val_df_lin, trn_df_lin

def predict_sub_pipeline(data, for_predict):
    trn_df, sub_df = train_sub_presict(data, for_predict)
    sub_df_lin, trn_df_lin = train_lin_sub(data, for_predict)

    return sub_df,trn_df, sub_df_lin, trn_df_lin