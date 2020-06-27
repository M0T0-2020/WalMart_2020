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
from sklearn.cluster import KMeans

import optuna

def create_is_sell_data(sell_prices_df, calendar_df, train_df):
    train_df.index = train_df['id']
    sell_prices_df['id'] = sell_prices_df['item_id'].astype('str')+'_'+sell_prices_df['store_id']+'_evaluation'
    sell_prices_data = sell_prices_df[sell_prices_df.wm_yr_wk.isin(calendar_df.wm_yr_wk.unique())]
    sell_prices_data.reset_index(drop=True, inplace=True)
    tmp = sell_prices_data.groupby(['id'])[['wm_yr_wk', 'sell_price']].apply(
        lambda x: x.set_index('wm_yr_wk')['sell_price'].to_dict()
    ).to_dict()
    d = calendar_df.d
    wm_yr_wk = calendar_df.wm_yr_wk
    price_data = {}
    for col in tqdm(train_df.id.unique()):
        price_data[col] = wm_yr_wk.map(tmp[col])
    price_data = pd.DataFrame(price_data)
    price_data.index = d
    is_sell = price_data.notnull().astype(float).T
    price_data = price_data.fillna(0)
    
    is_sell.index=train_df.id
    train_df.index=train_df.id
    is_sell = pd.concat([
        train_df[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']], is_sell
    ], axis=1)
    price_data = pd.concat([
        train_df[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']], price_data.T  
    ], axis=1)
    
    return price_data, is_sell

def sort_d_cols(d_cols):
    d_cols = [int(d.replace('d_','')) for d in d_cols]
    d_cols = sorted(d_cols)
    d_cols = [f'd_{d}' for d in d_cols]
    return d_cols

def select_near_event(x, event_name):
    z = ''
    for y in x:
        if y in event_name:
            z+=y+'_'
    if len(z)==0:
        return np.nan
    else:
        return z
    
def calendar_preprocessing(calendar):
    calendar['qaurter'] = pd.to_datetime(calendar['date']).dt.day.apply(lambda x: x//7)

    event_name = ['SuperBowl', 'ValentinesDay', 'PresidentsDay', 'LentStart', 'LentWeek2', 'StPatricksDay', 'Purim End', 
                'OrthodoxEaster', 'Pesach End', 'Cinco De Mayo', "Mother's day", 'MemorialDay', 'NBAFinalsStart', 'NBAFinalsEnd',
                "Father's day", 'IndependenceDay', 'Ramadan starts', 'Eid al-Fitr', 'LaborDay', 'ColumbusDay', 'Halloween', 
                'EidAlAdha', 'VeteransDay', 'Thanksgiving', 'Christmas', 'Chanukah End', 'NewYear', 'OrthodoxChristmas', 
                'MartinLutherKingDay', 'Easter']
    event_type = ['Sporting', 'Cultural', 'National', 'Religious']
    event_names = {'event_name_1':event_name, 'event_type_1':event_type}
    for event, event_name in event_names.items():
        calendar[f'new_{event}']=''
        for i in range(-1,-8,-1):
            calendar[f'new_{event}'] += calendar[event].shift(i).astype(str)+'|'
        calendar[f'new_{event}'] = calendar[f'new_{event}'].apply(lambda x: x.split('|'))
        calendar[f'new_{event}'] = calendar[f'new_{event}'].apply(lambda x: select_near_event(x, event_name))
    return calendar


class GroupPredict:
    def __init__(self, path):
        train = pd.read_csv(path+'sales_train_evaluation.csv')
        calendar = pd.read_csv(path+'calendar.csv')
        price = pd.read_csv(path+'sell_prices.csv')
        self.price_data, self.is_sell = create_is_sell_data(price, calendar, train)

        self.d_cols = [d for d in train.columns if 'd_' in d]
        train = train.reindex(
            columns=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']+self.d_cols
        )

        train = train.set_index('id', drop=False)
        self.train = pd.concat([
            train[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']],
            train[self.d_cols]*self.price_data[self.d_cols]
        ], axis=1)
        self.calendar = calendar_preprocessing(calendar)