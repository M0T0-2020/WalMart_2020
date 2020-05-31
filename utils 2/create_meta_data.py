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

def create_is_sell_data(sell_prices_df, calendar_df, train_df):
    sell_prices_df['id'] = sell_prices_df['item_id'].astype('str')+'_'+sell_prices_df['store_id']+'_validation'
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

def set_index(df, name):
    d = {}
    for col, value in df.iloc[0,:].items():
        if type(col)==str:
            if type(df[col].values[0])!=str:
                v = 'd'
            else:
                v='id'
        else:
            v=name
        d[col]=v
    return d

def dcol2int(col):
    if col[:2]=='d_':
        return int(col.replace('d_', ''))
    else:
        return col
    
def create_event_data(train_df, calendar_df):
    new_df = pd.DataFrame()
    D_COLS = [d for d in train_df.columns if type(d)!=str]
    for event_name in ['event_name_1', 'event_name_2']:
        tmp_df = pd.concat([
            train_df.groupby(['dept_id'])[D_COLS].mean().T.astype(float),
            train_df.groupby(['cat_id'])[D_COLS].mean().T.astype(float),
            calendar_df.loc[D_COLS,event_name].replace(np.nan, 'NAN')
        ],axis=1)

        dept_id_cols = train_df.dept_id.unique().tolist()
        cat_id_cols = train_df.cat_id.unique().tolist()

        tmp_df = pd.concat([
            tmp_df[[event_name]],
            tmp_df.groupby([event_name])[dept_id_cols].transform(
            lambda x: x.shift(1).rolling(len(x), min_periods=1).mean()
            ),
            tmp_df.groupby([event_name])[cat_id_cols].transform(
            lambda x: x.shift(1).rolling(len(x), min_periods=1).mean()
            )
        ], axis=1)

        tmp_df[dept_id_cols] = tmp_df[dept_id_cols]/tmp_df[dept_id_cols].rolling(56, min_periods=1).mean().shift(1)
        tmp_df[cat_id_cols] = tmp_df[cat_id_cols]/tmp_df[cat_id_cols].rolling(56, min_periods=1).mean().shift(1)
        tmp_df.loc[tmp_df[event_name]=='NAN', dept_id_cols+cat_id_cols]=1
        
        tmp_df.columns=[f'{event_name}_{col}' for col in tmp_df.columns]
        
        new_df = pd.concat([
            new_df, tmp_df
        ] ,axis=1)
    new_df.index=D_COLS
    return new_df


def create_metadata(path, d_cols, submmit=True):
    train_df = pd.read_csv(path+'sales_train_validation.csv')
    calendar_df = pd.read_csv(path+'calendar.csv')
    sell_prices_df = pd.read_csv(path+'sell_prices.csv')
    #sample_submission_df = pd.read_csv(path+'sample_submission.csv')

    calendar_df['d'] = calendar_df.d.str.replace('d_', '').astype(int)
    cols = train_df.columns
    cols = [dcol2int(col) for col in cols]
    train_df.columns=cols
    calendar_df['date']=pd.to_datetime(calendar_df.date)
    calendar_df.index = calendar_df.d
    price_data, is_sell = create_is_sell_data(sell_prices_df, calendar_df, train_df)
    
    str_cols = [ col for col in train_df.columns if 'id' in str(col)]
    new_columns = str_cols+d_cols
    train_df = train_df.reindex(columns=new_columns)
    
    
    train_df = pd.concat([
        train_df[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']],
        train_df.loc[:,d_cols]+is_sell[d_cols].replace(0, np.nan).replace(1, 0)
    ], axis=1)
    train_df.index = train_df.id
    del is_sell;gc.collect()
    
    df = train_df.loc[:,d_cols].T.astype(float)
    a = df.loc[d_cols[28:-56]].rolling(28, min_periods=1).sum().replace(0,np.nan)+df.loc[d_cols[28:-56]][::-1].rolling(28, min_periods=1).sum()[::-1].replace(0,np.nan)
    a[a.notnull()]=0
    df.loc[d_cols[28:-56]] += a
    df = df.loc[d_cols,:].T.astype(float)
    del a;gc.collect()
    
    #snap_data
    snap_data = calendar_df[['snap_CA', 'snap_WI', 'snap_TX', 'd']]
    snap_data.set_index('d', inplace=True)
    
    #dept_id_price
    dept_id_price = price_data[d_cols]/price_data.groupby(['dept_id', 'store_id'])[d_cols].transform('mean')
    dept_id_price = dept_id_price.T.astype(float)
    #dept_id_price['d'] = dept_id_price.index
    dept_id_price = dept_id_price.replace(0,np.nan)
    
    #cat_id_price
    cat_id_price = price_data[d_cols]/price_data.groupby(['cat_id', 'store_id'])[d_cols].transform('mean')
    cat_id_price = cat_id_price.T.astype(float)
    #cat_id_price['d'] = cat_id_price.index
    cat_id_price = cat_id_price.replace(0,np.nan)
    
    #price_data
    price_data = price_data[d_cols].T
    price_data.replace(0,np.nan, inplace=True)
    #price_data['d']=price_data.index
    
    #event_df
    event_df = create_event_data(train_df, calendar_df)
    #event_df.reset_index(inplace=True)
    
    #calendar_dict
    calendar_dict = calendar_df[['wday', 'month']].to_dict()
    
    return train_df, snap_data, dept_id_price, cat_id_price, price_data, event_df, calendar_dict, df