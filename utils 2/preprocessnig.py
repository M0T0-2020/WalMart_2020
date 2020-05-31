import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
import gc, pickle

def make_roll_data(data, win, agg={'mean', 'std'}):
    data_2 = data.groupby(['id'])['TARGET'].apply(
        lambda x:
        x.shift(1).rolling(win, min_periods=1).agg(agg)
    )
    data_2.columns=[f'roll_{win}_{col}' for col in data_2.columns]
    data = pd.concat([
        data, data_2
    ], axis=1)
    return data

def make_diff_data(data, win):
    diff_data = data.groupby(['id'])['TARGET'].apply(
        lambda x:
        abs(x.shift(1).diff(1)).rolling(win, min_periods=1).agg({'mean', 'std'})
    ) 
    diff_data.columns=[f'diff_{col}_{win}_1' for col in diff_data.columns]
    data = pd.concat([
        data, diff_data
    ], axis=1)
    
    diff_data = data.groupby(['id'])['TARGET'].apply(
        lambda x:
        abs(x.shift(1).diff(7)).rolling(win, min_periods=1).agg({'mean', 'std'})
    ) 
    diff_data.columns=[f'diff_{col}_{win}_7' for col in diff_data.columns]
    data = pd.concat([
        data, diff_data
    ], axis=1)
    return data

def make_shift_data(data):
    shift=7
    for i, p in  enumerate([0,7]):
        data[f'shift_{i+1}'] = data.groupby(['id'])['TARGET'].shift(shift+p)
    data['shift_3'] = data[['shift_1', 'shift_2']].mean(1)
    return data

def preprocessing(path,d_cols,test):
    train_df, snap_data, dept_id_price, cat_id_price, price_data, event_df, calendar_dict, df = create_metadata(path, d_cols)
    if test:
        train_df = train_df[train_df.id.isin(train_df.id.unique()[:2000])]
    
    data = train_df[d_cols[-360:]].stack(dropna=False).reset_index()
    data = data.rename(columns=set_index(data, 'TARGET'))
    data.sort_values('d', inplace=True)
    data.reset_index(drop=True, inplace=True)
    data = reduce_mem_usage(data)
    gc.collect()


    for key, value in train_df[['dept_id', 'cat_id', 'state_id', 'store_id']].to_dict().items():
        data[key] = data.id.map(value)
    
    data[f'snap']=0
    for key, value in snap_data.to_dict().items():
        k = key.replace('snap_', '')
        data.loc[data.state_id==k,'snap'] = data.loc[data.state_id==k, 'd'].map(value).fillna(0)
    for shift in [-3,-2,-1,1,2,3]:
        data[f'snap_{shift}'] = data.groupby(['id'])['snap'].shift(shift).fillna(0)


    dept_id_price = dept_id_price.stack(dropna=False).reset_index()
    cat_id_price = cat_id_price.stack(dropna=False).reset_index()

    dept_id_price.rename(columns=set_index(dept_id_price, 'dept_id_price'), inplace=True)
    cat_id_price.rename(columns=set_index(cat_id_price, 'cat_id_price'), inplace=True)

    data = pd.merge(
        data, dept_id_price, on=['d', 'id'], how='left'
    )
    data = pd.merge(
        data, cat_id_price, on=['d', 'id'], how='left'
    )


    del dept_id_price,cat_id_price;gc.collect()

    price_data = price_data.stack(dropna=False).reset_index()
    price_data.rename(columns=set_index(price_data, 'price'), inplace=True)
    data = pd.merge(
        data, price_data, on=['d', 'id'], how='left'
    )
    del price_data;gc.collect()

    data['wday'] = data.d.map(calendar_dict['wday'])
    del calendar_dict;gc.collect()


    tmp_dic = event_df.to_dict()
    data[f'dept_id_event_name_1']=1
    data[f'cat_id_event_name_1']=1
    for key, value in tmp_dic.items():
        if 'event_name_1' in key:
            if key[13:] in train_df.dept_id.unique().tolist():
                data.loc[data.dept_id==key[13:], f'dept_id_{key[:12]}']=data.loc[data.dept_id==key[13:], 'd'].map(value).fillna(1)
            if key[13:] in train_df.cat_id.unique().tolist():
                data.loc[data.cat_id==key[13:], f'cat_id_{key[:12]}']=data.loc[data.cat_id==key[13:], 'd'].map(value).fillna(1)
    for shift in [-3,-2,-1,1,2,3]:
        for event_name in ['dept_id_event_name_1', 'cat_id_event_name_1']:
            data[f'{event_name}_shift{shift}'] = data.groupby(['id'])[event_name].shift(shift).fillna(1)

    categories = [c for c in data.columns if data[c].dtype==object]
    print(categories)
    for c in categories:
        if c=='id':
            pass
        else:
            data[c] = pd.factorize(data[c])[0]
    
    return data

def create_sale_feature(data):
    cols = data.columns.tolist()
    
    data = make_roll_data(data=data,win=28,agg={'mean', 'std', 'skew'})
    data = make_roll_data(data=data,win=7,agg={'mean', 'min', 'max'})
    data = make_roll_data(data,win=56,agg={'std', 'skew'})
    data = make_diff_data(data=data, win=28)
    data = make_diff_data(data=data, win=7)
    data = make_shift_data(data=data)
    
    print([col for col in data.columns if not col in cols])

    return data