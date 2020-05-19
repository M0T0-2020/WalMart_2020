import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
import gc, pickle

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def set_index(df, name):
    d = {}
    for col in df.iloc[0,:].keys():
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

def make_roll_data(data, win):
    data_2 = data.groupby(['id'])['TARGET'].apply(
        lambda x:
        x.shift(1).rolling(win, min_periods=1).agg({'mean', 'std'})
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
    diff_data.columns=[f'diff_{col}_1' for col in diff_data.columns]
    data = pd.concat([
        data, diff_data
    ], axis=1)
    
    diff_data = data.groupby(['id'])['TARGET'].apply(
        lambda x:
        abs(x.shift(1).diff(7)).rolling(win, min_periods=1).agg({'mean', 'std'})
    ) 
    diff_data.columns=[f'diff_{col}_7' for col in diff_data.columns]
    data = pd.concat([
        data, diff_data
    ], axis=1)
    return data

def make_shift_data(data):
    shift=7
    
    data[f'shift_1'] = data.groupby(['id'])['TARGET'].apply(
        lambda x:
        x.shift(shift)+x.shift(shift+7)
    )
    data[f'shift_2'] = data[f'shift_1']+data.groupby(['id'])['TARGET'].apply(
        lambda x:
        x.shift(shift+14)+x.shift(shift+28)
    )
    return data

# 1~1941
D_COLS = [i+1 for i in range(1941)]
def preprocessing(path, test):
    if test:
        train_df = pd.read_csv(path+'train_df_short.csv')
    else:
        train_df = pd.read_csv(path+'train_df.csv')
    train_df.columns= [int(col) if col.isnumeric() else str(col) for col in train_df.columns]
    
    train_df.index=train_df.id
    data = pd.concat([
        train_df[D_COLS[-600:]].isnull().sum(axis=1),
        train_df[D_COLS].mean(1)
    ],axis=1)
    data.columns=['null_num_600', 'sell_mean']
    data['sell_mean_null_600'] = data['sell_mean']/data['null_num_600']
    ids = data.sort_values('sell_mean_null_600', ascending=False).index.tolist()
    gc.collect()
    
    
    data = pd.concat([
        train_df[train_df.id.isin(ids[5000:])][D_COLS[-60:]].stack(dropna=False).reset_index(),
        train_df[train_df.id.isin(ids[:5000])][D_COLS[-370:]].stack(dropna=False).reset_index()
        ], axis=0)
    data = data.rename(columns=set_index(data, 'TARGET'))
    data.sort_values('d', inplace=True)
    data.reset_index(drop=True, inplace=True)
    data = reduce_mem_usage(data)
    gc.collect()


    for key, value in train_df[['dept_id', 'cat_id', 'state_id', 'store_id']].to_dict().items():
        data[key] = data.id.map(value)

    snap_data = pd.read_csv(path+'snap_data.csv')
    snap_data.index=snap_data.d
    snap_data.drop('d',axis=1, inplace=True)
    
    data[f'snap']=0
    for key, value in snap_data.to_dict().items():
        k = key.replace('snap_', '')
        data.loc[data.state_id==k,'snap'] = data.loc[data.state_id==k, 'd'].map(value)
    for shift in [-3,-2,-1,1,2,3]:
        data[f'snap_{shift}'] = data.groupby(['id'])['snap'].shift(shift)


    dept_id_price = pd.read_csv(path+'dept_id_price.csv')
    cat_id_price = pd.read_csv(path+'cat_id_price.csv')

    dept_id_price.index=dept_id_price.d
    cat_id_price.index=cat_id_price.d

    dept_id_price = dept_id_price[dept_id_price.d.isin(data.d.unique())].drop('d', axis=1)
    cat_id_price = cat_id_price[cat_id_price.d.isin(data.d.unique())].drop('d', axis=1)

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

    price_df = pd.read_csv(path+'price_data.csv')
    price_df.index=price_df.d
    price_df = price_df[price_df.d.isin(data.d.unique())].drop('d', axis=1)
    price_df = price_df.stack(dropna=False).reset_index()
    price_df.rename(columns=set_index(price_df, 'price'), inplace=True)
    data = pd.merge(
        data, price_df, on=['d', 'id'], how='left'
    )
    del price_df;gc.collect()

    with open(path+'calendar_dict.pkl', 'rb') as f:
        calendar_dict = pickle.load(f)
        for key, value in calendar_dict.items():
            data[key] = data.d.map(value)
    del calendar_dict;gc.collect()

    event_df = pd.read_csv(path+'event_df.csv')
    event_df.index=event_df['index']
    event_df.drop('index', axis=1, inplace=True)

    tmp_dic = event_df.to_dict()
    data[f'dept_id_event_name_1']=1
    data[f'dept_id_event_name_2']=1
    data[f'cat_id_event_name_1']=1
    data[f'cat_id_event_name_2']=1
    for key, value in tmp_dic.items():
        if key[13:] in train_df.dept_id.unique().tolist():
            data.loc[data.dept_id==key, f'dept_id_{key[:12]}']=data.loc[data.dept_id==key, 'd'].map(value).fillna(1)
        if key[13:] in train_df.cat_id.unique().tolist():
            data.loc[data.cat_id==key, f'cat_id_{key[:12]}']=data.loc[data.cat_id==key, 'd'].map(value).fillna(1)
    for shift in [-7,-4,-3,-2,-1,1,2]:
        for event_name in ['dept_id_event_name_1', 'dept_id_event_name_2', 'cat_id_event_name_1', 'cat_id_event_name_2']:
            data[f'{event_name}_shift{shift}'] = data.groupby(['id'])[event_name].shift(shift).fillna(1)

    cols = data.columns.tolist()
    print(cols)

    data = make_roll_data(data=data, win=28)
    data = make_roll_data(data=data, win=7)
    data = make_diff_data(data=data, win=28)
    data = make_shift_data(data=data)
    gc.collect()

    print([col for col in data.columns if not col in cols])
    
    return data

def shift_seven(data):
    data[['shift_1', 'shift_2']] = data.groupby(['id'])[['shift_1', 'shift_2']].shift(7)
    return data

def shift_one(data):
    data[
        ['roll_28_std', 'roll_28_mean', 'diff_std_1', 'diff_mean_1', 'diff_std_7', 'diff_mean_7']
        ]=data.groupby(['id'])[
            ['roll_28_std', 'roll_28_mean', 'diff_std_1', 'diff_mean_1', 'diff_std_7', 'diff_mean_7']
            ].shift(1)
    return data