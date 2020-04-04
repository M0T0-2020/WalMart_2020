def Preprocessing(train_df, calendar_df, sell_prices_df):
    sell_prices_df['id'] = sell_prices_df['item_id'].astype('str')+'_'+sell_prices_df['store_id']+'_validation'
    d_cols = [f'd_{i}' for i in range(1,1914)]
    
    event_type_1 = pd.get_dummies(calendar_df.event_type_1)
    event_type_1.columns = [f'{col}_event_type_1' for col in event_type_1.columns]
    event_type_2 = pd.get_dummies(calendar_df.event_type_1)
    event_type_2.columns = [f'{col}_event_type_2' for col in event_type_2.columns]
    calendar_data = pd.concat([
        calendar_df.drop(columns=['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'])[['wday', 'd','month','snap_CA', 'snap_TX', 'snap_WI']],
        event_type_1,
        event_type_2
    ], axis=1)
    calendar_data = calendar_data.set_index('d').T
    
    
    
    sell_prices_data = sell_prices_df[sell_prices_df.wm_yr_wk.isin(calendar_df.wm_yr_wk.unique())]
    sell_prices_data.reset_index(drop=True, inplace=True)
    tmp = sell_prices_data.groupby(['id'])[['wm_yr_wk', 'sell_price']].apply(lambda x: x.set_index('wm_yr_wk')['sell_price'].to_dict()).to_dict()
    d = calendar_df.d
    wm_yr_wk = calendar_df.wm_yr_wk
    price_data = {}
    for col in tqdm(train_df.id.unique()):
        price_data[col] = wm_yr_wk.map(tmp[col])
    price_data = pd.DataFrame(price_data)
    price_data.index = d
    
    
    is_sell = price_data.notnull().astype(float).T
    price_data = price_data.fillna(0)
    
    train_df = train_df.T
    train_df.columns = train_df.loc['id', :].values
    train_df = train_df.T
    
    return train_df, calendar_df, calendar_data, price_data, is_sell


def make_calendar_data(calendar_data, train_cols):
    calendar_index = [
        'wday', 'month',
        'Cultural_event_type_1', 'National_event_type_1', 'Religious_event_type_1', 'Sporting_event_type_1',
        'Cultural_event_type_2', 'National_event_type_2', 'Religious_event_type_2', 'Sporting_event_type_2'
    ]
    calendar = calendar_data.loc[calendar_index,:]
    event_index = [
        'Cultural_event_type_1', 'National_event_type_1', 'Religious_event_type_1', 'Sporting_event_type_1',
        'Cultural_event_type_2', 'National_event_type_2', 'Religious_event_type_2', 'Sporting_event_type_2'
    ]
    for shift in [3, 7, 14, 28]:
        tmp_calendar = calendar.loc[event_index, :]
        tmp_calendar = tmp_calendar.T.shift(-shift).T
        tmp_calendar.index = [f'{col}_shift{shift}' for col in tmp_calendar.index]
        calendar = pd.concat([
            calendar,
            tmp_calendar
        ], axis=0)
    calendar = calendar[train_cols]
    calendar = torch.FloatTensor(calendar.values.astype(float))
    return calendar