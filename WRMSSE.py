class WRMSSE(nn.Module):
    def __init__(self, df, calendar, prices):
        super(WRMSSE, self).__init__()
        self.df = df
        self.df['all_id'] = 0  # for lv1 aggregation
        self.df['index'] = self.df.index
        self.id_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'all_id', 'index']
        self.calendar = calendar
        self.prices = prices
        
        self.group_ids = (
            ['all_id'],
            ['state_id'],
            ['store_id'],
            ['cat_id'],
            ['dept_id'],
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            ['item_id'],
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
        )
        
        
    def prepare_metrics(self, valid_d_cols):
        self.train_d_cols = [f'd_{i}' for i in range(1,1914) if f'd_{i}' in valid_d_cols]  #<-- コンペが last １ヶ月になったら 914 -->1942 変える
        #self.train_d_cols = [f'd_{i}' for i in range(1000,1914) if f'd_{i}' in valid_d_cols]  #<-- コンペが last １ヶ月になったら 914 -->1942 変える
        self.valid_d_cols = valid_d_cols
        self.weight_columns = self.train_d_cols[-28:]
        self.split_train_valid_data()
        self.get_weight()
        
        self.tensor_index = {}
        self.index_len = {}
        self.denominator = {}
        self.True_y = {}
        self.weight = {}
        for i, group_id in enumerate(tqdm(self.group_ids)):
            #  index dict
            self.tensor_index.update(self.df[['index']+group_id].groupby(group_id)['index'].unique().to_dict())
            self.index_len.update(self.df[['index']+group_id].groupby(group_id)['index'].nunique().to_dict())
            
            # denominator
            tmp_a = self.train_df.groupby(group_id)[self.train_d_cols].sum().T.to_dict()
            a = {}
            for key, value in tmp_a.items() :
                value = np.array(list(value.values()))
                value = ((value[1:]-value[:-1])**2).mean()
                if value<=0:
                    value=1
                a[key] = value
            self.denominator.update(a)
            
            # True y
            tmp_a = self.valid_df.groupby(group_id)[self.valid_d_cols].sum().T.to_dict()
            a={}
            for key, value in tmp_a.items():
                value = list(value.values())
                a[key] = np.array(value)
            self.True_y.update(a)
            
            #weight 
            lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1)
            self.weight.update((lv_weight / lv_weight.sum()).to_dict())
            
        self.tensor_index = np.array(list(self.tensor_index.values()))
        self.index_len = torch.FloatTensor(list(self.index_len.values()))
        self.denominator = torch.FloatTensor(list(self.denominator.values()))
        self.True_y = torch.FloatTensor(list(self.True_y.values()))
        self.weight = torch.FloatTensor(list(self.weight.values()))
        
        
    def split_train_valid_data(self):
        self.train_df = self.df[self.id_columns+self.train_d_cols]
        self.valid_df = self.df[self.id_columns+self.valid_d_cols]
        
    def get_weight(self):
        day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict()
        weight_df = self.train_df[['item_id', 'store_id'] + self.weight_columns].set_index(['item_id', 'store_id'])
        weight_df = weight_df.stack().reset_index().rename(columns={'level_2': 'd', 0: 'value'})
        weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)

        weight_df = weight_df.merge(self.prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
        weight_df['value'] = weight_df['value'] * weight_df['sell_price']
        weight_df = weight_df.set_index(['item_id', 'store_id', 'd']).unstack(level=2)['value']
        weight_df = weight_df.loc[zip(self.train_df.item_id, self.train_df.store_id), :].reset_index(drop=True)
        self.weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis=1, sort=False)
    
    
    def init_Loss(self, batch_size):
        self.batch_size = batch_size
        self.Loss = torch.zeros(42840, 28)
    
    def accululate_loss(self, pred, iter_num):
        zeros = torch.zeros(42840, self.batch_size)
        for i in tqdm(range(len(self.index_len))):
            lv_len = self.index_len[i]
            lv_tensor_index = self.tensor_index[i]
            index = lv_tensor_index[lv_tensor_index<self.batch_size*(iter_num+1)]
            index -=self.batch_size*iter_num
            index = index[index>=0]
            zeros[i, index] = 1/lv_len
        
        zeros = torch.mm(zeros, pred)
        self.Loss+=zeros
            
    
    def forward(self):
        self.Loss = torch.sqrt(((self.Loss-self.True_y)**2).mean(1))
        self.Loss = self.Loss*(1/self.denominator)*self.weight
        self.Loss = self.Loss.sum()
        return self.Loss