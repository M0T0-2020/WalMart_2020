


class indicate_index(torch.utils.data.Dataset):
    def __init__(self, index):
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return idx
    
def _create_batch_data(index, data, calendar):
    _data = data[index, :, :]
    x = torch.tensor([])
    for tmp_x in _data:
        tmp_x = torch.cat((tmp_x, calendar),dim=0)
        x = torch.cat((x,tmp_x.unsqueeze(0)), dim=0)
    return x

class Loss_func_groupid(nn.Module):
    def __init__(self, df, cols, index_index, group_id):
        super(Loss_func_item_id_state_id_, self).__init__()
        
        self.index_index = index_index
        last_d = int(cols[-1].replace('d_', ''))
        d_cols = df.columns[df.columns.str.startswith('d_')]
        train_d_cols = last_d-28*2
        self.train_d_cols = d_cols[:train_d_cols]
        test_d_cols = last_d-28
        self.test_d_cols = d_cols[:test_d_cols]
        self._create_denominator(df, group_id)
        
    def _create_denominator(self, df, group_id):
        g_df = df.groupby(group_id)#[d_cols].sum()
        
        train_value = g_df[self.train_d_cols].sum()
        train_value = train_value.loc[self.index_index,:]
        train_value = train_value.values
        train_value = train_value[:,1:]-train_value[:,:-1]
        train_value = train_value**2
        train_value = train_value.mean(1)
        train_value[train_value==0]=1
        self.train_value = torch.FloatTensor(train_value)
        
        test_value = g_df[self.test_d_cols].sum()
        test_value = test_value.loc[self.index_index,:]
        test_value = test_value.values
        test_value = test_value[:,1:]-test_value[:,:-1]
        test_value = test_value**2
        test_value = test_value.mean(1)
        test_value[test_value==0]=1
        self.test_value = torch.FloatTensor(test_value)
        
    def forward(self, preds, true, idx, length, train):
        a1=0
        a2=0
        Loss=0
        for i, _len in enumerate(length):
            _idx = idx[i]
            a2=a1+_len
            _preds = preds[a1:a2].sum(0)
            _true = true[a1:a2].sum(0)
            loss = (_preds -_true)**2
            loss = loss.mean()
            loss = loss.squeeze()
            if train:
                loss = loss/self.train_value[_idx]
            else:
                loss = loss/self.test_value[_idx]
            loss = torch.sqrt(loss)
            Loss+=loss/len(length)
            a1=a2
        return Loss

class TrainModel_groupId():
    def __init__(self, path, group_id):
        self.group_id = group_id
        self.path=path
        self.df = pd.read_csv(self.path+'sales_train_validation.csv')
        self.calendar_df = pd.read_csv(self.path+'calendar.csv')
        self.sell_prices_df = pd.read_csv(self.path+'sell_prices.csv')
        self.sample_submission_df = pd.read_csv(self.path+'sample_submission.csv')
        
        self.d_cols = self.df.columns[self.df.columns.str.startswith('d_')].values.tolist()
        
        self.train_df, self.calendar_df, self.calendar_data, self.price_data, self.is_sell = Preprocessing(df, calendar_df, sell_prices_df)
        
        
    def make_data_loader(self, cols):
        self.cols = cols
        state='CA'
        data_ca = self.make_data_group_id(
            cols, state, self.train_df, self.calendar_data, self.price_data, 
            self.is_sell, self.sample_submission_df, self.group_id
        )
        state='TX'
        data_tx = self.make_data_group_id(
            cols, state, self.train_df, self.calendar_data, self.price_data,
            self.is_sell, self.sample_submission_df, self.group_id
        )
        state='WI'
        data_wi = self.make_data_group_id(
            cols, state, self.train_df, self.calendar_data, self.price_data,
            self.is_sell, self.sample_submission_df, self.group_id
        )


        data = torch.cat(
            (data_ca, data_tx, data_wi),
            dim=0
        )
        calendar = make_calendar_data(self.calendar_data, cols)
        del data_ca, data_tx, data_wi; gc.collect()
        
        self.in_size=data.size()[1]+calendar.size()[0]
        
        data_set=item_id_store_id_Dataset(data, calendar)
        data_loader = torch.utils.data.DataLoader(data_set, batch_size = 200, shuffle = True)
        
        
        return data_loader

    def make_data_group_id(self, train_cols, state):
    
        group_sum_df = self.train_df.groupby(self.group_id)[train_cols].transform('sum')
        
        data_train = self.train_df[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']+train_cols]
        train_product = self.sample_submission_df[
            (self.sample_submission_df.id.str.contains(state))&(self.sample_submission_df.id.str.contains('_validation'))
            ].id.values
        #train_product = data_train[data_train.state_id==state]['id'].unique()
        
        data = data_train.loc[train_product,train_cols]
        group_sum_df = group_sum_df.loc[train_product, :]
        
        
        calendar_index = [ f'snap_{state}']
        event_index = [ f'snap_{state}']
        calendar = self.calendar_data.loc[calendar_index,:]
        for shift in [3, 7, 14, 28]:
            tmp_calendar = calendar.loc[event_index, :]
            tmp_calendar = tmp_calendar.T.shift(shift).T
            tmp_calendar.index = [f'{col}_shift{shift}' for col in tmp_calendar.index]
            calendar = pd.concat([
                calendar,
                tmp_calendar
            ], axis=0)
        calendar = calendar[train_cols]
        
        price = self.price_data.T[train_cols].loc[train_product,:]
        past_price_1 = self.price_data.loc[:,train_product].shift(3).T[train_cols]
        past_price_2 = self.price_data.loc[:,train_product].shift(7).T[train_cols]
        past_price_3 = self.price_data.loc[:,train_product].shift(14).T[train_cols]
        
        
        is_sell = self.is_sell_data[train_cols].loc[train_product,:]
        past_is_sell_1 = self.is_sell_data.T.shift(3).T.loc[train_product, train_cols]
        past_is_sell_2 = self.is_sell_data.T.shift(7).T.loc[train_product, train_cols]
        past_is_sell_3 = self.is_sell_data.T.shift(14).T.loc[train_product, train_cols]

        data = torch.FloatTensor(data.values.astype(float))
        group_sum_df = torch.FloatTensor(group_sum_df.values.astype(float))
        
        calendar = torch.FloatTensor(calendar.values.astype(float))
        
        price = torch.FloatTensor(price.values.astype(float))
        
        past_price_1 = torch.FloatTensor(past_price_1.values.astype(float))
        past_price_2 = torch.FloatTensor(past_price_2.values.astype(float))
        past_price_3 = torch.FloatTensor(past_price_3.values.astype(float))
        
        is_sell = torch.FloatTensor(is_sell.values.astype(float))
        past_is_sell_1 = torch.FloatTensor(past_is_sell_1.values.astype(float))
        past_is_sell_2 = torch.FloatTensor(past_is_sell_2.values.astype(float))
        past_is_sell_3 = torch.FloatTensor(past_is_sell_3.values.astype(float))
        
        data_list = []
        for idx in range(len(data)):
            _data = data[[idx],:]
            _group_sum_data = group_sum_df[[idx],:]
            _price = price[[idx],:]
            
            _past_price_1 = past_price_1[[idx],:]
            _past_price_2 = past_price_2[[idx],:]
            _past_price_3 = past_price_3[[idx],:]
            
            _is_sell = is_sell[[idx],:]
            
            _past_is_sell_1 = past_is_sell_1[[idx],:]
            _past_is_sell_2 = past_is_sell_2[[idx],:]
            _past_is_sell_3 = past_is_sell_3[[idx],:]
            
            x = torch.cat((
                _data, _group_sum_data,
                calendar,
                _price,
                _past_price_1, _past_price_2, _past_price_3,
                _is_sell,
                _past_is_sell_1, _past_is_sell_2, _past_is_sell_3
            ), dim=0)
            data_list.append(x.tolist())
        data_list = torch.FloatTensor(data_list)
        return data_list


    def prepare_training_groupId(self, model, df, cols, index_index):
        lr = 1e-4
        eta_min = 1e-3
        t_max = 10
        model = model.to(DEVICE)
        criterion = Loss_func_groupid(cols=cols, df=df, index_index=index_index, self.group_id)
        optimizer = RAdam(params=model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        return model, criterion, optimizer, scheduler

    def train_model_gruopId(self, model, data, calendar):
        self.df['index'] = self.df.index
        index_df = pd.concat([
            self.df.groupby(self.group_id)['index'].unique(),
            self.df.groupby(self.group_id)['index'].nunique()
        ], axis=1)
        index_df.columns=['index', 'length']
        index_df['index'] = index_df['index'].apply(lambda x: x.tolist())

        index_index = index_df.index
        index_df.reset_index(drop=True, inplace=True)
        
        data_set=indicate_index(index_df)
        data_loader = torch.utils.data.DataLoader(data_set, batch_size = 33, shuffle = True)
        
        model, criterion, optimizer, scheduler = self.prepare_training_groupId(model, self.df, self.cols, index_index)
        
        num_epochs = 40
        best_epoch = -1
        best_score = 10000
        early_stoppping_cnt = 0
        best_model = model
        
        
        for epoch in range(1,num_epochs+1):
            start_time = time.time()
            
            model.train()
            avg_loss = 0.
            
            for idx in tqdm(data_loader):
                optimizer.zero_grad()
                
                index = sum(index_df.iloc[idx]['index'].values.tolist(),[])
                length = index_df.iloc[idx]['length'].values.tolist()
                x_batch = _create_batch_data(index, data, calendar)
                x_batch = x_batch[:,:,:-28]
                gc.collect()
                
                x_batch, y_batch = split_X_y(x_batch)
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                
                preds = model(x_batch)
                
                loss = criterion(preds.cpu(), y_batch.cpu(), idx, length, train=True)
                loss = loss.to(DEVICE)
                
                loss.backward()
                optimizer.step()
                scheduler.step()

                avg_loss += loss.item() / len(data_loader)
                del loss; gc.collect()
            
            model.eval()
            avg_val_loss = 0.
            
            for idx in data_loader:
                x_batch = _create_batch_data(index, data, calendar)
                
                x_batch, y_batch = split_X_y(x_batch)
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                
                preds = model(x_batch)
                loss = criterion(preds.cpu(), y_batch.cpu(), idx, length, train=False)
                
                avg_val_loss += loss.item() / len(data_loader)
                del loss; gc.collect()
                
                
            if best_score>avg_val_loss:
                best_score = avg_val_loss
                early_stoppping_cnt=0
                best_epoch=epoch
                best_model = model
                elapsed = time.time() - start_time
                p_avg_val_loss = termcolor.colored(np.round(avg_val_loss, 4),"red")
                
                print(f'Epoch {epoch} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {p_avg_val_loss} time: {elapsed:.0f}s')
            else:
                early_stoppping_cnt+=1
                elapsed = time.time() - start_time
                print(f'Epoch {epoch} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f} time: {elapsed:.0f}s')
            
            if (epoch>10) and (early_stoppping_cnt>7):
                    break
        
        print(f'best_score : {best_score}    best_epoch : {best_epoch}')
        #torch.save(best_score.state_dict(), 'net.pt')
        
        return best_model, best_score