class item_id_store_id_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, calendar):
        self.data = data
        self.calendar = calendar
        self.datanum = len(data)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        _data = self.data[idx, :, :]
        x = torch.cat((_data, self.calendar), dim=0)
        return x, idx

class Loss_func_item_id_store_id_(nn.Module):
    def __init__(self, df, cols):
        super(Loss_func_item_id_store_id_, self).__init__()
        last_d = int(cols[-1].replace('d_', ''))
        d_cols = df.columns[df.columns.str.startswith('d_')]
        train_d_cols = last_d-28*2
        self.train_d_cols = d_cols[:train_d_cols]
        test_d_cols = last_d-28
        self.test_d_cols = d_cols[:test_d_cols]
        self._create_denominator(df)
        
    def _create_denominator(self, df):
        
        train_value = df[self.train_d_cols]
        train_value = train_value.values
        train_value = train_value[:,1:]-train_value[:,:-1]
        train_value = train_value**2
        train_value = train_value.mean(1)
        train_value[train_value==0]=1
        self.train_value = torch.FloatTensor(train_value)
        
        test_value = df[self.test_d_cols]
        test_value = test_value.values
        test_value = test_value[:,1:]-test_value[:,:-1]
        test_value = test_value**2
        test_value = test_value.mean(1)
        test_value[test_value==0]=1
        self.test_value = torch.FloatTensor(test_value)
        
    def forward(self, preds, true, idx, train):
        loss = (preds-true)**2
        loss = loss.mean(1)
        loss = loss.squeeze()
        if train:
            loss = loss/self.train_value[idx]
        else:
            loss = loss/self.test_value[idx]
        loss = torch.sqrt(loss)
        loss = loss.mean()
        return loss

class TrainModel_itemId_store():
    def __init__(self, path):
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
        data_ca = make_data(cols, state, self.train_df, self.calendar_data, self.price_data, self.is_sell, self.sample_submission_df)
        state='TX'
        data_tx = make_data(cols, state, self.train_df, self.calendar_data, self.price_data, self.is_sell, self.sample_submission_df)
        state='WI'
        data_wi = make_data(cols, state, se
                            lf.train_df, self.calendar_data, self.price_data, self.is_sell, self.sample_submission_df)


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


    def make_data_item_id_store_id_(self, train_cols, state):
        data_train = self.train_df[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']+train_cols]
        train_product = self.sample_submission_df[
            (self.sample_submission_df.id.str.contains(state))&(self.sample_submission_df.id.str.contains('_validation'))
            ].id.values
        #train_product = data_train[data_train.state_id==state]['id'].unique()
        
        data = data_train.loc[train_product,train_cols]
        
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
            _price = price[[idx],:]
            
            _past_price_1 = past_price_1[[idx],:]
            _past_price_2 = past_price_2[[idx],:]
            _past_price_3 = past_price_3[[idx],:]
            
            _is_sell = is_sell[[idx],:]
            
            _past_is_sell_1 = past_is_sell_1[[idx],:]
            _past_is_sell_2 = past_is_sell_2[[idx],:]
            _past_is_sell_3 = past_is_sell_3[[idx],:]
            
            x = torch.cat((
                _data, calendar,
                _price,
                _past_price_1, _past_price_2, _past_price_3,
                _is_sell,
                _past_is_sell_1, _past_is_sell_2, _past_is_sell_3
            ), dim=0)
            data_list.append(x.tolist())
        data_list = torch.FloatTensor(data_list)
        return data_list


    def prepare_training_item_id_store_id_(self, model, df, cols):
        lr = 1e-4
        eta_min = 1e-3
        t_max = 10
        model = model.to(DEVICE)
        criterion = Loss_func_item_id_store_id_(cols=cols, df=df)
        optimizer = RAdam(params=model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        return model, criterion, optimizer, scheduler
    
    def train_model_item_id_store_id_(self, model, data_loader):
        model, criterion, optimizer, scheduler = self.prepare_training_item_id_store_id_(model, self.df, self.cols)

        num_epochs = 40
        best_epoch = -1
        best_score = 10000
        early_stoppping_cnt = 0
        best_model = model

        for epoch in range(num_epochs):
            start_time = time.time()

            model.train()
            avg_loss = 0.
            #data_loader = torch.utils.data.DataLoader(data_set, batch_size = 150, shuffle = True)
            for x_batch, idx in tqdm(data_loader):
                optimizer.zero_grad()
                x_batch = x_batch[:,:,:-28]; gc.collect()

                x_batch, y_batch = split_X_y(x_batch)
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                preds = model(x_batch)

                loss = criterion(preds.cpu(), y_batch.cpu(), idx, train=True)
                loss = loss.to(DEVICE)

                loss.backward()
                optimizer.step()
                #scheduler.step()

                avg_loss += loss.item() / len(data_loader)
                del loss; gc.collect()

            model.eval()
            avg_val_loss = 0.

            for x_batch, idx in data_loader:
                x_batch, y_batch = split_X_y(x_batch)
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                preds = model(x_batch)
                loss = criterion(preds.cpu(), y_batch.cpu(), idx, train=False)

                avg_val_loss += loss.item() / len(data_loader)
                del loss; gc.collect()


            if best_score>avg_val_loss:
                best_score = avg_val_loss
                early_stoppping_cnt=0
                best_epoch=epoch
                best_model = model
                elapsed = time.time() - start_time
                p_avg_val_loss = termcolor.colored(np.round(avg_val_loss, 4),"red")

                print(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {p_avg_val_loss} time: {elapsed:.0f}s')
            else:
                early_stoppping_cnt+=1
                elapsed = time.time() - start_time
                print(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f} time: {elapsed:.0f}s')

            if (epoch>10) and (early_stoppping_cnt>7):
                    break

        print(f'best_score : {best_score}    best_epoch : {best_epoch}')
        #torch.save(best_score.state_dict(), 'net.pt')

        return best_model, best_score