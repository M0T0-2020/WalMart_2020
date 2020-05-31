def shift_seven(data, cols):
    #['shift_1', 'shift_2', 'shift_3', 'shift_4']
    data[cols] = data.groupby(['id'])[cols].shift(7)
    return data
    
def shift_one(data,cols):
    #['roll_28_std', 'roll_28_mean', 'diff_std_1', 'diff_mean_1', 'diff_std_7', 'diff_mean_7']
    data[cols]=data.groupby(['id'])[cols].shift(1)
    return data