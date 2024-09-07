import pandas as pd


def train_test_data_prep(df_train, df_test):
    df_test = df_test.sort_values(by=['pickup_cluster', 'ts']).drop_duplicates(
        subset=['ts', 'pickup_cluster'])
    temp = pd.concat([df_train, df_test])
    temp = temp.sort_values(by=['pickup_cluster', 'ts']).drop_duplicates(
        subset=['ts', 'pickup_cluster'])
    temp = temp.set_index(
        ['ts', 'pickup_cluster', 'mins', 'hour', 'month', 'quarter', 'dayofweek'])
    # temp = temp.set_index(
    #     ['ts', 'pickup_cluster', 'mins', 'hour', 'month', 'dayofweek'])

    temp['lag_1'] = temp.groupby(level=['pickup_cluster'])[
        'request_count'].shift(1)
    temp['lag_2'] = temp.groupby(level=['pickup_cluster'])[
        'request_count'].shift(2) 
    temp['lag_3'] = temp.groupby(level=['pickup_cluster'])[
        'request_count'].shift(3)
    temp['rolling_mean'] = temp.groupby(level=['pickup_cluster'])['request_count'].apply(
        lambda x: x.rolling(window=6).mean()).shift(1)

    temp = temp.reset_index(drop=False).dropna()
    temp = temp[['ts', 'pickup_cluster', 'mins', 'hour', 'month', 'quarter',
                 'dayofweek', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean', 'request_count']]
    # temp = temp[['ts', 'pickup_cluster', 'mins', 'hour', 'month',
    #              'dayofweek', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean', 'request_count']]
    train1 = temp[temp.ts.dt.day <= 23]
    test1 = temp[temp.ts.dt.day > 23]

    X = train1.iloc[:, 1:-1]
    y = train1.iloc[:, -1]
    X_test = test1.iloc[:, 1:-1]
    y_test = test1.iloc[:, -1]
    return X, y, X_test, y_test
