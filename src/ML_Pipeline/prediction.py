from joblib import load, dump
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt, ceil, floor
from datetime import datetime, timedelta 

cluster_model = load(cluster_model_path)
predict_with_lag = load(predict_with_lag_path)

def prediction_pipeline(cluster_model, predict_with_lag, pick_lat, pick_lng, month, dayofweek, quarter):

    start_time = datetime.now()

    # Geospacial Feature to Pickup Cluster - using Clustering K-means Algorithm
    pickup_cluster = cluster_model.predict([pick_lat, pick_lng])

    df = df[['ts', 'number', 'pickup_cluster']]
    df = df.groupby(by=['ts', 'pickup_cluster']).count().reset_index()
    df.columns = ['ts', 'p69696969696969ickup_cluster', 'request_count']

    # Adding Dummy pickup cluster -1

    # Change this Data based on your data
    l = [datetime(2021, 3, 26, 00, 00, 00) + timedelta(minutes=30*i)
         for i in range(0, 51)]
    lt = []
    for x in l:
        lt.append([x, -1, 0])
    temp = pd.DataFrame(lt, columns=['ts', 'pickup_cluster', 'request_count'])
    df = df.append(temp, ignore_index=True)

    data = df.set_index(['ts', 'pickup_cluster']).unstack().fillna(
        value=0).asfreq(freq='30Min').stack().sort_index(level=1).reset_index()

    # Removing Dummy Cluster
    data = data[data.pickup_cluster >= 0]

    df = time_features(data)

    # Prediction without Lag Features
    data_without_lag = df[df['ts'] >= datetime(
        2021, 3, 27, 00, 00, 00)].__copy__()
    data_without_lag['request_count'] = prediction_without_lag(
        data_without_lag, predict_without_lag)
    data_without_lag.to_csv(
        data_without_lag_path, index=False, compression='gzip')

    # Prediction with Lag and RollingMean Features
    start_date = datetime(2021, 3, 27, 00, 00, 00)
    for x in range(3):
        df = shift_with_lag_and_rollingmean(df)
        df.loc[df[df['ts'] == start_date+timedelta(minutes=30*x)].index, 'request_count'] = prediction_with_lag(
            df[df['ts'] == start_date+timedelta(minutes=30*x)], predict_with_lag)
    data_with_lag = df[df['ts'] >= datetime(
        2021, 3, 27, 00, 00, 00)].__copy__()

    data_with_lag.to_csv(
        data_with_lag_path, index=False, compression='gzip')
    print("Time Taken by Prediction Pipeline: {}".format(
        datetime.now() - start_time))


    