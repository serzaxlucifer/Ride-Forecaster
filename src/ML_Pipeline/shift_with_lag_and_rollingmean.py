import pandas as pd
import numpy as np
from joblib import load, dump
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt, ceil, floor
from datetime import datetime, timedelta


def shift_with_lag_and_rollingmean(df):
    df = df.sort_values(by=['pickup_cluster', 'ts']).drop_duplicates(
        subset=['ts', 'pickup_cluster'])
    df = df.set_index(['ts', 'pickup_cluster', 'mins',
                       'hour', 'month', 'quarter', 'dayofweek'])
    df['lag_1'] = df.groupby(level=['pickup_cluster'])[
        'request_count'].shift(1)
    df['lag_2'] = df.groupby(level=['pickup_cluster'])[
        'request_count'].shift(2) 
    df['lag_3'] = df.groupby(level=['pickup_cluster'])[
        'request_count'].shift(3)
    df['rolling_mean'] = df.groupby(level=['pickup_cluster'])['request_count'].apply(
        lambda x: x.rolling(window=3).mean()).shift(1)

    df = df.reset_index(drop=False).dropna()
    df = df[['ts', 'pickup_cluster', 'mins', 'hour', 'month', 'quarter',
             'dayofweek', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean', 'request_count']]
    return df
