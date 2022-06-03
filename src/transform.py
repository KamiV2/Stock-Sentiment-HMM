from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import json
from random import random
from config import (
    STOCKS,
    data_dir,
    transformed_dir
)
import os


encoder = LabelEncoder()


def noise(d=.5, r=.02):
    return (random() - d) * r


def categorize(df, stock, source, target_col, n_bins, distribution):
    bins, n = pd.cut(df[target_col], n_bins, retbins=True, duplicates='raise')
    df['label'] = encoder.fit_transform(bins)
    sub = {}
    for key, ds, in df.groupby('label'):
        sub[key] = np.mean(ds[target_col])
    dist = [sub[k] for k in range(len(n) - 1)]
    if stock not in distribution:
        distribution[stock] = {}
    distribution[stock][source] = list(dist)
    return df


def transform_stock_data(stock, n_bins):
    price_df = pd.read_csv(f'{data_dir}/candles/{stock}.csv')
    tweet_df = pd.read_csv(f'{data_dir}/tweets/{stock}.csv')
    news_df = pd.read_csv(f'{data_dir}/news/{stock}.csv')

    with open(f'{transformed_dir}/distribution.json', 'r') as handle:
        distribution = json.load(handle)
    news_df['score'] = news_df['score'].apply(lambda x: (x + noise() if x == 0 else x))
    tweet_df['score'] = tweet_df['score'].apply(lambda x: (x + noise() if x == 0 else x))

    returns = np.array(price_df['Close'].to_list()) / np.array(price_df['Open'].to_list())
    price_df['returns'] = list(returns)

    tweet_df = categorize(tweet_df, stock, 'tweets', 'score', n_bins, distribution)
    news_df = categorize(news_df, stock, 'news', 'score', n_bins, distribution)
    price_df = categorize(price_df, stock, 'returns', 'returns', n_bins, distribution)

    price_df.to_csv(f'{transformed_dir}/candles/{stock}.csv')
    tweet_df.to_csv(f'{transformed_dir}/tweets/{stock}.csv')
    news_df.to_csv(f'{transformed_dir}/news/{stock}.csv')

    with open(f'{transformed_dir}/distribution.json', 'w') as handle:
        json.dump(distribution, handle, indent=4)


def transform_data(n_bins=5):
    os.mkdir(f'{transformed_dir}/candles')
    os.mkdir(f'{transformed_dir}/tweets')
    os.mkdir(f'{transformed_dir}/news')
    with open(f'{transformed_dir}/distribution.json', 'w') as handle:
        json.dump({}, handle)
    for stock in STOCKS:
        transform_stock_data(stock, n_bins)

