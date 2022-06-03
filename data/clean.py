import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from tqdm import tqdm
from config import STOCKS

nltk.download('vader_lexicon')
model = SentimentIntensityAnalyzer()


def average_time_series(time_series):
    for date in time_series:
        if len(time_series[date]) > 0:
            time_series[date] = sum(time_series[date]) / len(time_series[date])
        else:
            time_series[date] = 0
    return time_series


def process_news_file(filename, stock):
    df = pd.read_csv(filename)
    time_series = {}
    for element in tqdm(df.itertuples()):
        string = element.title
        date = element.release_date[:10]
        score = model.polarity_scores(string)['compound']
        if date not in time_series:
            time_series[date] = []
        if score != 0:
            time_series[date].append(score)
    time_series = average_time_series(time_series)
    frame = {
        'date': list(time_series.keys()),
        'score': list(time_series.values())
    }
    output_df = pd.DataFrame(frame)
    output_df.to_csv(f'news/{stock}.csv')


def process_tweet_file(filename, stock):
    time_series = {}
    df = pd.read_csv(filename)
    for element in tqdm(df.itertuples()):
        string = element.body
        date = element.post_date
        score = model.polarity_scores(string)['compound']
        if date not in time_series:
            time_series[date] = []
        if score != 0:
            time_series[date].append(score)
    time_series = average_time_series(time_series)
    frame = {
        'date': list(time_series.keys()),
        'score': list(time_series.values())
    }
    output_df = pd.DataFrame(frame)
    output_df.to_csv(f'tweets/{stock}.csv')


def clean_candle_file(stock):
    cols = {'Date', 'Open', 'Low', 'Close', 'Volume'}
    df = pd.read_csv(f'../data/candles/{stock}.csv')
    for col in df.columns:
        if col not in cols:
            df = df.drop(col, axis=1)
    df = df.set_index('Date')
    df.to_csv(f'../data/candles/{stock}.csv')


def clean_tweet_file(stock):
    cols = {'date', 'score'}
    df = pd.read_csv(f'../data/tweets/{stock}.csv')
    for col in df.columns:
        if col not in cols:
            df = df.drop(col, axis=1)
    df = df.set_index('date')
    df.to_csv(f'../data/tweets/{stock}.csv')


def clean_news_file(stock):
    cols = {'date', 'score'}
    df = pd.read_csv(f'../data/news/{stock}.csv')
    for col in df.columns:
        if col not in cols:
            df = df.drop(col, axis=1)
    df = df.set_index('date')
    df.to_csv(f'../data/news/{stock}.csv')


def clean_data():
    for stock in STOCKS:
        print(f'Processing News for [{stock}]')
        process_news_file(f'raw_news/{stock}.csv', stock)
        print(f'Processing Tweets for [{stock}]')
        process_tweet_file(f'raw_tweets/{stock}.csv', stock)
        print(f'Cleaning Tweets for [{stock}]')
        clean_tweet_file(stock)
        print(f'Cleaning Candles for [{stock}]')
        clean_candle_file(stock)
        print(f'Cleaning News for [{stock}]')
        clean_news_file(stock)
