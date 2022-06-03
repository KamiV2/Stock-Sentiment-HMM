import pandas as pd
from config import transformed_dir


def build_merged_df(stock, source):
    assert source in {'news', 'tweets'}
    source_df = pd.read_csv(f'{transformed_dir}/{source}/{stock}.csv')
    candle_df = pd.read_csv(f'{transformed_dir}/candles/{stock}.csv')
    candle_df['return_label'] = candle_df['label']
    candle_df['date'] = candle_df['Date']
    source_df['sentiment_label'] = source_df['label']
    candle_df = candle_df[['date', 'return_label', 'returns']]
    source_df = source_df[['date', 'sentiment_label']]
    merged = source_df.merge(candle_df, on='date')
    merged = merged.dropna()
    return merged
