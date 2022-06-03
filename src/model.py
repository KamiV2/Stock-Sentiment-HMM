from src.markov import (
    Viterbi,
    build_emission_map,
    build_transition_map,
)
from src.markov import (
    get_historical_walks,
    generate_random_walk,
)
import json
from config import transformed_dir, STOCKS
from src.merge import build_merged_df
import numpy as np


def get_mean_error(df, n_bins, labels, model, transitions, lag=7, period=28, iterations=100):
    error = []
    for i in range(len(df) // lag - period // lag):
        sentiment_walk, agg_return = get_historical_walks(df, lag * i, period)
        stationary_distribution = []
        for b in range(n_bins):
            stationary_distribution.append(sentiment_walk.count(b) / len(sentiment_walk))
        sub_return = []
        for j in range(iterations):
            walk = generate_random_walk(transitions, n_steps=lag, stationary=list(stationary_distribution))
            return_walk = model.predict_argmax(walk)
            expected = 1
            for j in range(len(return_walk)):
                expected *= labels[return_walk[j]]
            sub_return.append(expected)
        mean_return = np.mean(sub_return)
        error.append((mean_return / agg_return - 1) ** 2)
    return np.mean(error) ** 0.5


def evaluate_model(stock, source, n_bins=5):
    assert stock in STOCKS
    assert source in {'news', 'tweets'}
    with open(f'{transformed_dir}/distribution.json', "r") as handle:
        label_distribution = json.load(handle)
    merged_df = build_merged_df(stock, source)
    sentiment_transition_map = build_transition_map(merged_df, target_col='sentiment_label')
    return_transition_map = build_transition_map(merged_df, target_col='return_label')
    emission_map = build_emission_map(merged_df)
    viterbi = Viterbi(emission_map, return_transition_map)
    labels = label_distribution[stock]["returns"]
    lag = 7
    period = 28
    iterations = 100
    mean_error = get_mean_error(
        merged_df, n_bins, labels, viterbi,
        sentiment_transition_map,
        lag, period, iterations,
    )
    print(f'[{stock[:4]}] : {source} | '
          f'Accuracy: {1 - round(mean_error, 5)} | '
          f'LAG: {lag} | '
          f'PERIOD: {period} | '
          f'ITER: {iterations}')
