import numpy as np
from numpy.random import choice
from pprint import pprint


def find_stationary_matrix(state_map):
    dim = len(state_map)
    matrix = np.zeros((dim, dim))
    for state in sorted(state_map):
        for next_state in sorted(state_map[state]):
            matrix[state][next_state] = state_map[state][next_state]
    S, U = np.linalg.eig(matrix.T)
    stationary = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
    stationary = np.real(stationary / np.sum(stationary))
    return stationary.tolist()


def generate_random_walk(state_map, n_steps=100, stationary=None):
    walk = []
    keys = sorted(state_map[0].keys())
    if stationary is None:
        stationary = find_stationary_matrix(state_map)
    next_move = choice(keys, p=stationary)
    for n in range(n_steps):
        walk.append(next_move)
        distribution = [state_map[next_move][k] for k in keys]

        next_move = choice(keys, p=distribution)
    return walk


def get_historical_walks(merged_df, start, period):
    sentiment_walk = list(merged_df["sentiment_label"].iloc[start:start + period])
    returns = list(merged_df["returns"].iloc[start:start + period])
    agg_return = 1
    for r in returns:
        agg_return *= r
    return sentiment_walk, agg_return
