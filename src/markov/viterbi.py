import pandas as pd


def build_transition_map(merged_df, target_col):
    assert target_col in {'return_label', 'sentiment_label'}
    transitions = {}
    for t_curr in merged_df[target_col].unique():
        transitions[t_curr] = {}
        for t_next in merged_df[target_col].unique():
            transitions[t_curr][t_next] = 0

    for i in range(len(merged_df) - 1):
        row = merged_df.iloc[i]
        next_row = merged_df.iloc[i + 1]
        transitions[row[target_col]][next_row[target_col]] += 1
    for t_curr in transitions:
        score_sum = 0
        for t_next in transitions[t_curr].values():
            score_sum += t_next
        for t_next in transitions[t_curr]:
            transitions[t_curr][t_next] = transitions[t_curr][t_next] / score_sum
    return transitions


def build_emission_map(merged_df):
    emission_map = {}
    for l_return in merged_df["return_label"].unique():
        emission_map[l_return] = {}
        for l_sentiment in merged_df["sentiment_label"].unique():
            emission_map[l_return][l_sentiment] = 0
    for i in range(len(merged_df) - 1):
        row = merged_df.iloc[i]
        emission_map[row["return_label"]][row["sentiment_label"]] += 1
    for s_return in emission_map:
        score_sum = 0
        for s_frequency in emission_map[s_return].values():
            score_sum += s_frequency
        for s_sentiment in emission_map[s_return]:
            emission_map[s_return][s_sentiment] = emission_map[s_return][s_sentiment] / score_sum
    return emission_map


class Viterbi(object):

    def __init__(self, observation_matrix, transition_matrix):
        self.observation_map = observation_matrix
        self.transition_map = transition_matrix

    def predict_argmax(self, sentiments):
        backtrack = []
        check_map = {0: 0}
        for sentiment in sentiments:
            next_check = {}
            back = {}
            for state in check_map.keys():
                if state in self.transition_map:
                    for transition in self.transition_map[state]:
                        score = self.transition_map[state][transition] + check_map[state] + \
                                self.observation_map[state][sentiment]
                        if (transition not in next_check) or (score > next_check[transition]):
                            back[transition] = state
                            next_check[transition] = score

            backtrack.append(back)
            check_map = next_check
        max_score = 0
        pointer = None
        for state in check_map:
            if pointer is None or check_map[state] > max_score:
                max_score = check_map[state]
                pointer = state
        max_path = [pointer]
        for i in range(len(backtrack)-1, 0, -1):
            max_path.insert(0, backtrack[i][pointer])
            pointer = backtrack[i][pointer]
        return max_path
