import os
from src import (
    evaluate_model,
    transform_data
)
from config import (
    transformed_dir,
    STOCKS
)

if __name__ == '__main__':

    if len(os.listdir(transformed_dir)) == 1:
        print('here')
        transform_data()

    print('Evaluating News Sentiment Model:')
    for stock in STOCKS:
        evaluate_model(stock, 'news')
    print('Evaluating Tweets Sentiment Model:')
    for stock in STOCKS:
        evaluate_model(stock, 'tweets')
        