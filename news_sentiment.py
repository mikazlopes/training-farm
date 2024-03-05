from datetime import datetime 
import torch
from alpaca_trade_api import REST 
from finbert_utils import estimate_sentiment
from finrl.config_tickers import TEST_TICKERS
from finrl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_API_BASE_URL

API_KEY = ALPACA_API_KEY 
API_SECRET = ALPACA_API_SECRET 
BASE_URL = "https://paper-api.alpaca.markets"

api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

def get_sentiment_by_ticker(tickers):
    sentiments = {}
    for ticker in tickers:
        news = api.get_news(symbol=ticker,
                            start='2024-01-01',
                            end='2024-03-04',
                            limit=50, 
                            sort= 'asc',
                            include_content=True,
                            exclude_contentless=True)
        sentiment_scores = []
        probabilities = []
        for ev in news:
            news_headline = ev.__dict__["_raw"]["headline"]
            news_summary = ev.__dict__["_raw"]["summary"]
            news_timestamp = ev.__dict__["_raw"]["created_at"]
            probability, sentiment = estimate_sentiment([news_headline])
            sentiment_score = {'positive': 1, 'neutral': 0, 'negative': -1}.get(sentiment, 0)
            sentiment_scores.append(sentiment_score)
            probabilities.append(probability.item() if isinstance(probability, torch.Tensor) else probability)  # Convert tensor to scalar if necessary
            
        if sentiment_scores:
            average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            average_probability = sum(probabilities) / len(probabilities)
            sentiments[ticker] = {
                'average_sentiment': average_sentiment,
                'average_probability': average_probability
            }
    return sentiments

sentiments_by_ticker = get_sentiment_by_ticker(TEST_TICKERS)

for ticker, averages in sentiments_by_ticker.items():
    print(f"{ticker}: Average Sentiment = {averages['average_sentiment']:.2f}, Average Probability = {averages['average_probability']:.2f}")
