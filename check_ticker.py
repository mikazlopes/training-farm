import yfinance as yf
from datetime import datetime, timedelta
from finrl.config_tickers import DRL_ALGO_TICKERS

# List of your tickers
tickers = DRL_ALGO_TICKERS

# Define the time period (last 5 years)
end_date = datetime.today()
start_date = end_date - timedelta(days=10*365)

# Convert dates to strings
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Check data availability
for ticker in tickers:
    data = yf.download(ticker, start=start_date_str, end=end_date_str)
    if len(data) == 0:
        print(f"No data found for {ticker}")
    else:
        print(f"Data available for {ticker}: {len(data)} data points")

