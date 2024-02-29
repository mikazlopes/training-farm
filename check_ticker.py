import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from finrl.config_tickers import VOLATILE_SMALL_CAP_TICKERS, TECH_TICKER, SINGLE_TICKER

# Checks a ticker list to see if all the tickers have data in the past years specified below in period_years

class AlpacaDataFetcher:
    def __init__(self, api_key, api_secret, base_url):
        self.api = tradeapi.REST(api_key, api_secret, base_url=base_url)

    def _fetch_data_for_ticker(self, ticker, start_date, end_date, time_interval):
        bars = self.api.get_bars(
            ticker,
            time_interval,
            start=start_date,
            end=end_date.strftime('%Y-%m-%d')
        ).df
        bars['symbol'] = ticker
        return bars

    def check_data_availability(self, tickers, start_date, end_date, time_interval='1Day'):
        for ticker in tickers:
            try:
                data = self._fetch_data_for_ticker(ticker, start_date, end_date, time_interval)
                if data.empty:
                    print(f"No data found for {ticker}")
                else:
                    print(f"Data available for {ticker}: {len(data)} data points")
            except Exception as e:
                print(f"Error fetching data for {ticker}: {str(e)}")

# Your Alpaca API key and secret key
from finrl.config_private import ALPACA_API_KEY as API_KEY
from finrl.config_private import ALPACA_API_SECRET as API_SECRET
from finrl.config_private import ALPACA_API_BASE_URL as BASE_URL

# Initialize AlpacaDataFetcher
data_fetcher = AlpacaDataFetcher(API_KEY, API_SECRET, BASE_URL)

# List of your tickers
tickers = SINGLE_TICKER

# Define the time period
end_date = datetime.today()
start_date = '2016-01-01'

# Check data availability
data_fetcher.check_data_availability(tickers, start_date, end_date)
