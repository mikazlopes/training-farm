from finrl.config_tickers import MIGUEL_TICKER, DOW_30_TICKER, NAS_100_TICKER, BOT30_TICKER, ROUNDED_TICKER, TECH_TICKER, SINGLE_TICKER, DRL_ALGO_TICKERS

CONFIGURATIONS = {
    "ticker_list": [SINGLE_TICKER],
    "period_years": [1],
    "steps": ["period_years * 100000"],
    "learning_rate": [4e-6, 3e-6, 2e-6, 1e-6],
    "batch_size": [512, 1024, 2048, 4096],
    "net_dimensions": [[128,64], [256,128], [512,256], [1024,512], [128,64,32], [256,128,64], [512,256,128]]
}