from finrl.config_tickers import MIGUEL_TICKER, DOW_30_TICKER, NAS_100_TICKER, BOT30_TICKER, ROUNDED_TICKER, TECH_TICKER, SINGLE_TICKER, DRL_ALGO_TICKERS

CONFIGURATIONS = {
    "ticker_list": [SINGLE_TICKER],
    "period_years": [6],
    "steps": ["period_years * 300000"],
    "initial_capital":[3e4, 1e6, 3e6],
    "horizon_length": [3000, 2000, 4000, 1500, 3500, 2500],
    "lambda_gae_adv": [0.95, 0.99, 0.90, 0.85, 0.80],
    "lambda_entropy": [0.01, 0.03, 0.05, 0.07, 0.09],
    "gamma":[0.985,0.990,0.995,0.999],
    "learning_rate": [3e-6, 3.5e-6, 4e-6, 2.5e-6, 2e-6, 1.5e-6, 1e-6],
    "batch_size": [2048, 1024, 4096, 512],
    "net_dimensions": [[128,64], [256,128], [512,256], [1024,512], [128,64,32], [256,128,64]]
}