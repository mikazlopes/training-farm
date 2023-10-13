import pytest
import main
from finrl.config_tickers import BOT30_TICKER, DOW_30_TICKER, TECH_TICKER, ROUNDED_TICKER, MIGUEL_TICKER

def test_generate_combinations():
    # Mock CONFIGURATIONS for the test
    main.CONFIGURATIONS = {
        "ticker_list": [BOT30_TICKER, DOW_30_TICKER, TECH_TICKER, ROUNDED_TICKER, MIGUEL_TICKER],
        "period_years": [3, 5, 10],
        "learning_rate": [4e-6, 3e-6, 2e-6, 1e-6],
        "batch_size": [512, 1024, 2048, 4096],
        "steps": ["period_years * 100000", "(period_years - 1) * 100000", "(period_years - 2) * 100000"],
        "net_dimensions": [[128,64], [256,128], [512,256], [1024,512], [128,64,32], [256,128,64], [512,256,128]]
    }
    result = main.generate_combinations()
    assert len(result) == 5040
    assert all('steps' in combo for combo in result)

def test_get_gpu_id(mocker):
    # Mock the torch.cuda.device_count method to return 2 GPUs
    mocker.patch('torch.cuda.device_count', return_value=2)

    first_call = main.get_gpu_id()
    assert first_call == 0
    second_call = main.get_gpu_id()
    assert second_call == 1

def test_start_process(mocker):
    # Mock subprocess to prevent actual process creation
    mocker.patch('subprocess.Popen', return_value=None)
    # Mock logger to prevent actual logging
    mocker.patch.object(main.logger, 'info')
    # Mock get_gpu_id to return a constant GPU id
    mocker.patch.object(main, 'get_gpu_id', return_value=1)
    
    script = 'sample_script.py'
    configurations = [{'key': 'value'}]
    manager = main.ProcessManager(configurations)
    
    uid = manager.start_process(script)
    assert len(uid) == 6
    assert uid in manager.processes


