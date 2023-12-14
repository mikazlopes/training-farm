#!/bin/bash

# Start MySQL
service mysql start

# Start hp-tuner.py in the background
python run-hp.py &

sleep 5

# Start optuna-dashboard
nohup optuna-dashboard mysql+pymysql://optuna_user:r00t4dm1n@localhost/optuna_example --host "0.0.0.0"


