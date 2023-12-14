#!/bin/bash

# Start MySQL
service mysql start

# Start hp-tuner.py in the foreground and wait for it to finish
python hp-tuner.py --period_years 1 --gpu_id -1 --num_instances 1 &

sleep 5

# Start optuna-dashboard
optuna-dashboard mysql+pymysql://optuna_user:r00t4dm1n@localhost/optuna_example --host "0.0.0.0" &

# Wait for optuna-dashboard to be up
sleep 5

# Prompt the user after run-hp.py has finished
while true; do
    read -p "Press y to stop the container: " input
    if [[ $input == "y" ]]; then
        break
    fi
done


