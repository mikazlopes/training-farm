#!/bin/bash

# Start MySQL
service mysql start

# Use NUM_INSTANCES environment variable if set, otherwise use default value 2
num_instances=${NUM_INSTANCES:-2}

# Start hp-tuner.py in the background with the specified number of instances
python hp-tuner.py --period_years 10 --num_instances "$num_instances" &

# Sleep for a short period to ensure hp-tuner.py starts properly
sleep 5

# Start optuna-dashboard if STUDY_MODE is not "client"
if [ "$STUDY_MODE" != "client" ]; then
    optuna-dashboard mysql+pymysql://optuna_user:r00t4dm1n@localhost/optuna_example --host "0.0.0.0" &
fi

# Wait for user input to stop the container
while true; do
    read -p "Press y to stop the container: " input
    if [[ $input == "y" ]]; then
        break
    fi
done


