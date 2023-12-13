#!/bin/bash

# Start MySQL
service mysql start

# Wait for MySQL to start up completely
while ! mysqladmin ping --silent; do
    sleep 1
done

# Set up the database if it doesn't exist
mysql -u root -e "CREATE DATABASE IF NOT EXISTS optuna_example;"
mysql -u root -e "CREATE USER IF NOT EXISTS 'optuna_user'@'localhost' IDENTIFIED BY 'r00t4dm1n';"
mysql -u root -e "GRANT ALL PRIVILEGES ON optuna_example.* TO 'optuna_user'@'localhost';"
mysql -u root -e "FLUSH PRIVILEGES;"

# Start optuna-dashboard in the background on port 8080
optuna-dashboard --port 8080 --storage "mysql://optuna_user:r00t4dm1n@localhost/optuna_example" &

# Execute the main process
exec "$@"

