import os
import sys
import uuid
import hashlib
import logging
import csv
import itertools
import threading
import time
import signal  # for handling the Ctrl+C interruption
import subprocess  # for starting the TrainerProcess
from flask import Flask, current_app  # importing current_app
from flask_socketio import SocketIO
from datetime import datetime, timedelta
from collections import deque, OrderedDict
from configurations import CONFIGURATIONS  # Importing the configuration file

logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
                logging.FileHandler("main.log"),
                logging.StreamHandler()])

logger = logging.getLogger(__name__)

def generate_combinations():
    configurations = CONFIGURATIONS.copy()
    steps = configurations.pop('steps')
    
    keys, values = zip(*configurations.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    all_combinations = []
    for comb in combinations:
        for step_expr in steps:
            new_comb = comb.copy()
            new_comb['steps'] = eval(step_expr.replace('period_years', str(new_comb['period_years'])))
            all_combinations.append(new_comb)
    
    return all_combinations


class TrainerProcess:
    def __init__(self, script, config):
        self.script = script
        self.config = config
        self.uid = hashlib.md5(str(uuid.uuid4()).encode('utf-8')).hexdigest()[:5]
        self.process = None
    
    def start(self):
        args = [sys.executable, self.script, '--uid', self.uid]
        for key, value in self.config.items():
            if key == "ticker_list":
                value = ','.join(value)  # Convert the list to a string with ',' as the separator
            elif key == "net_dimensions":
                value = str(value)  # pass the list as a string
            args.extend([f'--{key}', str(value)])
        self.process = subprocess.Popen(args)
        logger.info("Started %s with UID: %s and Config: %s", self.script, self.uid, self.config)


        
    def terminate(self):
        if self.process:
            self.process.terminate()


class ProcessManager:
    def __init__(self, configurations):
        self.processes = {}
        self.configurations = deque(configurations)
        self.leaderboard = OrderedDict()
        self.last_active = {}
        self.current_config_index = 0
        
    
    def start_process(self, script):
        if self.current_config_index < len(self.configurations):
            config = self.configurations[self.current_config_index]
            proc = TrainerProcess(script, config)
            proc.start()
            self.processes[script] = proc
            self.last_active[script] = datetime.now()
            self.current_config_index += 1
            return proc.uid
    
    def terminate_process(self, script, force_kill=False):
        if script in self.processes:
            if force_kill:  # If the process was inactive
                self.processes[script].terminate()  # Directly terminate
                self.processes[script].process.wait()
                del self.processes[script]
            else:  # If it's due to a low average
                # Inform the script to terminate itself
                sio.emit('terminate_yourself', room=script)
                # Optionally, you can add a wait period here and then forcefully terminate if the process doesn't shut down in the given period.
                sio.sleep(5)
                if script in self.processes:
                    self.processes[script].terminate()
                    self.processes[script].process.wait()
                    del self.processes[script]

    
    def restart_process(self, script, force_kill=False):
        self.terminate_process(script, force_kill)
        uid = self.start_process(script)
        self.last_active[script] = datetime.now()
        return uid
    
    def update_leaderboard(self, script, uid, return_value):
        self.leaderboard[(script, uid)] = return_value
        self.leaderboard = OrderedDict(sorted(self.leaderboard.items(), key=lambda x: x[1], reverse=True))
        logger.info("Leaderboard Updated: %s", self.leaderboard)
        if not self.configurations and not any(p.process and p.process.poll() is None for p in self.processes.values()):
            logger.info("All configurations processed. Final leaderboard: %s", self.leaderboard)

def monitor_processes(app):
    with app.app_context():
        while True:
            now = datetime.now()
            for script in list(manager.processes.keys()):
                last_time = manager.last_active.get(script, now)
                if now - last_time > timedelta(seconds=60):
                    logger.info(f"{script} has been inactive. Restarting...")
                    manager.restart_process(script, force_kill=True)
            sio.sleep(1)  # Check every second

app = Flask(__name__)
sio = SocketIO(app, cors_allowed_origins="*")
collected_values = {'trainer1/trainer1.py': deque(maxlen=5),
                    'trainer2/trainer2.py': deque(maxlen=5),
                    'trainer3/trainer3.py': deque(maxlen=5),
                    'trainer4/trainer4.py': deque(maxlen=5),
                    'trainer5/trainer5.py': deque(maxlen=5)}


def write_processes_to_csv(file_path, script, uid, return_value, config):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Extracting values from the config in the order of its keys
        config_values = [config[key] for key in config]
        writer.writerow([script, uid, return_value] + config_values)

@app.route('/')
def index():
    return 'Server is running!'


@sio.on('connect')
def on_connect():
    logger.info("Client Connected - Server side")
    sio.start_background_task(monitor_processes, current_app._get_current_object())

@sio.on('disconnect')
def on_disconnect():
    logger.info(f"Client Disconnected - Server side")

@sio.on('heartbeat')
def handle_heartbeat(data):
    script = data.get('script')
    if script:  # check if script is not None or empty
        manager.last_active[script] = datetime.now()
    else:
        logger.warning(f"Invalid heartbeat data received: {data}")


@sio.on('message')
def handle_message(data):
    script = data.get('script')
    manager.last_active[script] = datetime.now()  # Update the last active time for this script
    uid = data.get('uid')
    message_type = data.get('type')
    value = float(data.get('value'))

    if message_type == 'training':
        if script not in collected_values:
            logger.error("Invalid script name received: %s", script)
            return  # terminate the function early if the script name is invalid
        collected_values[script].append(value)
        logger.info(collected_values[script])
        if len(collected_values[script]) == 5:
            average = sum(collected_values[script]) / 5
            if average < 150:
                logger.info(f"Average for {script} too low, killing training")
                #Send termination signal
                sio.emit('terminate_process', {'script': script})
                collected_values[script].clear()
                uid = manager.start_process(script)

    elif message_type == 'returns':
        # Retrieving the configuration for the current script and UID
        current_config = manager.processes[script].config if script in manager.processes else {}
        write_processes_to_csv('processes.csv', script, uid, value, current_config)
        if script in manager.processes and manager.processes[script].process.poll() is None:
            manager.terminate_process(script)
        collected_values[script].clear()
        uid = manager.start_process(script)  # Starting new process with the next configuration from the queue
        manager.update_leaderboard(script, uid, value)  # Update leaderboard

def exit_handler(signum, frame):
    logger.info("Terminating all processes...")
    for process in manager.processes.values():
        process.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, exit_handler)

def start_scripts(scripts, manager):
    for script in scripts:
        uid = manager.start_process(script)
        time.sleep(300)  # Delay of 60 seconds between starting each script


if __name__ == "__main__":
    scripts = [ 'trainer1/trainer1.py', 
                'trainer2/trainer2.py',
                'trainer3/trainer3.py',
                'trainer4/trainer4.py',
                'trainer5/trainer5.py']
    configurations = generate_combinations()
    manager = ProcessManager(configurations=configurations)
    
    # Creating CSV file if not exist
    
    if not os.path.exists('processes.csv'):
        with open('processes.csv', 'w') as file:
            writer = csv.writer(file)
            # Extracting the first configuration to get its keys
            first_config = configurations[0] if configurations else {}
            headers = ['Script', 'UID', 'Return Value'] + list(first_config.keys())
            writer.writerow(headers)

    # Start the scripts in a separate thread
    thread = threading.Thread(target=start_scripts, args=(scripts, manager))
    thread.start()
    sio.run(app, port=5678)