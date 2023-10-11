import os
import sys
import uuid
import hashlib
import logging
import csv
import itertools
import threading
import time
import torch
import signal  # for handling the Ctrl+C interruption
import subprocess  # for starting the TrainerProcess
from flask import Flask, current_app  # importing current_app
from flask_socketio import SocketIO, join_room
from datetime import datetime, timedelta
from collections import deque, OrderedDict
from configurations import CONFIGURATIONS  # Importing the configuration file
import eventlet
eventlet.monkey_patch()

logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
                logging.FileHandler("main.log"),
                logging.StreamHandler()])

logger = logging.getLogger(__name__)

current_gpu_id = -1

def get_gpu_id():
    global current_gpu_id
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return -1
    current_gpu_id = (current_gpu_id + 1) % num_gpus
    return current_gpu_id


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
    def __init__(self, uid, script, config, gpu_id=-1):
        self.script = script
        self.config = config
        self.uid = uid
        self.gpu_id = gpu_id
        self.process = None

    def start(self):
        args = [sys.executable, self.script, '--uid', self.uid, '--gpu_id', int(self.gpu_id)]
        for key, value in self.config.items():
            if key == "ticker_list":
                value = ','.join(value)
            elif key == "net_dimensions":
                value = str(value)
            args.extend([f'--{key}', str(value)])
        self.process = subprocess.Popen(args)
        logger.info("Started %s with UID: %s and Config: %s", self.script, self.uid, self.gpu_id, self.config)

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
        uid = hashlib.md5(str(uuid.uuid4()).encode('utf-8')).hexdigest()[:6]
        config = self.configurations[self.current_config_index]
        gpu_id = get_gpu_id()
        proc = TrainerProcess(uid, script, config, gpu_id)
        proc.start()
        self.processes[uid] = proc
        self.last_active[uid] = datetime.now()
        self.current_config_index += 1
        collected_values[uid] = deque(maxlen=3)  # Initialize here
        return uid
    
    def terminate_process(self, uid, force_kill=False):
        if uid in self.processes:
            proc = self.processes[uid]
            if force_kill:
                proc.terminate()
                proc.process.wait()
                del self.processes[uid]
            else:
                sio.emit('terminate_yourself', {'script': uid}, room=uid)
                sio.sleep(5)
                if uid in self.processes:
                    logger.info(f"Had to force kill {uid}")
                    proc.terminate()
                    proc.process.wait()
                    del self.processes[uid]
    
    def restart_process(self, uid, force_kill=False):
        script = self.processes[uid].script  # getting the script path from the terminated process
        self.terminate_process(uid, force_kill)
        return self.start_process(script)
    
    
    def update_leaderboard(self, script, uid, return_value, cwd):
        self.leaderboard[(script, uid)] = {'return_value': return_value, 'cwd': cwd}
        self.leaderboard = OrderedDict(sorted(self.leaderboard.items(), key=lambda x: x[1]['return_value'], reverse=True))
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
                    #manager.restart_process(script, force_kill=True)
            sio.sleep(1)  # Check every second

app = Flask(__name__)
sio = SocketIO(app, cors_allowed_origins="*")
collected_values = {}

def write_processes_to_csv(file_path, script, uid, return_value, cwd, config):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        config_values = [config[key] for key in config]
        writer.writerow([script, uid, return_value, cwd] + config_values)


@app.route('/')
def index():
    return 'Server is running!'

@sio.on('connect')
def on_connect():
    logger.info("Client Connected - Server side")

@sio.on('join')
def on_join(data):
    uid = data['uid']
    join_room(uid)
    return {"status": "joined"}

@sio.on('disconnect')
def on_disconnect():
    logger.info(f"Client Disconnected - Server side")

@sio.on('heartbeat')
def handle_heartbeat(data):
    uid = data.get('uid')
    if uid:
        manager.last_active[uid] = datetime.now()
    else:
        logger.warning(f"Invalid heartbeat data received: {data}")

@sio.on('message')
def handle_message(data):
    uid = data.get('uid')
    manager.last_active[uid] = datetime.now()
    message_type = data.get('type')
    cwd = data.get('cwd')
    value = float(data.get('value'))

    if message_type == 'training':
        if uid not in collected_values:
            logger.error("Invalid UID received: %s", uid)
            return
        collected_values[uid].append(value)
        logger.info(f'{uid}: {collected_values[uid]}')
        if len(collected_values[uid]) == 3:
            average = sum(collected_values[uid]) / 3
            if average < 150:
                logger.info(f"Average for {uid} too low, killing training")
                collected_values[uid].clear()
                new_uid = manager.restart_process(uid)

    elif message_type == 'returns':
        current_config = manager.processes[uid].config if uid in manager.processes else {}
        write_processes_to_csv('processes.csv', SCRIPT_PATH, uid, value, cwd, current_config)
        if uid in manager.processes and manager.processes[uid].process.poll() is None:
            manager.terminate_process(uid)
        collected_values[uid].clear()
        new_uid = manager.start_process(SCRIPT_PATH)
        manager.update_leaderboard(SCRIPT_PATH, new_uid, value, cwd)

def exit_handler(signum, frame):
    logger.info("Terminating all processes...")
    for process in manager.processes.values():
        process.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, exit_handler)

def start_scripts(script, instances, manager):
    for _ in range(instances):
        manager.start_process(script)
        uid = list(manager.processes.keys())[-1]
        while not collected_values[uid]:
            time.sleep(5)

if __name__ == "__main__":
    SCRIPT_PATH = 'trainer.py'
    NUM_INSTANCES = 3
    
    configurations = generate_combinations()
    manager = ProcessManager(configurations=configurations)
    
    if not os.path.exists('processes.csv'):
        with open('processes.csv', 'w') as file:
            writer = csv.writer(file)
            first_config = configurations[0] if configurations else {}
            headers = ['Script', 'UID', 'Return Value', 'cwd'] + list(first_config.keys())
            writer.writerow(headers)

    thread = threading.Thread(target=start_scripts, args=(SCRIPT_PATH, NUM_INSTANCES, manager))
    thread.start()
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 5678)), app)