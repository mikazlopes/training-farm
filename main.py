from __future__ import annotations
import trio
import eventlet
eventlet.monkey_patch()
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
from flask import Flask, render_template, jsonify


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
        args = [sys.executable, self.script, '--uid', self.uid, '--gpu_id', str(self.gpu_id)]
        for key, value in self.config.items():
            if key == "ticker_list":
                value = ','.join(value)
            elif key == "net_dimensions":
                value = str(value)
            args.extend([f'--{key}', str(value)])
        self.process = subprocess.Popen(args)
        logger.info("Started %s with UID: %s, GPU ID: %d, and Config: %s", self.script, self.uid, self.gpu_id, self.config)


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

        if self.current_config_index >= len(self.configurations):
            logger.info("All configurations tested... waiting on existing processes to finish")
            return None

        uid = hashlib.md5(str(uuid.uuid4()).encode('utf-8')).hexdigest()[:6]
        config = self.configurations[self.current_config_index]
        gpu_id = get_gpu_id()
        proc = TrainerProcess(uid, script, config, gpu_id)
        proc.start()
        self.processes[uid] = proc
        self.last_active[uid] = datetime.now()
        self.current_config_index += 1
        collected_values[uid] = deque(maxlen=3)  # Initialize here
        send_progress_update()   # Emit progress update
        active_processes = get_active_processes()
        sio.emit('update_active_processes', active_processes)
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
                # Remove the process from collected_values
                if uid in collected_values:
                    del collected_values[uid]
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
        if self.current_config_index >= len(self.configurations) and not any(p.process and p.process.poll() is None for p in self.processes.values()):
            self.print_top_leaderboard(5)

    def print_top_leaderboard(self, count=5):
        sorted_leaderboard = sorted(self.leaderboard.items(), key=lambda x: x[1]['return_value'], reverse=True)
        top_entries = sorted_leaderboard[:count]

        logger.info("Top %d leaderboard entries:", count)
        for idx, (key, value) in enumerate(top_entries, start=1):
            script, uid = key
            logger.info("%d. UID: %s, Script: %s, Return Value: %s, CWD: %s", idx, uid, script, value['return_value'], value['cwd'])


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
sio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

collected_values = {}

def write_processes_to_csv(file_path, script, uid, return_value, cwd, config):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        config_values = [config[key] for key in config]
        writer.writerow([script, uid, return_value, cwd] + config_values)

@app.route('/dashboard')
def dashboard():
    total_configurations = len(manager.configurations)
    progress_percentage = 100 * manager.current_config_index / total_configurations if total_configurations else 0
    
    # Fetching top 5 from the leaderboard
    sorted_leaderboard = sorted(manager.leaderboard.items(), key=lambda x: x[1]['return_value'], reverse=True)

    # Fetching active and completed processes data for the tables
    active_processes = get_active_processes()
    completed_processes = [{'uid': key[1], 'return_value': value['return_value']} for key, value in manager.leaderboard.items()]

    return render_template('dashboard.html',
                           leaderboard=sorted_leaderboard[:5],
                           active_processes=active_processes,
                           completed_processes=completed_processes,
                           progress_percentage=progress_percentage)

def get_active_processes():
    processes = []
    for uid, values in collected_values.items():
        if not values:
            last_value = 0
        else:
            last_value = values[-1]
        processes.append({'uid': uid, 'last_value': last_value})
    return processes


@app.route('/start_process', methods=['POST'])
def start_new_process():
    uid = manager.start_process(SCRIPT_PATH)
    send_active_processes()   # Emit update
    send_completed_processes()   # Emit update
    return jsonify(status='success', uid=uid)

@app.route('/stop_process/<string:uid>', methods=['POST'])
def stop_process(uid):
    manager.terminate_process(uid, force_kill=True)
    
    # Remove the process from collected_values
    if uid in collected_values:
        del collected_values[uid]
        
    send_active_processes()   # Emit update
    send_completed_processes()   # Emit update
    return jsonify(status='success', uid=uid)



@app.route('/stop_all_processes', methods=['POST'])
def stop_all():
    for uid in list(manager.processes.keys()):
        manager.terminate_process(uid, force_kill=True)
    return jsonify(status='success')


@app.route('/')
def index():
    return 'Server is running!'

@sio.on('connect')
def on_connect():
    send_active_processes()   # Emit initial data
    send_completed_processes()   # Emit initial data
    send_leaderboard() #Emit initial data
    logger.info("Client Connected - Server side")

@sio.on('get_leaderboard')
def send_leaderboard():
    sorted_leaderboard = sorted(manager.leaderboard.items(), key=lambda x: x[1]['return_value'], reverse=True)
    sio.emit('update_leaderboard', sorted_leaderboard[:5])

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

@sio.on('get_active_processes')
def send_active_processes():
    data = get_active_processes()
    sio.emit('update_active_processes', data)


@sio.on('get_completed_processes')
def send_completed_processes():
    # Extract and send completed processes info
    data = [{'uid': key[1], 'return_value': value['return_value']} for key, value in manager.leaderboard.items()]
    sio.emit('update_completed_processes', data)

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
        active_processes = get_active_processes()
        sio.emit('update_active_processes', active_processes)
        logger.info(f'{uid}: {collected_values[uid]}')
        if len(collected_values[uid]) == 3:
            average = sum(collected_values[uid]) / 3
            if average < -300000:
                logger.info(f"Average for {uid} too low, killing training")
                # Remove the process from collected_values
                if uid in collected_values:
                    del collected_values[uid]
                new_uid = manager.restart_process(uid)

    elif message_type == 'returns':
        current_config = manager.processes[uid].config if uid in manager.processes else {}
        write_processes_to_csv('processes.csv', SCRIPT_PATH, uid, value, cwd, current_config)
        if uid in manager.processes and manager.processes[uid].process.poll() is None:
            manager.terminate_process(uid)
        # Remove the process from collected_values
        if uid in collected_values:
            del collected_values[uid]
        new_uid = manager.start_process(SCRIPT_PATH)
        manager.update_leaderboard(SCRIPT_PATH, uid, value, cwd)
        sorted_leaderboard = sorted(manager.leaderboard.items(), key=lambda x: x[1]['return_value'], reverse=True)
        sio.emit('update_leaderboard', sorted_leaderboard[:5])

def send_progress_update():
    total_configurations = len(manager.configurations)
    progress_percentage = 100 * manager.current_config_index / total_configurations if total_configurations else 0
    sio.emit('update_progress', progress_percentage)

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
    
    with app.app_context():
        # SCRIPT_PATH = 'nd_trainer.py'
        SCRIPT_PATH = 'trainer.py'
        NUM_INSTANCES = 1
    
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
    sio.run(app, host="0.0.0.0", port=5678, debug=False)
