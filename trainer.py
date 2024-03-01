
from __future__ import annotations

from finrl.config import INDICATORS, SENTIMENT, ECONOMY, RETURNS, PRICE_MOVEMENT, VOLUME, DATE 
from finrl.config import DATA_SAVE_DIR, RESULTS_DIR, TENSORBOARD_LOG_DIR, TRAIN_END_DATE, TRAIN_START_DATE, TEST_END_DATE, TEST_START_DATE, TRAINED_MODEL_DIR, INTERM_RESULTS
from finrl.config_private import ALPACA_API_BASE_URL, ALPACA_API_KEY, ALPACA_API_SECRET, ALPHA_VANTAGE_KEY, EOD_KEY

from finrl.meta.env_stock_trading.env_stocktrading_nd import StockTradingEnv
from finrl.meta.data_processor import DataProcessor

import logging
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
import pickle
import os
import sys
import time
import gym
import numpy.random as rd
import requests
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal
from finrl.meta.data_processor import DataProcessor
import socketio
import argparse
import ast
from datetime import datetime
import hashlib
from tensorboard_writer import TensorBoardWriter
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

parser = argparse.ArgumentParser(description='Trainer Script')
    
parser.add_argument('--ticker_list', type=str, required=True, help='List of tickers')
parser.add_argument('--period_years', type=int, required=True, help='Period in years')
parser.add_argument('--steps', type=int, required=True, help='Number of steps')
parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
parser.add_argument('--net_dimensions', type=str, required=True, help='Comma-separated string of network dimensions, e.g. [128,64]')
parser.add_argument('--uid', required=True, help='Unique Identifier for this run')
parser.add_argument('--gpu_id', type=int, required=True, help='ID of GPU to be used')
parser.add_argument('--gamma', type=float, required=True, help='Gamma')
parser.add_argument('--horizon_length', type=int, required=True, help='Number of Env exploring Steps')
parser.add_argument('--lambda_gae_adv', type=float, required=True, help='Lambda GAE')
parser.add_argument('--lambda_entropy', type=float, required=True, help='Lambda Entropy')
parser.add_argument('--initial_capital', type=float, required=True, help='Agent Initial Capital for Trading')
arguments = parser.parse_args()

# Access the arguments as attributes of arguments
ticker_list = arguments.ticker_list.split(',')  # Convert the string back to a list
period_years = arguments.period_years
totalTimesteps = arguments.steps
learning_rate = arguments.learning_rate
batch_size = arguments.batch_size
netDimensions = ast.literal_eval(arguments.net_dimensions)  # Convert the string representation of list back to a list
script_uid = arguments.uid
gpuID = arguments.gpu_id
gamma = arguments.gamma
hlength = arguments.horizon_length
l_gae_adv = arguments.lambda_gae_adv
l_entropy = arguments.lambda_entropy
initial_capital = arguments.initial_capital
break_step = len(ticker_list) * 5e7

id_name = script_uid

def check_and_make_directories(directories: list[str]):
    for directory in directories:
        if not os.path.exists("./" + directory):
            os.makedirs("./" + directory)

CACHE_DIR = './cache'  # Specify your cache directory

check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR, INTERM_RESULTS])

writer = TensorBoardWriter.get_writer(log_dir=TENSORBOARD_LOG_DIR + '/' + script_uid)

if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
                logging.FileHandler('logs/' + id_name + ".log"),
                logging.StreamHandler()]
)
 
def subtract_years_from_date(date_str, period_years):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    new_date_obj = date_obj.replace(year=date_obj.year - period_years)
    return new_date_obj.strftime("%Y-%m-%d")

TRAIN_START_DATE = subtract_years_from_date(TRAIN_START_DATE, period_years=period_years)

if_nd = False

action_dim = len(ticker_list)

if if_nd:
    state_dim = 1 + 2 + 3 * action_dim + len(INDICATORS) * action_dim + len(SENTIMENT) * action_dim + len(ECONOMY) * action_dim + len(RETURNS) * action_dim + len(PRICE_MOVEMENT) * action_dim + len(VOLUME) * action_dim + len(DATE) * action_dim + 1
else:
    state_dim = 1 + 2 + 3 * action_dim + len(INDICATORS) * action_dim + 1

logging.info(f'''The State Dimension in Training is {state_dim},
             Initial: {1 + 2 + 3 * action_dim},
             Indicators (Tech): {len(INDICATORS) * action_dim},
             Sentiment: {len(SENTIMENT) * action_dim},
             Economy: {len(ECONOMY) * action_dim},
             Returns: {len(RETURNS) * action_dim},
             Price Movement: {len(PRICE_MOVEMENT) * action_dim},
             Volume: {len(VOLUME) * action_dim},
             Date: {len(DATE) * action_dim}''')

sio = socketio.Client()
script_name = id_name + '/' + id_name + '.py'  

@sio.event
def connect():
    sio.emit('join', {'uid': id_name}, callback=joined_room)
    logging.info(f'Connection established by {id_name}')
    # Rest of your code

def joined_room(data):
    logging.info("Successfully joined room!")
    sio.start_background_task(send_heartbeat)

@sio.on('terminate_yourself')
def on_terminate(data):
    if data and data.get('script') == id_name:
        logging.info(f"Received termination signal for {id_name}. Exiting...")
        sys.exit(0)

def send_heartbeat():
    try:
        while True:
            try:
                if sio.connected:
                    sio.emit('heartbeat', {'uid': id_name})
                    time.sleep(5)  # Send heartbeat every 3 seconds
            except socketio.exceptions.BadNamespaceError as e:
                logging.warning(f"Namespace error occurred: {e}")
    except Exception as e:
        logging.error(f"An error occurred in {id_name} send_heartbeat: {e}", exc_info=True)

@sio.event
def disconnect():
    logging.info(f'{id_name} - Disconnected from server')
    time.sleep(10)  # wait for 10 seconds
    if not sio.connected:
        try:
            sio.connect('http://localhost:5678')
        except Exception as e:
            logging.error(f"{id_name} Failed to reconnect: {e}", exc_info=True)

sio.connect('http://localhost:5678')

class ActorPPO(nn.Module):
    def __init__(self, dims: [int], state_dim: int, action_dim: int): # type: ignore
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])
        self.action_std_log = nn.Parameter(torch.zeros((1, action_dim)), requires_grad=True)  # trainable parameter

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state: Tensor) -> (Tensor, Tensor):  # for exploration # type: ignore
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = Normal(action_avg, action_std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        return action, logprob

    def get_logprob_entropy(self, state: Tensor, action: Tensor) -> (Tensor, Tensor): # type: ignore
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = Normal(action_avg, action_std)
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action: Tensor) -> Tensor:
        return action.tanh()


class CriticPPO(nn.Module):
    def __init__(self, dims: [int], state_dim: int, _action_dim: int): # type: ignore
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *dims, 1])

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)  # advantage value


def build_mlp(dims: [int]) -> nn.Sequential:  # MLP (MultiLayer Perceptron) # type: ignore
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    del net_list[-1]  # remove the activation of output layer
    return nn.Sequential(*net_list)


class Config:
    def __init__(self, agent_class=None, env_class=None, env_args=None):
        self.env_class = env_class  # env = env_class(**env_args)
        self.env_args = env_args  # env = env_class(**env_args)

        if env_args is None:  # dummy env_args
            env_args = {'env_name': None, 'state_dim': None, 'action_dim': None, 'if_discrete': None}
        self.env_name = env_args['env_name']  # the name of environment. Be used to set 'cwd'.
        self.state_dim = env_args['state_dim']  # vector dimension (feature number) of state
        self.action_dim = env_args['action_dim']  # vector dimension (feature number) of action
        self.if_discrete = env_args['if_discrete']  # discrete or continuous action space

        self.agent_class = agent_class  # agent = agent_class(...)

        '''Arguments for reward shaping'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 1.0  # an approximate target reward usually be closed to 256

        '''Arguments for training'''
        self.gpu_id = int(0)  # `int` means the ID of single GPU, -1 means CPU
        self.net_dims = (64, 32)  # the middle layer dimension of MLP (MultiLayer Perceptron)
        self.learning_rate = 6e-5  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 5e-3  # 2 ** -8 ~= 5e-3
        self.batch_size = int(128)  # num of transitions sampled from replay buffer.
        self.horizon_len = int(hlength)  # collect horizon_len step while exploring, then update network
        self.buffer_size = None  # ReplayBuffer size. Empty the ReplayBuffer for on-policy.
        self.repeat_times = 8.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
        self.seed = 312

        '''Arguments for evaluate'''
        self.cwd = None  # current working directory to save model. None means set automatically
        self.break_step = +np.inf  # break training if 'total_step > break_step'
        self.eval_times = int(32)  # number of times that get episodic cumulative return
        self.eval_per_step = int(2e4)  # evaluate the agent per training steps

    def init_before_training(self):
        if self.cwd is None:  # set cwd (current working directory) for saving model
            self.cwd = f'./{self.env_name}_{self.agent_class.__name__[5:]}'
        os.makedirs(self.cwd, exist_ok=True)


def get_gym_env_args(env, if_print: bool) -> dict:
    if {'unwrapped', 'observation_space', 'action_space', 'spec'}.issubset(dir(env)):  # isinstance(env, gym.Env):
        env_name = env.unwrapped.spec.id
        state_shape = env.observation_space.shape
        state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

        if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        if if_discrete:  # make sure it is discrete action space
            action_dim = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
            action_dim = env.action_space.shape[0]

    env_args = {'env_name': env_name, 'state_dim': state_dim, 'action_dim': action_dim, 'if_discrete': if_discrete}
    logging.info(f"env_args = {repr(env_args)}") if if_print else None
    return env_args


def kwargs_filter(function, kwargs: dict) -> dict:
    import inspect
    sign = inspect.signature(function).parameters.values()
    sign = {val.name for val in sign}
    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}  # filtered kwargs


def build_env(env_class=None, env_args=None):
    if env_class.__module__ == 'gym.envs.registration':  # special rule
        env = env_class(id=env_args['env_name'])
    else:
        env = env_class(**kwargs_filter(env_class.__init__, env_args.copy()))
    for attr_str in ('env_name', 'state_dim', 'action_dim', 'if_discrete'):
        setattr(env, attr_str, env_args[attr_str])
    return env


class AgentBase:
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()): # type: ignore
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.repeat_times = args.repeat_times
        self.reward_scale = args.reward_scale
        self.soft_update_tau = args.soft_update_tau

        self.states = None  # assert self.states == (1, state_dim)
    
        self.device = torch.device(f"cuda:{gpuID}" if (torch.cuda.is_available() and (gpuID >= 0)) else "cpu")

        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act = self.act_target = act_class(net_dims, state_dim, action_dim).to(self.device)
        self.cri = self.cri_target = cri_class(net_dims, state_dim, action_dim).to(self.device) \
            if cri_class else self.act

        self.act_optimizer = torch.optim.Adam(self.act.parameters(), args.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), args.learning_rate) \
            if cri_class else self.act_optimizer

        self.criterion = torch.nn.SmoothL1Loss()

    @staticmethod
    def optimizer_update(optimizer, objective: Tensor):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))


class AgentPPO(AgentBase):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()): # type: ignore
        self.if_off_policy = False
        self.act_class = getattr(self, "act_class", ActorPPO)
        self.cri_class = getattr(self, "cri_class", CriticPPO)
        AgentBase.__init__(self, net_dims, state_dim, action_dim, gpu_id, args)

        self.ratio_clip = getattr(args, "ratio_clip", 0.25)  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_gae_adv = getattr(args, "lambda_gae_adv", l_gae_adv)  # could be 0.80~0.99
        logging.info(f'Lambda_gae_adv: {self.lambda_gae_adv}')
        self.lambda_entropy = getattr(args, "lambda_entropy", l_entropy)  # could be 0.00~0.10
        logging.info(f'Lambda_entropy: {self.lambda_entropy}')
        self.lambda_entropy = torch.tensor(self.lambda_entropy, dtype=torch.float32, device=self.device)

    def explore_env(self, env, horizon_len: int) -> [Tensor]: # type: ignore
        states = torch.zeros((horizon_len, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.action_dim), dtype=torch.float32).to(self.device)
        logprobs = torch.zeros(horizon_len, dtype=torch.float32).to(self.device)
        rewards = torch.zeros(horizon_len, dtype=torch.float32).to(self.device)
        dones = torch.zeros(horizon_len, dtype=torch.bool).to(self.device)

        ary_state = self.states[0]

        get_action = self.act.get_action
        convert = self.act.convert_action_for_env
        for i in range(horizon_len):
            
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device)
            action, logprob = [t.squeeze(0) for t in get_action(state.unsqueeze(0))[:2]]

            ary_action = convert(action).detach().cpu().numpy()
            ary_state, reward, done, extra, _ = env.step(ary_action)
            if done:
                ary_state = env.reset()[0]

            states[i] = state
            actions[i] = action
            logprobs[i] = logprob
            rewards[i] = reward
            dones[i] = done

        self.states[0] = ary_state
        rewards = (rewards * self.reward_scale).unsqueeze(1)
        undones = (1 - dones.type(torch.float32)).unsqueeze(1)
        return states, actions, logprobs, rewards, undones

    def update_net(self, buffer) -> [float]: # type: ignore # type: ignore
        with torch.no_grad():
            states, actions, logprobs, rewards, undones = buffer
            buffer_size = states.shape[0]

            '''get advantages reward_sums'''
            bs = 2 ** 10  # set a smaller 'batch_size' when out of GPU memory.
            values = [self.cri(states[i:i + bs]) for i in range(0, buffer_size, bs)]
            values = torch.cat(values, dim=0).squeeze(1)  # values.shape == (buffer_size, )

            advantages = self.get_advantages(rewards, undones, values)  # advantages.shape == (buffer_size, )
            reward_sums = advantages + values  # reward_sums.shape == (buffer_size, )
            del rewards, undones, values

            advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-5)
        assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size,)

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(buffer_size * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for _ in range(update_times):
            indices = torch.randint(buffer_size, size=(self.batch_size,), requires_grad=False)
            state = states[indices]
            action = actions[indices]
            logprob = logprobs[indices]
            advantage = advantages[indices]
            reward_sum = reward_sums[indices]

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, reward_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = torch.min(surrogate1, surrogate2).mean()

            obj_actor = obj_surrogate + obj_entropy.mean() * self.lambda_entropy
            self.optimizer_update(self.act_optimizer, -obj_actor)

            obj_critics += obj_critic.item()
            obj_actors += obj_actor.item()
        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return obj_critics / update_times, obj_actors / update_times, a_std_log.item()

    def get_advantages(self, rewards: Tensor, undones: Tensor, values: Tensor) -> Tensor:
        advantages = torch.empty_like(values)  # advantage value

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        next_state = torch.tensor(self.states, dtype=torch.float32).to(self.device)
        next_value = self.cri(next_state).detach()[0, 0]

        advantage = 0  # last_gae_lambda
        for t in range(horizon_len - 1, -1, -1):
            delta = rewards[t] + masks[t] * next_value - values[t]
            advantages[t] = advantage = delta + masks[t] * self.lambda_gae_adv * advantage
            next_value = values[t]
        return advantages


class PendulumEnv(gym.Wrapper):  # a demo of custom gym env
    def __init__(self):
        gym.logger.set_level(40)  # Block warning
        gym_env_name = "Pendulum-v0" if gym.__version__ < '0.18.0' else "Pendulum-v1"
        super().__init__(env=gym.make(gym_env_name))

        '''the necessary env information when you design a custom env'''
        self.env_name = gym_env_name  # the name of this env.
        self.state_dim = self.observation_space.shape[0]  # feature number of state
        self.action_dim = self.action_space.shape[0]  # feature number of action
        self.if_discrete = False  # discrete action or continuous action

    def reset(self) -> np.ndarray:  # reset the agent in env
        return self.env.reset()

    def step(self, action: np.ndarray) -> (np.ndarray, float, bool, dict):  # type: ignore # agent interacts in env
        # We suggest that adjust action space to (-1, +1) when designing a custom env.
        state, reward, done, info_dict = self.env.step(action * 2)
        return state.reshape(self.state_dim), float(reward), done, info_dict

    
def train_agent(args: Config):
    args.init_before_training()

    env = build_env(args.env_class, args.env_args)
    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    agent.states = env.reset()[0][np.newaxis, :]


    evaluator = Evaluator(eval_env=build_env(args.env_class, args.env_args),
                          eval_per_step=args.eval_per_step,
                          eval_times=args.eval_times,
                          cwd=args.cwd)
    
    
    torch.set_grad_enabled(False)
    while True: # start training
        buffer_items = agent.explore_env(env, args.horizon_len)

        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer_items)
        torch.set_grad_enabled(False)

        evaluator.evaluate_and_save(agent.act ,args.horizon_len, logging_tuple)
        if (evaluator.total_step > args.break_step) or os.path.exists(f"{args.cwd}/stop"):
            torch.save(agent.act.state_dict(), args.cwd + '/actor.pth')
            writer.close()
            logging.info(f"Training Finished: {evaluator.total_step}" )
            break  # stop training when reach `break_step` or `mkdir cwd/stop`


def render_agent(env_class, env_args: dict, net_dims: [int], agent_class, actor_path: str, render_times: int = 8): # type: ignore
    env = build_env(env_class, env_args)
    state_dim = env_args['state_dim']
    action_dim = env_args['action_dim']
    agent = agent_class(net_dims, state_dim, action_dim, gpu_id=-1)
    actor = agent.act

    logging.info(f"| render and load actor from: {actor_path}")
    actor.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))
    for i in range(render_times):
        cumulative_reward, episode_step = get_rewards_and_steps(env, actor, if_render=True)
        logging.info(f"|{i:4}  cumulative_reward {cumulative_reward:9.3f}  episode_step {episode_step:5.0f}")

        
class Evaluator:
    def __init__(self, eval_env, eval_per_step: int = 1e4, eval_times: int = 8, cwd: str = '.'):
        self.cwd = cwd
        self.env_eval = eval_env
        self.eval_step = 0
        self.total_step = 0
        self.start_time = time.time()
        self.eval_times = eval_times  # number of times that get episodic cumulative return
        self.eval_per_step = eval_per_step  # evaluate the agent per training steps

        self.recorder = []
        logging.info(f"\n| `step`: Number of samples, or total training steps, or running times of `env.step()`."
              f"\n| `time`: Time spent from the start of training to this moment."
              f"\n| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode."
              f"\n| `stdR`: Standard dev of cumulative rewards, which is the sum of rewards in an episode."
              f"\n| `avgS`: Average of steps in an episode."
              f"\n| `objC`: Objective of Critic network. Or call it loss function of critic network."
              f"\n| `objA`: Objective of Actor network. It is the average Q value of the critic network."
              f"\n| {'step':>8}  {'time':>8}  | {'avgR':>8}  {'stdR':>6}  {'avgS':>6}  | {'objC':>8}  {'objA':>8}")
            
    def evaluate_and_save(self, actor, horizon_len: int, logging_tuple: tuple):
        self.total_step += horizon_len
        if self.eval_step + self.eval_per_step > self.total_step:
            if self.total_step % 100000 == 0:
                writer.add_scalar('Objective/Critic', logging_tuple[0], self.total_step)
                writer.add_scalar('Objective/Actor', logging_tuple[1], self.total_step)
            if self.total_step % 1000000 == 0:
                logging.info(f"UID: {script_uid} | Total Steps {self.total_step} | Critic Network Loss {logging_tuple[0]:8.2f} | Q Value {logging_tuple[1]:8.2f}")
            return
        
        self.eval_step = self.total_step

        rewards_steps_ary = [get_rewards_and_steps(self.env_eval, actor) for _ in range(self.eval_times)]
        rewards_steps_ary = np.array(rewards_steps_ary, dtype=np.float32)
        avg_r = rewards_steps_ary[:, 0].mean()  # average of cumulative rewards
        std_r = rewards_steps_ary[:, 0].std()  # std of cumulative rewards
        avg_s = rewards_steps_ary[:, 1].mean()  # average of steps in an episode

        used_time = time.time() - self.start_time
        self.recorder.append((self.total_step, used_time, avg_r))

        money_value = avg_r/2**-11

        avg_return = (money_value - initial_capital)/initial_capital
        
        logging.info(f"| {self.total_step:8.2e}  {used_time:8.0f}  "
                    f"| {avg_r:8.2f}  {std_r:6.2f}  {avg_s:6.0f}  "
                    f"| {logging_tuple[0]:8.2f}  {logging_tuple[1]:8.2f} "
                    f"| {money_value:8.2f}    {avg_return:8.2f}")
        
        writer.add_scalar('Scaled_Reward/Avg', avg_r, self.total_step)
        writer.add_scalar('Scaled_Reward/Std', std_r, self.total_step)
        writer.add_scalar('Money_Value', money_value, self.total_step)
        writer.add_scalar('Return/Avg', avg_return, self.total_step)
          
        sio.send({'uid': script_uid, 'type': 'training', 'value': float(avg_r)})


def get_rewards_and_steps(env, actor, if_render: bool = False) -> (float, int):  # cumulative_rewards and episode_steps # type: ignore
    device = next(actor.parameters()).device  # net.parameters() is a Python generator.

    state = env.reset()[0]
    episode_steps = 0
    cumulative_returns = 0.0  # sum of rewards in an episode
    for episode_steps in range(totalTimesteps):
        tensor_state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        tensor_action = actor(tensor_state)
        action = tensor_action.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
        state, reward, done, extra, _ = env.step(action)
        cumulative_returns += reward

        if if_render:
            env.render()
        if done:
            break
    return cumulative_returns, episode_steps + 1


# from elegantrl.agents import AgentA2C

MODELS = {"ppo": AgentPPO}
OFF_POLICY_MODELS = ["ddpg", "td3", "sac"]
ON_POLICY_MODELS = ["ppo"]
# MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}
#
# NOISE = {
#     "normal": NormalActionNoise,
#     "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
# }


class DRLAgent:
    """Implementations of DRL algorithms
    Attributes
    ----------
        env: gym environment class
            user-defined class
    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env, price_array, tech_array, turbulence_array, sentiment_array=None, economy_array=None, returns_array=None, price_movement_array=None, volume_array=None, date_array=None, if_nd=False):
        self.env = env
        self.price_array = price_array
        self.tech_array = tech_array
        self.turbulence_array = turbulence_array
        if if_nd:
            self.sentiment_array = sentiment_array
            self.economy_array = economy_array
            self.returns_array = returns_array
            self.price_movement_array = price_movement_array
            self.volume_array = volume_array
            self.date_array = date_array
        self.if_nd = if_nd


    def get_model(self, model_name, model_kwargs):

        if self.if_nd:
            env_config = {
                "price_array": self.price_array,
                "tech_array": self.tech_array,
                "turbulence_array": self.turbulence_array,
                "sentiment_array": self.sentiment_array,
                "economy_array": self.economy_array,
                "returns_array": self.returns_array,
                "price_movement_array": self.price_movement_array,
                "volume_array": self.volume_array,
                "date_array": self.date_array,
                "if_nd": self.if_nd,
                "initial_capital": initial_capital,
                "script_uid": script_uid,
                "if_train": True,
            }
        else:
             env_config = {
                "price_array": self.price_array,
                "tech_array": self.tech_array,
                "turbulence_array": self.turbulence_array,
                "if_nd": self.if_nd,
                "initial_capital": initial_capital,
                "script_uid": script_uid,
                "if_train": True,
            }
        environment = self.env(config=env_config)
        env_args = {'config': env_config,
              'env_name': environment.env_name,
              'state_dim': environment.state_dim,
              'action_dim': environment.action_dim,
              'if_discrete': False}
        agent = MODELS[model_name]
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        model = Config(agent_class=agent, env_class=self.env, env_args=env_args)
        model.if_off_policy = model_name in OFF_POLICY_MODELS
        if model_kwargs is not None:
            try:
                model.learning_rate = model_kwargs["learning_rate"]
                model.batch_size = model_kwargs["batch_size"]
                model.gamma = model_kwargs["gamma"]
                model.seed = model_kwargs["seed"]
                model.net_dims = model_kwargs["net_dimension"]
                model.break_step = model_kwargs["target_step"]
                model.eval_per_step = model_kwargs["eval_gap"]
                model.eval_times = model_kwargs["eval_times"]
            except BaseException:
                raise ValueError(
                    "Fail to read arguments, please check 'model_kwargs' input."
                )
        return model

    def train_model(self, model, cwd, total_timesteps=5000):
        model.cwd = cwd
        model.break_step = total_timesteps
        train_agent(model)

    @staticmethod
    def DRL_prediction(model_name, cwd, net_dimension, environment):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        agent_class = MODELS[model_name]
        environment.env_num = 1
        agent = agent_class(net_dimension, environment.state_dim, environment.action_dim)
        actor = agent.act
        # load agent
        try:  
            cwd = cwd + '/actor.pth'
            logging.info(f"| load actor from: {cwd}")
            actor.load_state_dict(torch.load(cwd, map_location=lambda storage, loc: storage))
            act = actor
            device = agent.device
        except BaseException:
            raise ValueError("Fail to load agent!")

        # test on the testing env
        _torch = torch
        state = environment.reset()[0]
        episode_returns = []  # the cumulative_return / initial_account
        episode_total_assets = [environment.initial_total_asset]
        with _torch.no_grad():
            for i in range(environment.max_step):
                s_tensor = _torch.as_tensor((state,), device=device)
                a_tensor = act(s_tensor)  # action_tanh = act.forward()
                action = (
                    a_tensor.detach().cpu().numpy()[0]
                )  # not need detach(), because with torch.no_grad() outside
                state, reward, done, extra, _ = environment.step(action)

                total_asset = (
                    environment.amount
                    + (
                        environment.price_ary[environment.day] * environment.stocks
                    ).sum()
                )
                episode_total_assets.append(total_asset)
                episode_return = total_asset / environment.initial_total_asset
                episode_returns.append(episode_return)
                if done:
                    break

        logging.info("Test Finished!")

        action_df = pd.DataFrame(environment.action_log)
        action_df.to_csv(f"{RESULTS_DIR}/{script_uid}_detailed_actions.csv", index=False)
        logging.info(f"Detailed action log saved to results/{script_uid}_detailed_actions.csv")

        # Plotting
        fig, axs = plt.subplots(4, 1, figsize=(15, 25), sharex=True)  # Increase the subplot count to 4

        # Plot 1: Executed Trading Actions
        for stock_index in sorted(action_df['stock_index'].unique()):
            df = action_df[action_df['stock_index'] == stock_index]
            axs[0].plot(df['day'], df['quantity'], label=f'Stock {stock_index}')
        axs[0].set_title('Executed Trading Actions Over Time')
        axs[0].set_ylabel('Executed Quantity')
        axs[0].legend()

        # Plot 2: Stock Holdings
        for stock_index in sorted(action_df['stock_index'].unique()):
            df = action_df[action_df['stock_index'] == stock_index]
            axs[1].plot(df['day'], df['stocks_held'], label=f'Stock {stock_index}')
        axs[1].set_title('Stock Holdings Over Time')
        axs[1].set_ylabel('Quantity Held')
        axs[1].legend()

        # Plot 3: Cash Amount and Portfolio Value
        axs[2].plot(action_df['day'], action_df['cash_amount'], label='Cash Amount', color='blue')
        axs[2].plot(action_df['day'], action_df['portfolio_value'], label='Portfolio Value', color='green')
        axs[2].set_title('Cash and Portfolio Value Over Time')
        axs[2].set_xlabel('Day')
        axs[2].set_ylabel('Value')
        axs[2].legend()

        # Plot 4: Correlation of Reward Value and Profit/Loss
        axs[3].scatter(action_df['Reward Value'], action_df['Profit/Loss'], alpha=0.5)
        axs[3].set_title('Correlation between Reward Value and Profit/Loss')
        axs[3].set_xlabel('Reward Value')
        axs[3].set_ylabel('Profit/Loss')
        axs[3].grid(True)

        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/{script_uid}_detailed_actions_plot.png")
        plt.close(fig)  # Close the figure to free memory
        logging.info(f"Graphical detailed action log saved to results/{script_uid}_detailed_actions_plot.png")

        # return episode total_assets on testing data
        logging.info(f"episode_return: {episode_return}")
        sio.send({'uid': script_uid, 'type': 'returns', 'cwd': cwd, 'value': float(episode_return)})
        return episode_total_assets

class TrainingTesting:

    def _generate_cache_key(self, tickers, start_date, end_date):
        combined_string = '_'.join(tickers) + f"_{start_date}_{end_date}"
        hashed_key = hashlib.md5(combined_string.encode()).hexdigest()
        return f"{hashed_key}.pkl"

    def _generate_sentiment_cache_key(self, tickers, start_date, end_date):
        sentiment_key = '_'.join(tickers) + f"_sentiment_{start_date}_{end_date}"
        hashed_key = hashlib.md5(sentiment_key.encode()).hexdigest()
        return f"{hashed_key}_sentiment.pkl"

    def _save_to_cache(self, data, cache_key):
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(CACHE_DIR, cache_key)
        with open(cache_file, 'wb') as file:
            pickle.dump(data, file)

    def _load_from_cache(self, cache_key):
        cache_file = os.path.join(CACHE_DIR, cache_key)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as file:
                return pickle.load(file)
        return None

    
    def train(
        self,
        start_date,
        end_date,
        ticker_list,
        data_source,
        time_interval,
        technical_indicator_list,
        drl_lib,
        env,
        model_name,
        if_vix=True,
        **kwargs,
    ):
        
        # Check if processed data is in cache
        dp = DataProcessor(data_source, tech_indicator=technical_indicator_list, **kwargs)
        processed_cache_key = self._generate_sentiment_cache_key(ticker_list, start_date, end_date)
        processed_data = self._load_from_cache(processed_cache_key)
        # Check if raw data is in cache
        raw_cache_key = self._generate_cache_key(ticker_list, start_date, end_date)
        raw_data = self._load_from_cache(raw_cache_key)

        if processed_data is None:           

            if raw_data is None:
                logging.info("Not using cache for raw data")
                data = dp.download_data(ticker_list, start_date, end_date, time_interval)
                data = dp.clean_data(data)
                print(data.head(50))
                print(data.tail(50))
                data = dp.add_technical_indicator(data, technical_indicator_list)
                print(data.head(50))
                print(data.tail(50))
          
                if if_vix:
                    data = dp.add_vix(data)
                else:
                    data = dp.add_turbulence(data)

                raw_data = data
                self._save_to_cache(raw_data, raw_cache_key)

            else:

                logging.info('Using cache for raw data')

            if if_nd:
                
                logging.info('Adding Returns and Directions')

            
                # Function to calculate increase/decrease and returns within each ticker group
                def calculate_metrics(group):
                    group['close'] = group['close'].ffill()  # Forward fill the 'close' column
                    group['returns'] = group['close'].pct_change()  # Now calculate the percent change
                    group['increase_decrease'] = np.where(group['returns'] > 0, 1, 0)
                    return group

                # Apply the function to each group without additional sorting
                grouped_data = raw_data.groupby('tic', group_keys=False).apply(calculate_metrics)

                # Fill NaN values in the entire DataFrame
                grouped_data = grouped_data.fillna(0)

                raw_data = grouped_data

                # Assuming 'raw_data' is your DataFrame and it has a column 'timestamp'
                # Ensure that your timestamp column is in pandas datetime format
                raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'])

                # Extract date-time features for all rows
                raw_data['year'] = raw_data['timestamp'].dt.year
                raw_data['month'] = raw_data['timestamp'].dt.month
                raw_data['day'] = raw_data['timestamp'].dt.day
                raw_data['weekday'] = raw_data['timestamp'].dt.weekday
                raw_data['hour'] = raw_data['timestamp'].dt.hour
                raw_data['minute'] = raw_data['timestamp'].dt.minute

                raw_data['hour_sin'] = np.sin((2 * np.pi * raw_data['hour']) / 24.0)  # Adjusted to 24
                raw_data['hour_cos'] = np.cos(2 * np.pi * raw_data['hour'] / 24.0)  # Adjusted to 24
                raw_data['day_sin'] = np.sin((2 * np.pi * raw_data['day']) / 31.0)  # 31 for days in a month
                raw_data['day_cos'] = np.cos(2 * np.pi * raw_data['day'] / 31.0)  # 31 for days in a month
                raw_data['month_sin'] = np.sin((2 * np.pi * raw_data['month']) / 12.0)  # 12 for months in a year
                raw_data['month_cos'] = np.cos(2 * np.pi * raw_data['month'] / 12.0)  # 12 for months in a year
                raw_data['minute_sin'] = np.sin((2 * np.pi * raw_data['minute']) / 60.0)  # Adjusted to 60
                raw_data['minute_cos'] = np.cos(2 * np.pi * raw_data['minute'] / 60.0)  # Adjusted to 60
                raw_data['weekday_sin'] = np.sin(2 * np.pi * raw_data['weekday'] / 7.0)  # Adjusted to 7
                
                def fetch_alpha_vantage_data(url):
                    response = requests.get(url)
                    if response.status_code == 200:
                        return response.json()
                    else:
                        return None
                        
                logging.info('Initalize data partition')

                # Initialize Dask Client
                client = Client(n_workers=4, threads_per_worker=8)

                if len(raw_data) < 5e6:
                    # Determine the number of partitions
                    npartitions = 5
                else:
                    npartitions = len(raw_data) // (5 * 10**5)  # For instance, one partition per 5 million rows

                # Convert pandas DataFrame to Dask DataFrame
                dask_data = dd.from_pandas(raw_data, npartitions=npartitions)

                def process_economic_indicator_partition(partition, av_data, column_name, frequency, start_date, end_date):
                    start_date = pd.to_datetime(start_date)
                    end_date = pd.to_datetime(end_date)
                    for record in av_data:
                        record_date = pd.to_datetime(record['date'])
                        if start_date <= record_date <= end_date:
                            if column_name == 'cpi':
                                value = float(record['value']) / 100 #scale the CPI value
                            else:
                                value = float(record['value'])
                            if frequency == 'annual':
                                mask = (partition['year'] == record_date.year) | (partition['year'] == record_date.year + 1)
                            elif frequency == 'monthly':
                                mask = (partition['year'] == record_date.year) & (partition['month'] == record_date.month)
                            elif frequency == 'daily':
                                mask = (partition['year'] == record_date.year) & (partition['month'] == record_date.month) & (partition['day'] == record_date.day)
                            
                            # Update values where mask is True
                            partition.loc[mask, column_name] = value

                    # Forward fill missing values
                    partition[column_name] = partition[column_name].ffill()
                    return partition



                def integrate_alpha_vantage_data(dask_data, av_data, column_name, frequency, start_date, end_date):
                    if column_name not in dask_data.columns:
                        dask_data[column_name] = 0.0
                    processed_data = dask_data.map_partitions(process_economic_indicator_partition, av_data, column_name, frequency, start_date, end_date)
                    return processed_data

                logging.info('Adding Economy Indicators')
                # API Key for Alpha Vantage
                av_api_key = ALPHA_VANTAGE_KEY
                logging.info("Adding CPI")
                # Consumer Price Index
                cpi_url = f"https://www.alphavantage.co/query?function=CPI&interval=monthly&apikey={av_api_key}"
                cpi_data = fetch_alpha_vantage_data(cpi_url)
                if cpi_data:
                    dask_data = integrate_alpha_vantage_data(dask_data, cpi_data['data'], 'cpi', cpi_data['interval'], start_date, end_date)
                logging.info("Adding Inflation Rate")
                # Inflation Rate
                inflation_url = f"https://www.alphavantage.co/query?function=INFLATION&apikey={av_api_key}"
                inflation_data = fetch_alpha_vantage_data(inflation_url)
                if inflation_data:
                    dask_data = integrate_alpha_vantage_data(dask_data, inflation_data['data'], 'inflation', inflation_data['interval'], start_date, end_date)
                logging.info("Adding Interest Rates")
                # Effective Federal Funds Rate
                fed_rate_url = f"https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval=daily&apikey={av_api_key}"
                fed_rate_data = fetch_alpha_vantage_data(fed_rate_url)
                if fed_rate_data:
                    dask_data = integrate_alpha_vantage_data(dask_data, fed_rate_data['data'], 'federal_funds_rate', fed_rate_data['interval'], start_date, end_date)
                logging.info("Unemployment Rate")
                # Unemployment Rate
                unemployment_url = f"https://alphavantage.co/query?function=UNEMPLOYMENT&apikey={av_api_key}"
                unemployment_data = fetch_alpha_vantage_data(unemployment_url)
                if unemployment_data:
                    dask_data = integrate_alpha_vantage_data(dask_data, unemployment_data['data'], 'unemployment_rate', unemployment_data['interval'], start_date, end_date)
                
                

                def fetch_sentiment_data(tickers, start_date, end_date, api_token):
                    # Join the tickers into a comma-separated string
                    formatted_tickers = ','.join(tickers)
                    url = f'https://eodhistoricaldata.com/api/sentiments?s={formatted_tickers}&from={start_date}&to={end_date}&api_token={api_token}&fmt=json'
                    response = requests.get(url)
                    sentiment_data = response.json()
                    return sentiment_data

                def process_partition(partition, ticker_sentiments):
                    for ticker, records in ticker_sentiments.items():
                        formatted_ticker = ticker.split('.')[0]  # Strip off '.US'
                        for record in records:
                            record_date = pd.to_datetime(record['date'])
                            #normalized_sentiment = record['count'] * record['normalized']
                            normalized_sentiment = record['normalized']
                            mask = (partition['tic'] == formatted_ticker) & \
                                (partition['year'] == record_date.year) & \
                                (partition['month'] == record_date.month) & \
                                (partition['day'] == record_date.day)
                            partition.loc[mask, 'sentiment'] = normalized_sentiment
                    return partition

                def integrate_sentiment_data(raw_data, sentiment_data, start_date, end_date):
                    start_date = pd.to_datetime(start_date)
                    end_date = pd.to_datetime(end_date)
                    ticker_sentiments = {ticker.split('.')[0]: [record for record in records if start_date <= pd.to_datetime(record['date']) <= end_date]
                                        for ticker, records in sentiment_data.items()}

                    if 'sentiment' not in raw_data.columns:
                        raw_data['sentiment'] = 0.0

                    processed_data = raw_data.map_partitions(process_partition, ticker_sentiments)
                    processed_data['sentiment'] = processed_data['sentiment'].fillna(0)
                    return processed_data

                # Example usage
                eod_api_key = EOD_KEY
                logging.info('Downloading Sentiment data')
                sentiment_data = fetch_sentiment_data(ticker_list, start_date, end_date, eod_api_key)

                # Process data using Dask with multi-threaded scheduler
                logging.info('Processing Sentiment data')
                processed_dask_data = integrate_sentiment_data(dask_data, sentiment_data, start_date, end_date)

                logging.info('Convert Dask DataFrame back to Pandas DataFrame in chunks')
                
                # Convert Dask DataFrame back to Pandas DataFrame in chunks
                
                import gc
                processed_data = pd.DataFrame()

                for chunk in processed_dask_data.to_delayed():
                    temp_df = chunk.compute()
                    processed_data = pd.concat([processed_data, temp_df])

                    # Optional: Clear memory
                    del temp_df
                    gc.collect()

                logging.info('Closing Dask Connections')
                #client.close()


                logging.info('Saving processed data cache')
                self._save_to_cache(processed_data, processed_cache_key)

        else:

            logging.info('Using cached processed data')
        
        logging.info('Configuring Environment')

        
        if if_nd:

            data = processed_data

        else:
            
            data = raw_data
        
        print(data.head(50))
        print(data.tail(50))
        # print(data.sample(20))

        num_rows = data.shape[0] / len(ticker_list)
        logging.info(f"Number of rows: {num_rows}")

        if if_nd:

            price_array, tech_array, turbulence_array, sentiment_array, economy_array, returns_array, price_movement_array, volume_array, date_array = dp.df_to_array(data, if_vix, if_nd)

        else:

            price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix, if_nd)

        if if_nd:
            env_config = {
                "price_array": price_array,
                "tech_array": tech_array,
                "turbulence_array": turbulence_array,
                "sentiment_array": sentiment_array,
                "economy_array": economy_array,
                "returns_array": returns_array,
                "price_movement_array": price_movement_array,
                "volume_array": volume_array,
                "date_array": date_array,
                "if_nd": if_nd,
                "initial_capital": initial_capital,
                "script_uid": script_uid,
                "if_train": True,
            }
        else:
            env_config = {
                "price_array": price_array,
                "tech_array": tech_array,
                "turbulence_array": turbulence_array,
                "if_nd": if_nd,
                "initial_capital": initial_capital,
                "script_uid": script_uid,
                "if_train": True,
            } 

        env_instance = env(config=env_config)

        # read parameters
        cwd = kwargs.get("cwd", "./" + str(model_name))

        
        DRLAgent_erl = DRLAgent
        break_step = kwargs.get("break_step", 1e6)
        erl_params = kwargs.get("erl_params")

        if if_nd:

            agent = DRLAgent_erl(
                env=env,
                price_array=price_array,
                tech_array=tech_array,
                turbulence_array=turbulence_array,
                sentiment_array=sentiment_array,
                returns_array=returns_array,
                economy_array=economy_array,
                price_movement_array=price_movement_array,
                volume_array=volume_array,
                date_array=date_array,
                if_nd=if_nd
            )
        else:
            agent = DRLAgent_erl(
                env=env,
                price_array=price_array,
                tech_array=tech_array,
                turbulence_array=turbulence_array,
                if_nd=if_nd
            )

        model = agent.get_model(model_name, model_kwargs=erl_params)
        logging.info('Training Started')
        trained_model = agent.train_model(
            model=model, cwd=cwd, total_timesteps=break_step
        )

    def test(
    self,
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    if_vix=True,
    **kwargs,
    ):
        dp = DataProcessor(data_source, tech_indicator=technical_indicator_list, **kwargs)
         # Check if processed data is in cache
        processed_cache_key = self._generate_sentiment_cache_key(ticker_list, start_date, end_date)
        processed_data = self._load_from_cache(processed_cache_key)
        # Check if raw data is in cache
        raw_cache_key = self._generate_cache_key(ticker_list, start_date, end_date)
        raw_data = self._load_from_cache(raw_cache_key)


        if processed_data is None:

            if raw_data is None:
                logging.info("Not using cache for raw data")
                data = dp.download_data(ticker_list, start_date, end_date, time_interval)
                data = dp.clean_data(data)
                data = dp.add_technical_indicator(data, technical_indicator_list)

                if if_vix:
                    data = dp.add_vix(data)
                else:
                    data = dp.add_turbulence(data)

                raw_data = data
                self._save_to_cache(raw_data, raw_cache_key)

            else:

                logging.info('Using cache for raw data')

            if if_nd:

                logging.info('Adding Returns and Directions')

                # Function to calculate increase/decrease and returns within each ticker group
                def calculate_metrics(group):
                    group['returns'] = group['close'].pct_change()
                    group['increase_decrease'] = np.where(group['returns'] > 0, 1, 0)
                    return group

                # Apply the function to each group without additional sorting
                grouped_data = raw_data.groupby('tic', group_keys=False).apply(calculate_metrics)

                # Fill NaN values in the entire DataFrame
                grouped_data = grouped_data.fillna(0)

                raw_data = grouped_data

                # Assuming 'raw_data' is your DataFrame and it has a column 'timestamp'
                # Ensure that your timestamp column is in pandas datetime format
                raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'])

                # Extract date-time features for all rows
                raw_data['year'] = raw_data['timestamp'].dt.year
                raw_data['month'] = raw_data['timestamp'].dt.month
                raw_data['day'] = raw_data['timestamp'].dt.day
                raw_data['weekday'] = raw_data['timestamp'].dt.weekday
                raw_data['hour'] = raw_data['timestamp'].dt.hour
                raw_data['minute'] = raw_data['timestamp'].dt.minute

                raw_data['hour_sin'] = np.sin((2 * np.pi * raw_data['hour']) / 24.0)  # Adjusted to 24
                raw_data['hour_cos'] = np.cos(2 * np.pi * raw_data['hour'] / 24.0)  # Adjusted to 24
                raw_data['day_sin'] = np.sin((2 * np.pi * raw_data['day']) / 31.0)  # 31 for days in a month
                raw_data['day_cos'] = np.cos(2 * np.pi * raw_data['day'] / 31.0)  # 31 for days in a month
                raw_data['month_sin'] = np.sin((2 * np.pi * raw_data['month']) / 12.0)  # 12 for months in a year
                raw_data['month_cos'] = np.cos(2 * np.pi * raw_data['month'] / 12.0)  # 12 for months in a year
                raw_data['minute_sin'] = np.sin((2 * np.pi * raw_data['minute']) / 60.0)  # Adjusted to 60
                raw_data['minute_cos'] = np.cos(2 * np.pi * raw_data['minute'] / 60.0)  # Adjusted to 60
                raw_data['weekday_sin'] = np.sin(2 * np.pi * raw_data['weekday'] / 7.0)  # Adjusted to 7
                
                def fetch_alpha_vantage_data(url):
                    response = requests.get(url)
                    if response.status_code == 200:
                        return response.json()
                    else:
                        return None
                        
                logging.info('Initalize data partition')

                # Initialize Dask Client
                client = Client(n_workers=4, threads_per_worker=8)

                if len(raw_data) < 5e6:
                    # Determine the number of partitions
                    npartitions = 5
                else:
                    npartitions = len(raw_data) // (5 * 10**5)  # For instance, one partition per 5 million rows

                # Convert pandas DataFrame to Dask DataFrame
                dask_data = dd.from_pandas(raw_data, npartitions=npartitions)

                def process_economic_indicator_partition(partition, av_data, column_name, frequency, start_date, end_date):
                    start_date = pd.to_datetime(start_date)
                    end_date = pd.to_datetime(end_date)
                    for record in av_data:
                        record_date = pd.to_datetime(record['date'])
                        if start_date <= record_date <= end_date:
                            if column_name == 'cpi':
                                value = float(record['value']) / 100 #scale the CPI value
                            else:
                                value = float(record['value'])
                            if frequency == 'annual':
                                mask = (partition['year'] == record_date.year) | (partition['year'] == record_date.year + 1)
                            elif frequency == 'monthly':
                                mask = (partition['year'] == record_date.year) & (partition['month'] == record_date.month)
                            elif frequency == 'daily':
                                mask = (partition['year'] == record_date.year) & (partition['month'] == record_date.month) & (partition['day'] == record_date.day)
                            
                            # Update values where mask is True
                            partition.loc[mask, column_name] = value

                    # Forward fill missing values
                    partition[column_name] = partition[column_name].ffill()
                    return partition



                def integrate_alpha_vantage_data(dask_data, av_data, column_name, frequency, start_date, end_date):
                    if column_name not in dask_data.columns:
                        dask_data[column_name] = 0.0

                    processed_data = dask_data.map_partitions(process_economic_indicator_partition, av_data, column_name, frequency, start_date, end_date)
                    return processed_data

                logging.info('Adding Economy Indicators')
                # API Key for Alpha Vantage
                av_api_key = ALPHA_VANTAGE_KEY
                logging.info("Adding CPI")
                # Consumer Price Index
                cpi_url = f"https://www.alphavantage.co/query?function=CPI&interval=monthly&apikey={av_api_key}"
                cpi_data = fetch_alpha_vantage_data(cpi_url)
                if cpi_data:
                    dask_data = integrate_alpha_vantage_data(dask_data, cpi_data['data'], 'cpi', cpi_data['interval'], start_date, end_date)
                logging.info("Adding Inflation Rate")
                # Inflation Rate
                inflation_url = f"https://www.alphavantage.co/query?function=INFLATION&apikey={av_api_key}"
                inflation_data = fetch_alpha_vantage_data(inflation_url)
                if inflation_data:
                    dask_data = integrate_alpha_vantage_data(dask_data, inflation_data['data'], 'inflation', inflation_data['interval'], start_date, end_date)
                logging.info("Adding Interest Rates")
                # Effective Federal Funds Rate
                fed_rate_url = f"https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval=daily&apikey={av_api_key}"
                fed_rate_data = fetch_alpha_vantage_data(fed_rate_url)
                if fed_rate_data:
                    dask_data = integrate_alpha_vantage_data(dask_data, fed_rate_data['data'], 'federal_funds_rate', fed_rate_data['interval'], start_date, end_date)
                logging.info("Unemployment Rate")
                # Unemployment Rate
                unemployment_url = f"https://alphavantage.co/query?function=UNEMPLOYMENT&apikey={av_api_key}"
                unemployment_data = fetch_alpha_vantage_data(unemployment_url)
                if unemployment_data:
                    dask_data = integrate_alpha_vantage_data(dask_data, unemployment_data['data'], 'unemployment_rate', unemployment_data['interval'], start_date, end_date)
                

                def fetch_sentiment_data(tickers, start_date, end_date, api_token):
                    # Join the tickers into a comma-separated string
                    formatted_tickers = ','.join(tickers)
                    url = f'https://eodhistoricaldata.com/api/sentiments?s={formatted_tickers}&from={start_date}&to={end_date}&api_token={api_token}&fmt=json'
                    response = requests.get(url)
                    sentiment_data = response.json()
                    return sentiment_data

                def process_partition(partition, ticker_sentiments):
                    for ticker, records in ticker_sentiments.items():
                        formatted_ticker = ticker.split('.')[0]  # Strip off '.US'
                        for record in records:
                            record_date = pd.to_datetime(record['date'])
                            #normalized_sentiment = record['count'] * record['normalized']
                            normalized_sentiment = record['normalized']
                            mask = (partition['tic'] == formatted_ticker) & \
                                (partition['year'] == record_date.year) & \
                                (partition['month'] == record_date.month) & \
                                (partition['day'] == record_date.day)
                            partition.loc[mask, 'sentiment'] = normalized_sentiment
                
                    return partition

                def integrate_sentiment_data(raw_data, sentiment_data, start_date, end_date):
                    start_date = pd.to_datetime(start_date)
                    end_date = pd.to_datetime(end_date)
                    ticker_sentiments = {ticker.split('.')[0]: [record for record in records if start_date <= pd.to_datetime(record['date']) <= end_date]
                                        for ticker, records in sentiment_data.items()}

                    if 'sentiment' not in raw_data.columns:
                        raw_data['sentiment'] = 0.0

                    processed_data = raw_data.map_partitions(process_partition, ticker_sentiments)
                    processed_data['sentiment'] = processed_data['sentiment'].fillna(0)
                    return processed_data

                # Example usage
                api_token = EOD_KEY
                logging.info('Downloading Sentiment data')
                sentiment_data = fetch_sentiment_data(ticker_list, start_date, end_date, api_token)

                # Process data using Dask with multi-threaded scheduler
                logging.info('Processing Sentiment data')
                processed_dask_data = integrate_sentiment_data(dask_data, sentiment_data, start_date, end_date)

                # Convert Dask DataFrame back to Pandas DataFrame in chunks
                #processed_data = pd.concat([part.compute() for part in processed_dask_data.to_delayed()])

                #processed_data = processed_dask_data.compute()
                logging.info('Convert Dask DataFrame back to Pandas DataFrame in chunks')
                # Convert Dask DataFrame back to Pandas DataFrame in chunks
                chunk_size = 5 * 10**6  # Adjust this based on your memory capacity
                import gc
                processed_data = pd.DataFrame()

                for chunk in processed_dask_data.to_delayed():
                    temp_df = chunk.compute()
                    processed_data = pd.concat([processed_data, temp_df])

                    # Optional: Clear memory
                    del temp_df
                    gc.collect()

                logging.info('Closing Dask Connections')
                #client.close()

                # Replace None with zero in the sentiment column
                #processed_data['sentiment'].fillna(0, inplace=True)

                logging.info('Saving processed data cache')
                self._save_to_cache(processed_data, processed_cache_key)

        else:

            logging.info('Using cached processed data')
        
        logging.info('Configuring Environment')

        if if_nd:
            data = processed_data
        else:
            data = raw_data

        if if_nd:

            price_array, tech_array, turbulence_array, sentiment_array, economy_array, returns_array, price_movement_array, volume_array, date_array = dp.df_to_array(data, if_vix, if_nd)

        else:

            price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix, if_nd)


        if if_nd:
            env_config = {
                "price_array": price_array,
                "tech_array": tech_array,
                "turbulence_array": turbulence_array,
                "sentiment_array": sentiment_array,
                "economy_array": economy_array,
                "returns_array": returns_array,
                "price_movement_array": price_movement_array,
                "volume_array": volume_array,
                "date_array": date_array,
                "if_nd": if_nd,
                "initial_capital": initial_capital,
                "script_uid": script_uid,
                "if_train": False,
            }
        else:
            env_config = {
                "price_array": price_array,
                "tech_array": tech_array,
                "turbulence_array": turbulence_array,
                "if_nd": if_nd,
                "initial_capital": initial_capital,
                "script_uid": script_uid,
                "if_train": False,
            }

        env_instance = env(config=env_config)

        # load elegantrl needs state dim, action dim and net dim
        net_dimension = kwargs.get("net_dimension", 2**7)
        cwd = kwargs.get("cwd", "./" + str(model_name))
        logging.info(f"price_array: {len(price_array)}")

        
        DRLAgent_erl = DRLAgent
        episode_total_assets = DRLAgent_erl.DRL_prediction(
            model_name=model_name,
            cwd=cwd,
            net_dimension=net_dimension,
            environment=env_instance,
        )
        return episode_total_assets

env = StockTradingEnv

ERL_PARAMS = {"learning_rate": learning_rate,"batch_size": batch_size,"gamma":  gamma,
        "seed":312,"net_dimension":netDimensions, "target_step":totalTimesteps, "eval_gap":break_step // 10,
        "eval_times":4} 


if __name__ == '__main__':
    
    trainTest = TrainingTesting()


    trainTest.train(start_date = TRAIN_START_DATE, 
        end_date = TRAIN_END_DATE,
        ticker_list = ticker_list, 
        data_source = 'alpaca',
        time_interval= '5Min', 
        technical_indicator_list= INDICATORS,
        drl_lib='elegantrl', 
        env=env,
        model_name='ppo',
        if_vix=True, 
        API_KEY = ALPACA_API_KEY, 
        API_SECRET = ALPACA_API_SECRET, 
        API_BASE_URL = ALPACA_API_BASE_URL,
        ALPHAVANTAGE = ALPHA_VANTAGE_KEY,
        EOD = EOD_KEY,
        erl_params=ERL_PARAMS,
        cwd='./trained_models/' + str(script_uid) + '-' + str(ERL_PARAMS["net_dimension"]) + '-' + TRAIN_START_DATE,  #current_working_dir,
        break_step=break_step)
    


    account_value_erl=trainTest.test(start_date = TEST_START_DATE, 
                        end_date = TEST_END_DATE,
                        ticker_list = ticker_list, 
                        data_source = 'alpaca',
                        time_interval= '5Min', 
                        technical_indicator_list= INDICATORS,
                        drl_lib='elegantrl', 
                        env=env, 
                        model_name='ppo',
                        if_vix=True, 
                        API_KEY = ALPACA_API_KEY, 
                        API_SECRET = ALPACA_API_SECRET, 
                        API_BASE_URL = ALPACA_API_BASE_URL,
                        ALPHAVANTAGE = ALPHA_VANTAGE_KEY,
                        EOD = EOD_KEY,
                        cwd='./trained_models/' + str(script_uid) + '-' + str(ERL_PARAMS["net_dimension"]) + '-' + TRAIN_START_DATE,  #current_working_dir,
                        net_dimension = ERL_PARAMS['net_dimension'])
   
