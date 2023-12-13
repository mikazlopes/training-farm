
from __future__ import annotations

API_KEY = "PKPVXMUY5SEZI92CVN2U"
API_SECRET = "QpiszkVxetaOhYCPEhafggIGU6CaIV8gbmIXhaFu"
API_BASE_URL = 'https://paper-api.alpaca.markets'
data_url = 'wss://data.alpaca.markets'

from finrl.config import INDICATORS
from finrl.config_tickers import DRL_ALGO_TICKERS
import optuna
from optuna.trial import TrialState
import multiprocessing
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.meta.env_stock_trading.env_stock_papertrading import AlpacaPaperTrading
from finrl.meta.data_processor import DataProcessor
import logging
import numpy as np
import pickle
import os
import time
import gym
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal
from finrl.meta.data_processor import DataProcessor
import argparse
from datetime import datetime
import hashlib



parser = argparse.ArgumentParser(description='Trainer Script')
    
parser.add_argument('--period_years', type=int, required=True, help='Period in years')
parser.add_argument('--gpu_id', type=int, required=True, help='ID of GPU to be used')

args = parser.parse_args()

# Access the arguments as attributes of args
ticker_list = DRL_ALGO_TICKERS
period_years = args.period_years
totalTimesteps = period_years * 100000

id_name = 'hp-tuner'

gpuID = args.gpu_id

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

TRAIN_START_DATE = subtract_years_from_date("2022-01-01", period_years=period_years)
TRAIN_END_DATE = '2023-06-30'
TEST_START_DATE = '2023-07-01'
TEST_END_DATE = '2023-12-11'

action_dim = len(ticker_list)
state_dim = 1 + 2 + 3 * action_dim + len(INDICATORS) * action_dim


class ActorPPO(nn.Module):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])
        self.action_std_log = nn.Parameter(torch.zeros((1, action_dim)), requires_grad=True)  # trainable parameter

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state: Tensor) -> (Tensor, Tensor):  # for exploration
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = Normal(action_avg, action_std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        return action, logprob

    def get_logprob_entropy(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
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
    def __init__(self, dims: [int], state_dim: int, _action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *dims, 1])

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)  # advantage value


def build_mlp(dims: [int]) -> nn.Sequential:  # MLP (MultiLayer Perceptron)
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
        self.horizon_len = int(2000)  # collect horizon_len step while exploring, then update network
        self.buffer_size = None  # ReplayBuffer size. Empty the ReplayBuffer for on-policy.
        self.repeat_times = 8.0  # repeatedly update network using ReplayBuffer to keep critic's loss small

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
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
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
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.if_off_policy = False
        self.act_class = getattr(self, "act_class", ActorPPO)
        self.cri_class = getattr(self, "cri_class", CriticPPO)
        AgentBase.__init__(self, net_dims, state_dim, action_dim, gpu_id, args)

        self.ratio_clip = getattr(args, "ratio_clip", 0.25)  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_gae_adv = getattr(args, "lambda_gae_adv", 0.95)  # could be 0.80~0.99
        self.lambda_entropy = getattr(args, "lambda_entropy", 0.01)  # could be 0.00~0.10
        self.lambda_entropy = torch.tensor(self.lambda_entropy, dtype=torch.float32, device=self.device)

    def explore_env(self, env, horizon_len: int) -> [Tensor]:
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

    def update_net(self, buffer) -> [float]:
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

    def step(self, action: np.ndarray) -> (np.ndarray, float, bool, dict):  # agent interacts in env
        # We suggest that adjust action space to (-1, +1) when designing a custom env.
        state, reward, done, info_dict = self.env.step(action * 2)
        return state.reshape(self.state_dim), float(reward), done, info_dict

    
def train_agent(args: Config, trial):
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

        evaluator.evaluate_and_save(agent.act, args.horizon_len, logging_tuple, trial)
        if (evaluator.total_step > args.break_step) or os.path.exists(f"{args.cwd}/stop"):
            torch.save(agent.act.state_dict(), args.cwd + '/actor.pth')
            last_avg_r = evaluator.recorder[-1][-1]
            break
    
    return last_avg_r
        


def render_agent(env_class, env_args: dict, net_dims: [int], agent_class, actor_path: str, render_times: int = 8):
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
            
    def evaluate_and_save(self, actor, horizon_len: int, logging_tuple: tuple, trial):
        self.total_step += horizon_len
        if self.eval_step + self.eval_per_step > self.total_step:
            return
        self.eval_step = self.total_step

        rewards_steps_ary = [get_rewards_and_steps(self.env_eval, actor) for _ in range(self.eval_times)]
        rewards_steps_ary = np.array(rewards_steps_ary, dtype=np.float32)
        avg_r = rewards_steps_ary[:, 0].mean()  # average of cumulative rewards
        std_r = rewards_steps_ary[:, 0].std()  # std of cumulative rewards
        avg_s = rewards_steps_ary[:, 1].mean()  # average of steps in an episode

        used_time = time.time() - self.start_time
        self.recorder.append((self.total_step, used_time, avg_r))
        
        logging.info(f"| {self.total_step:8.2e}  {used_time:8.0f}  "
              f"| {avg_r:8.2f}  {std_r:6.2f}  {avg_s:6.0f}  "
              f"| {logging_tuple[0]:8.2f}  {logging_tuple[1]:8.2f}")
        
        trial.report(avg_r, self.total_step)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()


def get_rewards_and_steps(env, actor, if_render: bool = False) -> (float, int):  # cumulative_rewards and episode_steps
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

    def __init__(self, env, price_array, tech_array, turbulence_array):
        self.env = env
        self.price_array = price_array
        self.tech_array = tech_array
        self.turbulence_array = turbulence_array

    def get_model(self, model_name, model_kwargs):
        env_config = {
            "price_array": self.price_array,
            "tech_array": self.tech_array,
            "turbulence_array": self.turbulence_array,
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
                model.target_step = model_kwargs["target_step"]
                model.eval_gap = model_kwargs["eval_gap"]
                model.eval_times = model_kwargs["eval_times"]
            except BaseException:
                raise ValueError(
                    "Fail to read arguments, please check 'model_kwargs' input."
                )
        return model

    def train_model(self, model, cwd, trial, total_timesteps=5000):
        model.cwd = cwd
        model.break_step = total_timesteps
        final_reward = train_agent(model, trial)
        return final_reward

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
        # return episode total_assets on testing data
        logging.info(f"episode_return: {episode_return}")
        #sio.send({'uid': script_uid, 'type': 'returns', 'cwd': cwd, 'value': float(episode_return)})
        return episode_return

class TrainingTesting:

    def __init__(
            self,
            train_start_date,
            train_end_date,
            test_start_date,
            test_end_date,
            ticker_list,
            data_source,
            time_interval,
            technical_indicator_list,
            drl_lib,
            env,
            model_name,
            if_vix=True,
            n_trials=100,
            **kwargs,
        ):
        self.CACHE_DIR = './cache'  # Specify your cache directory
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.ticker_list = ticker_list
        self.data_source = data_source
        self.time_interval = time_interval
        self.technical_indicator_list = technical_indicator_list
        self.drl_lib = drl_lib
        self.env = env
        self.model_name = model_name
        self.if_vix = if_vix
        self.n_trials = n_trials
        self.kwargs = kwargs


    def _generate_cache_key(self, tickers, start_date, end_date):
        # Concatenate tickers with start and end dates
        combined_string = '_'.join(tickers) + f"_{start_date}_{end_date}"
        # Create a hash of the combined string
        hashed_key = hashlib.md5(combined_string.encode()).hexdigest()
        # Return the hash with a .pkl extension
        return f"{hashed_key}.pkl"

    def _save_to_cache(self, data, cache_key):
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(self.CACHE_DIR, cache_key)
        with open(cache_file, 'wb') as file:
            pickle.dump(data, file)

    def _load_from_cache(self, cache_key):
        cache_file = os.path.join(self.CACHE_DIR, cache_key)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as file:
                return pickle.load(file)
        return None
    
    def objective(self, trial):
        # Suggest hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 4e-6, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024, 2048, 4096])
        gamma = trial.suggest_float('gamma', 0.9, 0.9999)
        net_dimension = trial.suggest_categorical('net_dimension', ['128,64', '256,128', '512,256', '1024,512', '128,64,32', '256,128,64', '512,256,128', '1024,512,256'])
        break_step = trial.suggest_int('target_step', low=100000, high=1000000, step=20000)
        eval_gap = trial.suggest_int('eval_gap', low=10, high=60, step=10)
        eval_times = trial.suggest_int('eval_times', low=16, high=64, step=16)

        # Set up model kwargs with the suggested hyperparameters
        model_kwargs = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gamma": gamma,
            "net_dimension": net_dimension,
            "target_step": break_step,
            "seed": 312,
            "eval_gap": eval_gap,
            "eval_times": eval_times,
            # ... add other hyperparameters here ...
        }

        return self.train_with_config(model_kwargs, trial)
    
    def train_with_config(self, model_kwargs, trial):
        # Initialize your environment and agent here as per your normal training process
        # First, try to load the data from cache
        if 'net_dimension' in model_kwargs:
            model_kwargs['net_dimension'] = tuple(map(int, model_kwargs['net_dimension'].split(',')))

        cache_key = self._generate_cache_key(ticker_list, self.train_start_date, self.train_end_date)
        data = self._load_from_cache(cache_key)
        dp = DataProcessor(self.data_source,tech_indicator=self.technical_indicator_list, **self.kwargs)
        
        # If data is not in cache, download and treat it
        if data is None:
        # fetch data
            # download data
            print("Not using cache")
            data = dp.download_data(ticker_list, self.train_start_date, self.train_end_date, self.time_interval)
            data = dp.clean_data(data)
            data = dp.add_technical_indicator(data, self.technical_indicator_list)
            if self.if_vix:
                data = dp.add_vix(data)
            else:
                data = dp.add_turbulence(data)

            # Save the treated data to cache for future use
            self._save_to_cache(data, cache_key)

        else:
            logging.info(f'{id_name} using cache')
        
        price_array, tech_array, turbulence_array = dp.df_to_array(data, self.if_vix)
        env_config = {
            "price_array": price_array,
            "tech_array": tech_array,
            "turbulence_array": turbulence_array,
            "if_train": True,
        }

        env_instance = env(config=env_config)

        # read parameters
        cwd = './optuna/trial' + str(trial.number) + '-' + str(model_kwargs.get("net_dimension"))
      
        env_instance = env(config=env_config)
        DRLAgent_erl = DRLAgent
        # Use model_kwargs directly
        agent = DRLAgent_erl(
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
        )
        model = agent.get_model(self.model_name, model_kwargs=model_kwargs)
        trained_model_reward = agent.train_model(
            model=model, cwd=cwd, trial=trial, total_timesteps=model_kwargs.get("target_step", 1e6)
        )

        test_model_reward = self.test(cwd, **model_kwargs)

        return test_model_reward      
        
    
    def optimize_hyperparameters(self):
        study_name = "FinRL-HP"
        storage_url = "mysql+mysqlconnector://optuna_user:r00t4dm1n@localhost/optuna_example"
        study = optuna.create_study(
            study_name=study_name,
            load_if_exists=True,
            storage=storage_url,
            direction='maximize',
            pruner=optuna.pruners.MedianPruner()
        )
        study.optimize(self.objective, n_trials=self.n_trials)

        # Output the optimization results
        trial = study.best_trial
        print(f'Best trial value: {trial.value}')
        print('Best hyperparameters:')
        for key, value in trial.params.items():
            print(f'{key}: {value}')
    

    def test(
    self,
    cwd,
    **kwargs,
    ):

         # First, try to load the data from cache
        cache_key = self._generate_cache_key(self.ticker_list, self.test_start_date, self.test_end_date)
        data = self._load_from_cache(cache_key)
        dp = DataProcessor(self.data_source, tech_indicator=self.technical_indicator_list, **self.kwargs)
        
        # If data is not in cache, download and treat it
        if data is None:
        # fetch data
            # download data
            print("Not using cache")
            data = dp.download_data(self.ticker_list, self.test_start_date, self.test_end_date, self.time_interval)
            data = dp.clean_data(data)
            data = dp.add_technical_indicator(data, self.technical_indicator_list)
            if self.if_vix:
                data = dp.add_vix(data)
            else:
                data = dp.add_turbulence(data)

            # Save the treated data to cache for future use
            self._save_to_cache(data, cache_key)
        
        price_array, tech_array, turbulence_array = dp.df_to_array(data, self.if_vix)

        # Save the treated data to cache for future use
        self._save_to_cache(data, cache_key)

        env_config = {
            "price_array": price_array,
            "tech_array": tech_array,
            "turbulence_array": turbulence_array,
            "if_train": False,
        }
        env_instance = env(config=env_config)

        # load elegantrl needs state dim, action dim and net dim
        net_dimension = kwargs.get("net_dimension", 2**7)

        DRLAgent_erl = DRLAgent
        episode_total_assets = DRLAgent_erl.DRL_prediction(
            model_name=self.model_name,
            cwd=cwd,
            net_dimension=net_dimension,
            environment=env_instance,
        )
        return episode_total_assets

env = StockTradingEnv

def run_optimization():
    trainTest = TrainingTesting(train_start_date = TRAIN_START_DATE, 
        train_end_date = TRAIN_END_DATE,
        test_start_date = TEST_START_DATE,
        test_end_date = TEST_END_DATE, 
        ticker_list = ticker_list, 
        data_source = 'alpaca',
        time_interval= '1Min', 
        technical_indicator_list= INDICATORS,
        drl_lib='elegantrl', 
        env=env,
        model_name='ppo',
        if_vix=True,
        n_trials=200, 
        API_KEY = API_KEY, 
        API_SECRET = API_SECRET, 
        API_BASE_URL = API_BASE_URL)

    trainTest.optimize_hyperparameters()


def run_multiprocessing():
    process_count = 2  # Number of processes to run in parallel
    processes = []

    for _ in range(process_count):
        p = multiprocessing.Process(target=run_optimization)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    run_multiprocessing()
