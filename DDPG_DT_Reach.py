import gym
import panda_gym
import os
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from stable_baselines3 import DDPG, HerReplayBuffer

models_dir = "models/DDPG_Reach"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make("PandaReach-v2", render=True)
env.reset()

# Train a decision tree for feature extraction
num_samples = 10000
state_dim = env.observation_space["observation"].shape[0]
action_dim = env.action_space.shape[0]

states = np.zeros((num_samples, state_dim))
actions = np.zeros((num_samples, action_dim))
next_states = np.zeros((num_samples, state_dim))

state = env.reset()
for i in range(num_samples):
    action = env.action_space.sample()
    next_state, _, done, _ = env.step(action)
    states[i] = state["observation"]
    actions[i] = action
    next_states[i] = next_state["observation"]
    state = next_state if not done else env.reset()

X = np.concatenate([states, actions], axis=1)
y = next_states
decision_tree = DecisionTreeRegressor()
decision_tree.fit(X, y)

# Custom environment wrapper to add decision tree output to observations
class DecisionTreeWrapper(gym.Wrapper):
    def __init__(self, env, dt_model):
        super(DecisionTreeWrapper, self).__init__(env)
        self.dt_model = dt_model

    def _preprocess_observation(self, observation):
        return observation['observation'][:6]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        preprocessed_obs = self._preprocess_observation(obs)
        new_obs = {'observation': preprocessed_obs, 'desired_goal': obs['desired_goal'], 'achieved_goal': obs['achieved_goal']}
        return new_obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        preprocessed_obs = self._preprocess_observation(obs)
        return {'observation': preprocessed_obs, 'desired_goal': obs['desired_goal'], 'achieved_goal': obs['achieved_goal']}
        
wrapped_env = DecisionTreeWrapper(env, decision_tree)

model = DDPG(policy="MultiInputPolicy", env=env, replay_buffer_class=HerReplayBuffer, verbose=1, tensorboard_log=logdir, train_freq=100)

TIMESTEPS = 10000
for i in range(1, 30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DDPG_Reach_DT")
    model.save(f'{models_dir}/{TIMESTEPS*i}')
        
episodes = 10

for ep in range(episodes):
    obs = wrapped_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = wrapped_env.step(action)

wrapped_env = DecisionTreeWrapper(env, dt_model)

# Evaluate the agent
episode_reward = 0
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, info = wrapped_env.step(action)
    wrapped_env.render()
    episode_reward += reward
    if terminated or info.get("is_success", False):
        print("Reward:", episode_reward, "Success?", info.get("is_success", False))
        episode_reward = 0.0
        obs = wrapped_env.reset()

wrapped_env.close()