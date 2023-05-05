import gym
import panda_gym
import time
import numpy as np
from tensorboardX import SummaryWriter

env = gym.make("PandaReach-v2", render=True)

Kp = 10
Ki = 0
Kd = 1

# Set up the logger
log_writer = SummaryWriter(logdir='logs/PID_Reach')

episode_count = 0
num_episodes = 1000
episode_reward = 0
episode_length = 0
total_episode_rewards = []
total_episode_lengths = []
success_count = 0

start_time = time.time()
for ep in range(num_episodes):
    obs = env.reset()
    error_prev = obs["desired_goal"] - obs["achieved_goal"]
    integral = np.zeros(3)

    while True:
        error = obs["desired_goal"] - obs["achieved_goal"]
        derivative = error - error_prev
        integral += error
        action = Kp * error + Ki * integral + Kd * derivative

        obs, reward, done, info = env.step(action)
        error_prev = error
        episode_reward += reward
        episode_length += 1

        if done or info.get("is_success", False):
            episode_count += 1
            total_episode_rewards.append(episode_reward)
            total_episode_lengths.append(episode_length)
            if info.get("is_success", False):
                success_count += 1

            mean_episode_reward = np.mean(total_episode_rewards[-100:])
            mean_episode_length = np.mean(total_episode_lengths[-100:])
            success_rate = success_count / episode_count

            log_writer.add_scalar("rollout/ep_rew_mean", mean_episode_reward, episode_count)
            log_writer.add_scalar("rollout/ep_len_mean", mean_episode_length, episode_count)
            log_writer.add_scalar("rollout/success_rate", success_rate, episode_count)

            episode_reward = 0
            episode_length = 0
            break

end_time = time.time()
log_writer.close()
env.close()
print(f"Total time: {end_time - start_time}s")
