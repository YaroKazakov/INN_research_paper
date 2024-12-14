import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import pathlib
import csv
import torch

from src.TRPO.optimize import Env_name

def train():
    env = gym.make(Env_name)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1_500_000)
    model.save(Env_name)

def create_dataset(file_name = "data_training.csv", render_mode = "rgb_array", Nepisodes = 100):
    dir = "data/" + Env_name
    pathlib.Path(dir).mkdir(exist_ok=True)
    stream = open(dir + '/' + file_name, 'w', newline='')
    spamwriter = csv.writer(stream, delimiter=' ',
                quotechar='|', quoting=csv.QUOTE_MINIMAL)
    env = gym.make(Env_name, render_mode=render_mode) #rgb_array")
    model = PPO.load("sb/" + Env_name)
    obs, _ = env.reset()

    AR = 0
    Nstep = 0
    while Nepisodes:
        action, _states = model.predict(obs, deterministic=True)
        spamwriter.writerow(torch.cat((torch.Tensor(obs), torch.Tensor(action))).numpy())
        obs, reward, terminated, truncated, info = env.step(action)
        Nstep +=1
        AR += reward
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()
            print(f"current episode = {Nepisodes} {AR = }  {Nstep = }  {AR/Nstep =}")
            Nepisodes -= 1
            Nstep = 0
            AR = 0


if __name__=="__main__":
    #train()
    create_dataset(file_name = "data_training.csv", render_mode = "rgb_array", Nepisodes = 100)
    create_dataset(file_name="data_training_val.csv", render_mode="human", Nepisodes=1)
