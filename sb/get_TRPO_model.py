import gymnasium as gym
import numpy as np
from sb3_contrib import TRPO
import pathlib
import csv
import torch

#ENV_NAME = "HalfCheetah-v4"
ENV_NAME =  "Ant-v4"


def train():
    env = gym.make(ENV_NAME)
    model = TRPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1_500_000)
    model.save(ENV_NAME)

def create_dataset(file_name = "data_training.csv", render_mode = "rgb_array", Nepisodes = 100):
    dir = "data/" + ENV_NAME
    pathlib.Path(dir).mkdir(exist_ok=True)
    stream = open(dir + '/' + file_name, 'w', newline='')
    spamwriter = csv.writer(stream, delimiter=' ',
                quotechar='|', quoting=csv.QUOTE_MINIMAL)
    env = gym.make(ENV_NAME, render_mode=render_mode) #rgb_array")
    model = TRPO.load("sb/" + ENV_NAME)
    obs, _ = env.reset()
    while Nepisodes:
        action, _states = model.predict(obs, deterministic=True)
        spamwriter.writerow(torch.cat((torch.Tensor(obs), torch.Tensor(action))).numpy())
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()
            print(f"current episode = {Nepisodes}")
            Nepisodes -= 1

if __name__=="__main__":
    train()
    create_dataset(file_name = "data_training.csv", render_mode = "rgb_array", Nepisodes = 100)
    create_dataset(file_name="data_training_val.csv", render_mode="human", Nepisodes=1)