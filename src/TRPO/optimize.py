import time
import gymnasium as gym
import torch
import numpy as np
from src.TRPO.agent import AgentTRPO
from src.TRPO.rollout import rollout, update_step, get_entropy
from src.TRPO.agent import TinyModel
from pprint import pprint
import os
import cv2
from torchvision.transforms import v2, InterpolationMode
import matplotlib.pyplot as plt


#Env_name = "Acrobot-v1"
Env_name = 'LunarLander-v3'
#Env_name = "Swimmer-v4"
#Env_name = "Walker2d-v4"
#Env_name = "HalfCheetah-v4"
#Env_name = "Ant-v4"


env = gym.make(Env_name, render_mode="rgb_array")
observation_shape = env.observation_space.shape
print("Observation Space", env.observation_space)
print("Action Space", env.action_space)


def make_agent(n_neurons = [100,100], isKAN=False):
    n_actions = 0
    isactiondiscrete = False
    try:
        n_actions = env.action_space.n
        isactiondiscrete = True
    except:
        n_actions = env.action_space.shape[0]
    assert n_actions != 0
    agent = AgentTRPO(env.observation_space, n_actions,
                      n_neurons=n_neurons,
                      isactiondiscrete = isactiondiscrete,
                      iskan=isKAN)
    return agent


def train(epochs = 30, model_name = "acrobat.pth", metrics = "metrics.dat", agent = None):
    #The max_kl hyperparameter determines how large the KL discrepancy between the old and new policies can be at each stage.
    max_kl = 0.01
    numeptotal = 0  # Number of episodes we have completed so far.
    start_time = time.time()
    os.makedirs(os.path.dirname(metrics), exist_ok=True)
    out_stream = open(metrics,"w")
    maxreward = -1000
    for i in range(epochs):
        print(f"\n********** Iteration %{i} ************")
        print("Rollout")
        paths = rollout(env, agent)
        print("Made rollout")

        # Updating policy.
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        returns = np.concatenate([path["cumulative_returns"] for path in paths])
        old_probs = np.concatenate([path["policy"] for path in paths])
        loss, kl = update_step(agent, observations, actions, returns, old_probs, max_kl)
        # Report current progress
        episode_rewards = np.array([path["rewards"].sum() for path in paths])
        stats = {}
        numeptotal += len(episode_rewards)
        #stats["returns"] = returns
        #stats["old_probs"] = old_probs
        stats["Total number of episodes"] = numeptotal
        stats["Average sum of rewards per episode"] = episode_rewards.mean()
        stats["Std of rewards per episode"] = episode_rewards.std()
        stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time)/60.)
        stats["KL between old and new distribution"] = kl.data.numpy()
        stats["Entropy"] = get_entropy(agent, observations).data.numpy()
        stats["Surrogate loss"] = loss.data.numpy()
        print(numeptotal, episode_rewards.mean(), episode_rewards.std())
        out_stream.write(f"{numeptotal} {episode_rewards.mean()} {episode_rewards.std()}\n")
        for k, v in stats.items():
            print(k + ": " + " " * (40 - len(k)) + str(v))
        if episode_rewards.mean() > maxreward:
            os.makedirs(os.path.dirname(model_name), exist_ok = True)
            torch.save(agent.model, model_name)
            maxreward = episode_rewards.mean()


def resizeNN(modelH, models, interpolation = InterpolationMode.BICUBIC, sizes = (100,100)):
    '''
    :param modelH: Large model
    :param models: Small model that is modified here
    '''
    sdH = modelH.state_dict()
    print(f"{sdH.keys()=}")
    sds = models.state_dict()
    print("Start NN resize")
    # BICUBIC
    #sds['linear1.weight'] = cv2.Resize(sdH['linear1.weight'].unsqueeze(0), sds['linear1.weight'].shape, interpolation=cv2.INTER_LINEAR)[0]
    print(f"{sds['linear1.weight'].shape = }")
    sds['linear1.weight'] = v2.Resize(size=sds['linear1.weight'].shape, interpolation = interpolation)(sdH['linear1.weight'].unsqueeze(0))[0]
    new_shape = (1,sds['linear1.bias'].shape[0])
    sds['linear1.bias'] = v2.Resize(size=new_shape, interpolation = interpolation)(sdH['linear1.bias'].unsqueeze(0).unsqueeze(0))[0][0]

    sds['linear2.weight'] = v2.Resize(size=sds['linear2.weight'].shape, interpolation = interpolation)(sdH['linear2.weight'].unsqueeze(0))[0]
    new_shape = (1,sds['linear2.bias'].shape[0])
    sds['linear2.bias'] = v2.Resize(size=new_shape, interpolation = interpolation)(sdH['linear2.bias'].unsqueeze(0).unsqueeze(0))[0][0]
    
    sds['linear4.weight'] = v2.Resize(size=sds['linear4.weight'].shape, interpolation = interpolation)(sdH['linear4.weight'].unsqueeze(0))[0]
    new_shape = (1,sds['linear4.bias'].shape[0])
    sds['linear4.bias'] = v2.Resize(size=new_shape, interpolation = interpolation)(sdH['linear4.bias'].unsqueeze(0).unsqueeze(0))[0][0]
    
    models.load_state_dict(sds)


def test(model_name = "acrobat.pth", nrollout = 4, file_ = "data_training.csv",
         sample_=False, render = False, ModelH = None, agent = None):
    """
    Test model from the path: model_name
    """
    #agent.model = TinyModel(observation_shape[0],n_actions)
    #agent.model.load_state_dict(torch.load("acrobat.pth", weights_only=False))
    print(f"load model from {model_name}")
    agent.model =torch.load(model_name, weights_only=False).cpu()
    agent.model.eval()
    
    if ModelH != None:
        modelH = torch.load(ModelH, weights_only=False)
        resizeNN(modelH, agent.model)
        agent.model.eval()

    render_mode = "rgb_array"
    if render: render_mode = "human"
    env = gym.make(Env_name, render_mode=render_mode)
    env.reset()
    print(f"Start testing the model over epochs...")
    stats = {}
    episode_rewards = np.array([])
    for _ in range(nrollout):
        paths = rollout(env, agent, max_pathlength=2500, n_timesteps=500, file = file_, sample=sample_)
        episode_rewards = np.append(episode_rewards, np.array([path["rewards"].sum() for path in paths]))
    stats["Average sum of rewards per episode"] = episode_rewards.mean()
    print(stats)
    env.close()
    return stats

