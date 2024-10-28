import time
import gymnasium as gym
import torch
import numpy as np
from src.TRPO.agent import AgentTRPO
from src.TRPO.rollout import rollout, update_step, get_entropy
from src.TRPO.agent import TinyModel
from pprint import pprint
import os

#env = gym.make("Acrobot-v1", render_mode="rgb_array")
#env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
Env_name = 'LunarLander-v3'
#Env_name = "Acrobot-v1"
env = gym.make(Env_name, render_mode="rgb_array")
env.reset()
observation_shape = env.observation_space.shape
n_actions = env.action_space.n
print("Observation Space", env.observation_space)
print("Action Space", env.action_space)
agent_ = AgentTRPO(env.observation_space, n_actions)

def sometests():
    # Check if log-probabilities satisfies all the requirements
    log_probs = agent.get_log_probs(torch.tensor(env.reset()[0][np.newaxis], dtype=torch.float32))
    assert (
        isinstance(log_probs, torch.Tensor) and
        log_probs.requires_grad
    ), "log_probs must be a torch.Tensor with grad"
    assert log_probs.shape == (1, n_actions)
    sums = torch.exp(log_probs).sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums))
    # Demo use
    print("sampled:", [agent.act(n_actions,env.reset()[0]) for _ in range(5)])
    print("greedy:", [agent.act(n_actions,env.reset()[0], sample=False) for _ in range(5)])

    paths = rollout(env, agent, max_pathlength=5, n_timesteps=100)
    pprint(paths[-1])

    assert (paths[0]['policy'].shape == (5, n_actions))
    assert (paths[0]['cumulative_returns'].shape == (5,))
    assert (paths[0]['rewards'].shape == (5,))
    assert (paths[0]['observations'].shape == (5,) + observation_shape)
    assert (paths[0]['actions'].shape == (5,))



def train(epochs = 30, model_name = "acrobat.pth", metrics = "metrics.dat", agent = agent_):
    #The max_kl hyperparameter determines how large the KL discrepancy between the old and new policies can be at each stage.
    max_kl = 0.01
    numeptotal = 0  # Number of episodes we have completed so far.
    start_time = time.time()
    os.makedirs(os.path.dirname(metrics), exist_ok=True)
    out_stream = open(metrics,"w")
    maxreward = -1000;
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



def test(model_name = "acrobat.pth", nrollout = 4, file_ = "data_training.csv", sample_=False, agent = agent_):
    """
    Test model from the path: model_name
    """
    agent.model = TinyModel(observation_shape[0],n_actions)
    #agent.model.load_state_dict(torch.load("acrobat.pth", weights_only=False))
    agent.model =torch.load(model_name, weights_only=False).cpu()

    scores = []
    env = gym.make(Env_name, render_mode="human")
    #env.reset()
    print(f"Start testing the model over epochs...")

    env.reset()
    stats = {}
    episode_rewards = np.array([])
    for _ in range(nrollout):
        #print(env.step(2))
        paths = rollout(env, agent, max_pathlength=2500, n_timesteps=500, file = file_, sample=sample_)
        episode_rewards = np.append(episode_rewards, np.array([path["rewards"].sum() for path in paths]))
    stats["Average sum of rewards per episode"] = episode_rewards.mean()
    print(stats)

    return stats

if __name__ == "__main__":
    #train(epochs=50, model_name = Env_name + ".pth")
    #test(model_name = "/media/eakozyrev/diskD/RL/INN_RL/data/" + Env_name + ".pth")
    #test(model_name="/media/eakozyrev/diskD/RL/INN_RL/src/INN/test_mode1.pth")
    #test(model_name="/media/eakozyrev/diskD/RL/INN_RL/src/INN/test_mode2.pth")
    #test(model_name="/media/eakozyrev/diskD/RL/INN_RL/src/INN/test_mode3.pth")
    #test(model_name="/media/eakozyrev/diskD/RL/INN_RL/src/INN/test_mode4.pth")
    pass