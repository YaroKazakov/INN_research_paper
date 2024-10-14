import time
import gymnasium as gym
import torch
import numpy as np
from agent import AgentTRPO
from rollout import rollout, update_step, get_entropy
from agent import TinyModel
from pprint import pprint

env = gym.make("Acrobot-v1", render_mode="rgb_array")
#env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
env.reset()
observation_shape = env.observation_space.shape
n_actions = env.action_space.n
print("Observation Space", env.observation_space)
print("Action Space", env.action_space)
agent = AgentTRPO(env.observation_space, n_actions)

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



def train(epochs = 30, model_name = "acrobat.pth"):
    #The max_kl hyperparameter determines how large the KL discrepancy between the old and new policies can be at each stage.
    max_kl = 0.01
    numeptotal = 0  # Number of episodes we have completed so far.
    start_time = time.time()

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
        for k, v in stats.items():
            print(k + ": " + " " * (40 - len(k)) + str(v))
    torch.save(agent.model, model_name)



def test(model_name = "acrobat.pth"):
    """
    Test model from the path: model_name
    """
    agent.model = TinyModel(observation_shape[0],n_actions)
    #agent.model.load_state_dict(torch.load("acrobat.pth", weights_only=False))
    agent.model =torch.load(model_name, weights_only=False)

    scores = []
    env = gym.make("Acrobot-v1", render_mode="human")
    #env.reset()
    print(f"Start testing the model over epochs...")
    env.reset()
    for _ in range(90000):
        #print(env.step(2))
        rollout(env, agent, max_pathlength=2500, n_timesteps=500)


if __name__ == "__main__":
    train(epochs=10, model_name = "acrobat.pth")
    test(model_name = "acrobat.pth")