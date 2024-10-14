import torch
import torch.nn as nn
from typing import Tuple
import numpy as np
from scipy.stats import poisson, norm, rv_discrete, rv_continuous
import gymnasium

class TinyModel(torch.nn.Module):

    def __init__(self,input_dim,n_actions):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(input_dim, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, n_actions)
        self.softmax = torch.nn.LogSoftmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

def is_discrete(dist):
    if hasattr(dist, 'dist'):
        return isinstance(dist.dist, rv_discrete)
    else: return isinstance(dist, rv_discrete)

class AgentTRPO(nn.Module):
    def __init__(self, state_shape: Tuple[int], n_actions: int):
        super().__init__()
        #assert isinstance(state_shape, tuple)
        #assert len(state_shape) == 1
        print("---",type(state_shape))
        if isinstance(state_shape,gymnasium.spaces.discrete.Discrete):
            input_dim = state_shape.n
        else:
            input_dim = state_shape.shape[0]
        # Define the policy network
        self.model = TinyModel(input_dim,n_actions) #nn.Sequential(nn.Linear(input_dim,32), nn.ReLU(), nn.Linear(32,n_actions),nn.LogSoftmax())


    def forward(self, states: torch.Tensor):
        """
        takes agent's observation, returns log-probabilities
        :param state_t: a batch of states, shape = [batch_size, state_shape]
        """
        log_probs = self.model(states)
        return log_probs

    def get_log_probs(self, states: torch.Tensor):
        '''
        Log-probs for training
        '''
        return self.forward(states)

    def get_probs(self, states: torch.Tensor):
        '''
        Probs for interaction
        '''
        return torch.exp(self.forward(states))

    def act(self, n_actions, obs: np.ndarray, sample: bool = True):

        with torch.no_grad():
            probs = self.get_probs(torch.tensor(obs[np.newaxis], dtype=torch.float32)).numpy()

        if sample:
            action = int(np.random.choice(n_actions, p=probs[0]))
        else:
            action = int(np.argmax(probs))

        return action, probs[0]
