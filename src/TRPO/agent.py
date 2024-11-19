import torch
import torch.nn as nn
from typing import Tuple
import numpy as np
from scipy.stats import poisson, norm, rv_discrete, rv_continuous
import gymnasium
#torch.manual_seed(0)
#torch.use_deterministic_algorithms(True, warn_only=True)

class TinyModel(torch.nn.Module):

    def __init__(self,input_dim,n_actions, n_neurons = 100, isactiondiscrete = True):
        super(TinyModel, self).__init__()
        self.isactiondiscrete = isactiondiscrete
        #self.norm0 = torch.nn.BatchNorm1d(8)
        self.linear1 = torch.nn.Linear(input_dim, n_neurons)
        #self.norm1 = torch.nn.BatchNorm1d(100)
        self.activation = torch.nn.PReLU()
        self.linear2 = torch.nn.Linear(n_neurons, n_neurons)
        #self.norm2 = torch.nn.BatchNorm1d(100)
        #self.linear3 = torch.nn.Linear(200, 200)
        self.linear4 = torch.nn.Linear(n_neurons, n_actions)
        self.softmax = torch.nn.LogSoftmax()

    def forward(self, x):
        #print(f"::::::::::::::::::::in{x=}")
        x = self.linear1(x)
        #x = self.norm1(x)
        x = self.activation(x)
        x = self.linear2(x)
        #x = self.norm2(x)
        x = self.activation(x)
        #x = self.linear3(x)
        #x = self.activation(x)
        x = self.linear4(x)
        #x = self.activation(x)
        if self.isactiondiscrete:
            x = self.softmax(x)
        #print(f"::::::::::::::::::::out{x=}")
        return x

def is_discrete(dist):
    if hasattr(dist, 'dist'):
        return isinstance(dist.dist, rv_discrete)
    else: return isinstance(dist, rv_discrete)

class AgentTRPO(nn.Module):
    def __init__(self, state_shape: Tuple[int], n_actions: int, n_neurons = 100, isactiondiscrete = False):
        torch.manual_seed(123)
        super().__init__()
        #assert isinstance(state_shape, tuple)
        #assert len(state_shape) == 1
        print("---",type(state_shape))
        self.isactiondiscrete = isactiondiscrete
        if isinstance(state_shape,gymnasium.spaces.discrete.Discrete):
            input_dim = 1
        else:
            input_dim = state_shape.shape[0]
        # Define the policy network
        self.input_dim = input_dim
        self.naction = n_actions
        self.model = TinyModel(input_dim,n_actions, n_neurons = n_neurons, isactiondiscrete = self.isactiondiscrete) #nn.Sequential(nn.Linear(input_dim,32), nn.ReLU(), nn.Linear(32,n_actions),nn.LogSoftmax())


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
        return torch.exp(self.forward(states)) #self.forward(states) #torch.exp(self.forward(states))

    def act(self, n_actions, obs: np.ndarray, sample: bool = True):

        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                probs = self.get_probs(torch.tensor(obs[np.newaxis], dtype=torch.float32)).numpy()
            elif isinstance(obs,int):
                probs = self.get_probs(torch.tensor([[obs]], dtype=torch.float32)).numpy()

        if not self.isactiondiscrete:
            return self.get_log_probs(torch.tensor(obs[np.newaxis], dtype=torch.float32)).detach().numpy()[0], probs[0]

        if sample:
            action = int(np.random.choice(n_actions, p=probs[0]))
        else:
            action = int(np.argmax(probs))

        return action, probs[0]
