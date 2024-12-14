import torch
import torch.nn as nn
from typing import Tuple
import numpy as np
from scipy.stats import poisson, norm, rv_discrete, rv_continuous
import gymnasium
#torch.manual_seed(0)
#torch.use_deterministic_algorithms(True, warn_only=True)
import matplotlib.pyplot as plt
import os
script_dir = os.path.dirname( __file__ )
import sys
mymodule_dir = os.path.join( script_dir, '..', '..','Convolutional_KANs', 'kan_convolutional')
sys.path.append( mymodule_dir )

stream_ = open("activation.txt","w")

from Convolutional_KANs.kan_convolutional.KANConv import KAN_Convolutional_Layer
from Convolutional_KANs.kan_convolutional.KANLinear import KANLinear

class TinyModel(torch.nn.Module):

    def __init__(self,input_dim,n_actions, n_neurons = [100,100],
                 isactiondiscrete = True,
                 iskan = False):
        super(TinyModel, self).__init__()
        self.iskan = iskan
        self.isactiondiscrete = isactiondiscrete
        #self.norm0 = torch.nn.BatchNorm1d(8)
        self.linear1 = torch.nn.Linear(input_dim, n_neurons[0])
        #self.norm1 = torch.nn.BatchNorm1d(100)
        self.activation = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(n_neurons[0], n_neurons[1])
        #self.norm2 = torch.nn.BatchNorm1d(100)
        #self.linear3 = torch.nn.Linear(200, 200)
        self.linear4 = torch.nn.Linear(n_neurons[1], n_actions)
        self.softmax = torch.nn.LogSoftmax()


        self.linear1_kan = KANLinear(input_dim, n_neurons[0])
        self.linear2_kan = KANLinear(n_neurons[0], n_neurons[1])
        self.linear4_kan = KANLinear(n_neurons[1], n_actions)

        self.size = input_dim*n_neurons[0] + n_neurons[0] + n_neurons[0]*n_neurons[1] + n_neurons[1] + n_neurons[1]*n_actions + n_actions
        self.size = self.size/(input_dim*100 + 100 + 100*100. + 100. + 100.*n_actions + n_actions)
        self.size = int(self.size*100)

    def forward(self, x):
        if not self.iskan:
            x = self.linear1(x)
            x = self.activation(x)
            x = self.linear2(x)
            x = self.activation(x)
            #stream_.write(" ".join(str(el) for el in x[0].tolist()) + "\n")
            x = self.linear4(x)
            if self.isactiondiscrete:
                x = self.softmax(x)
            return x
        if self.iskan:
            x = self.linear1_kan(x)
            x = self.linear2_kan(x)
            #stream_.write(" ".join(str(el) for el in x[0].tolist()) + "\n")
            x = self.linear4_kan(x)
            if self.isactiondiscrete:
                x = self.softmax(x)
            return x

def is_discrete(dist):
    if hasattr(dist, 'dist'):
        return isinstance(dist.dist, rv_discrete)
    else: return isinstance(dist, rv_discrete)









class AgentTRPO(nn.Module):
    def __init__(self, state_shape: Tuple[int], n_actions: int,
                 n_neurons = 100, isactiondiscrete = False, iskan = False):
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
        self.model = TinyModel(input_dim,n_actions, n_neurons = n_neurons,
                               isactiondiscrete = self.isactiondiscrete,
                               iskan = iskan) #nn.Sequential(nn.Linear(input_dim,32), nn.ReLU(), nn.Linear(32,n_actions),nn.LogSoftmax())


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
