import numpy
import torch
import matplotlib.pyplot as plt
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

import torch.nn as nn
import numpy as np
import os
import sys
from src.TRPO.optimize import Env_name

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'TorchIntegral', 'torch_integral')
sys.path.append( mymodule_dir )
#import agent

from torch_integral import IntegralWrapper
from torch_integral import standard_continuous_dims

def main():
    model_path = f"data/{Env_name}/{Env_name}_INN_100.pth"
    model = torch.load(model_path, weights_only=False)
    print(f"{model.state_dict()=}")
    #model = MnistNet()
    #model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval()

    dense_layer_name = "linear2.weight"  # Replace with the name of your Dense layer
    print(dir(model))
    print(model.state_dict().keys())
    W = model.state_dict()[dense_layer_name].cpu().detach().numpy()
    W_INN = W.copy()
    
    W1 = np.fft.fft(W,axis = 0)
    W1 = (W1.real**2 + W1.imag**2)**0.5

    plt.step(np.arange(W.shape[0]),np.sum(W1,axis = 1),label = "INN FFT", where='mid')
    plt.ylabel("FFT magnitude", fontsize=12)
    plt.xlabel("frequency", fontsize=12)

    model_path = f"data/{Env_name}/{Env_name}_NN_100.pth"
    model = torch.load(model_path, weights_only=False)
    model.eval()
    W = model.state_dict()[dense_layer_name].cpu().detach().numpy()
    W_NN = W.copy()
    W1 = np.fft.fft(W,axis = 0)
    W1 = (W1.real**2 + W1.imag**2)**0.5
    plt.step(np.arange(W.shape[0]),np.sum(W1,axis = 1),label = "NN FFT", where='mid')
    plt.legend()
    plt.savefig("Figs/FFT.png")

    fig, ax = plt.subplots()
    plt.clf()
    im = plt.imshow(W_INN, interpolation='none')
    fig.colorbar(im)
    plt.savefig("Figs/W_INN.png")

    plt.clf()
    im = plt.imshow(W_NN, interpolation='none')
    fig.colorbar(im)
    plt.savefig("Figs/W_NN.png")    





def test():
    dense_layer_name = "linear1.weight"
    model_path = f"data/{Env_name}/{Env_name}_INN_100.pth"
    model = torch.load(model_path, weights_only=False)
    model.eval()
    print(f"{model(torch.Tensor([-0.18584757, 1.3153473, -0.40518945, -0.18528453, -0.20594057, -0.21450147, 0.0, 0.0]).to(device='cuda')) = }")
    input = numpy.array([0.05189705, 0.5882736, -0.09816949, -0.15374942, -0.06036125, 0.033021268, 0.0, 0.0]).T
    layer1 = model.state_dict()["linear1.weight"].cpu().detach().numpy().dot(input) + model.state_dict()["linear1.bias"].cpu().detach().numpy()
    layer1 = torch.nn.ReLU()(torch.Tensor(layer1))
    plt.plot(layer1,label = "INN")
    #plt.savefig("dense_layer_weights0.png")


    model_path = f"data/{Env_name}/{Env_name}_NN_100.pth"
    model = torch.load(model_path, weights_only=False)
    model.eval()
    layer1 = model.state_dict()["linear1.weight"].cpu().detach().numpy().dot(input) + model.state_dict()["linear1.bias"].cpu().detach().numpy()
    layer1 = torch.nn.ReLU()(torch.Tensor(layer1))
    plt.plot(layer1,label = "NN")
    plt.legend()
    plt.savefig("dense_layer_weights0.png")

    model_path = f"data/{Env_name}/{Env_name}_NN_100.pth" #src/INN/test_mode0.pth"  #data/LunarLander-v3.pth" #src/INN/test_mode0.pth"  # Replace with your model file path
    model = torch.load(model_path, weights_only=False)
    W = model.state_dict()[dense_layer_name].cpu().detach().numpy()
    #plt.plot(W[1,:],"--")
   
    plt.show()


if __name__=="__main__":
    main()
    #test()
