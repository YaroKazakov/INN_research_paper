import numpy
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'TorchIntegral', 'torch_integral')
sys.path.append( mymodule_dir )
#import agent

from torch_integral import IntegralWrapper
from torch_integral import standard_continuous_dims

def main():

    model_path = "/home/eakozyrev/diskD/RL/INN_RL/data/Lunar/LunarLander-v3_INN_100.pth" #src/INN/test_mode0.pth"  #data/LunarLander-v3.pth" #src/INN/test_mode0.pth"  # Replace with your model file path
    model = torch.load(model_path, weights_only=False)
    #model = MnistNet()
    #model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval()

    dense_layer_name = "linear1.bias"  # Replace with the name of your Dense layer
    print(dir(model))
    print(model.state_dict().keys())
    W = model.state_dict()[dense_layer_name].cpu().detach().numpy()

    #plt.plot(W[0,:])
    #plt.plot(W[1,:])
    #pool = torch.nn.AvgPool1d(kernel_size=4, stride=4)(torch.Tensor([[W[1,:]]]))
    #plt.plot(list(range(0,100,4)),pool[0,0]*4,"o")
    #plt.plot(W[2,:])
    #plt.plot(W[3,:])
    #plt.imshow(W, interpolation='none')

    wrapper = IntegralWrapper(init_from_discrete=False, verbose=True)
    continuous_dims = standard_continuous_dims(model)
    model_int = wrapper(model, [1, 8], continuous_dims)

    model_int.resize([4,8,100])
    size = model_int.eval().calculate_compression()
    model = model_int.get_unparametrized_model()
    W = model.state_dict()[dense_layer_name].cpu().detach().numpy()

    print("="*30)
    related_groups, continuous_dims = wrapper.preprocess_model(
        model,
        torch.Tensor([[1]*8]),
        continuous_dims,
        None,
        None,
        None,
    )
    print(f"{related_groups = }")
    print(f"{continuous_dims = }")
    print(f"{list(model.modules()) = }")
    name = 'linear1.weight'
    btv = wrapper.build_parameterization(list(model.modules())[1],"bias", continuous_dims[name])
    print(f"{btv.grid = }")




def test():
    print(f"{model(torch.Tensor([-0.18584757, 1.3153473, -0.40518945, -0.18528453, -0.20594057, -0.21450147, 0.0, 0.0]).to(device='cuda')) = }")
    input = numpy.array([[1]*8]).T
    layer1 = model.state_dict()["linear1.weight"].cpu().detach().numpy().dot(input)
    layer1 = torch.nn.ReLU()(torch.Tensor(layer1))
    plt.plot(layer1)


    #plt.plot(W[1,:],"--")

    #plt.plot(list(range(0,100,4)),W[1,:],"+")
    plt.legend()
    plt.savefig("dense_layer_weights0.png")

    model_path = "/home/eakozyrev/diskD/RL/INN_RL/data/Lunar/LunarLander-v3_NN_100.pth" #src/INN/test_mode0.pth"  #data/LunarLander-v3.pth" #src/INN/test_mode0.pth"  # Replace with your model file path
    model = torch.load(model_path, weights_only=False)
    W = model.state_dict()[dense_layer_name].cpu().detach().numpy()
    #plt.plot(W[1,:],"--")

    plt.show()


if __name__=="__main__":
    main()