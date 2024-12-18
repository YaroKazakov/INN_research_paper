import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'TRPO')
sys.path.append( mymodule_dir )
import agent


class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(
            1, 16, 3, padding=1, bias=True, padding_mode="replicate"
        )
        self.conv_2 = nn.Conv2d(
            16, 32, 5, padding=2, bias=True, padding_mode="replicate"
        )
        self.conv_3 = nn.Conv2d(
            32, 64, 5, padding=2, bias=True, padding_mode="replicate"
        )
        self.f_1 = nn.ReLU()
        self.f_2 = nn.ReLU()
        self.f_3 = nn.ReLU()
        self.pool = nn.AvgPool2d(2, 2)
        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        x = self.f_1(self.conv_1(x))
        x = self.pool(x)
        x = self.f_2(self.conv_2(x))
        x = self.pool(x)
        x = self.f_3(self.conv_3(x))
        x = self.pool(x)
        x = self.linear(x[:, :, 0, 0])

        return x

model_path = "/home/YaroKazakov/INN_research_paper/data/LunarLander-v3.pth" #src/INN/test_mode0.pth"  #data/LunarLander-v3.pth" #src/INN/test_mode0.pth"  # Replace with your model file path
model = torch.load(model_path, weights_only=False)
#model = MnistNet()
#model.load_state_dict(torch.load(model_path, weights_only=False))
model.eval()

dense_layer_name = "linear1.weight"  # Replace with the name of your Dense layer
print(dir(model))
print(model.state_dict().keys())
W = model.state_dict()[dense_layer_name].cpu().detach().numpy()

#plt.plot(W[4,:])
plt.imshow(W, interpolation='none')
plt.savefig("dense_layer_weights0.png")
plt.show()

