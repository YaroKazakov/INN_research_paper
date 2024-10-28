import matplotlib.pyplot as plt
import torch
from torch_integral import IntegralWrapper
from torch_integral import standard_continuous_dims
import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, 'src', 'TRPO')
sys.path.append( mymodule_dir )
import agent
import optimize


model =torch.load("data/test_mode0.pth", weights_only=False) #.cpu()
#print(f"{model(torch.Tensor([[1]*8]).to(device='cuda')) = }")
wrapper = IntegralWrapper(init_from_discrete=False, verbose=True)
continuous_dims = standard_continuous_dims(model)
model_int = wrapper(model, [1, 8], continuous_dims)
sizes = []
rewards = []
dir = "data/Lunar/"
os.makedirs(dir, exist_ok = True)


for el in [100,90,80,70,60,50,40,30,20]: #,10,7]:
    model_int.resize([4,8,el,el])
    size = model_int.eval().calculate_compression()
    sizes.append(size)
    #model = model_int.get_unparametrized_model()
    #torch.save(model, dir+"test_model.pth")
    btv = optimize.test(model_name= (dir+f"LunarLander-v3_{el}.pth"), nrollout = 15, file_ = "", sample_ = False)
    rewards.append(btv["Average sum of rewards per episode"])
    #torch.save(model4, "test_mode4.pth")
    #train(epochs=1, model_name='data/Lunar/LunarLander-v3' + ".pth", metrics="data/Lunar/metric_TRPO.csv")

for i,j in zip(sizes,rewards):
    print(f"{(1.-i)*100.:.1f} & {j}")

plt.plot(sizes,rewards,"*-")
plt.xlabel("compression rate")
plt.ylabel("mean reward")
plt.savefig("Lunar_compression.png")
plt.show()
