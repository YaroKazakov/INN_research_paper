import matplotlib.pyplot as plt
import numpy as np

plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
import pandas as pd
import torch
from torch_integral import IntegralWrapper
import torch_integral as inn
from torch_integral import standard_continuous_dims
import os
import sys
from src.TRPO.optimize import Env_name
from torchsummary import summary
import scipy.stats as stats
import numpy as np

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, 'src', 'TRPO')
sys.path.append( mymodule_dir )
import optimize
import matplotlib.pyplot as plt

list_scan = [100, 95, 90, 85, 70, 60,50,40,30,20,10,5]

def run_scan(test_type = "NN"):
    model =torch.load(f"data/{Env_name}/{Env_name}_"+test_type+"_100.pth", weights_only=False) #.cpu()
    #print(f"{model(torch.Tensor([[1]*8]).to(device='cuda')) = }")
    wrapper = IntegralWrapper(init_from_discrete=False, verbose=True)
    continuous_dims = standard_continuous_dims(model)
    model_int = wrapper(model, [1, optimize.observation_shape[0]], continuous_dims)
    sizes = []
    rewards = []
    dir = f"data/{Env_name}/"
    os.makedirs(dir, exist_ok = True)
    path_to_data = f"data/{Env_name}/" + test_type +"_reward.dat"
    #list_scan = [40,35,30,20,15,10,8,6]
    for el in list_scan:
        agent_ = optimize.make_agent(n_neurons=el)
        model_int.resize([agent_.naction, agent_.input_dim, el, el])
        size = model_int.eval().calculate_compression()
        sizes.append(size)
        model = model_int.get_unparametrized_model()
        name = f"test_model_{el}.pth" #
        torch.save(model, dir+name)
        ModelH = None
        if test_type == "NN":
            ModelH = f'data/{Env_name}/{Env_name}_NN_100.pth'
        if test_type == "INN":
            ModelH = f'data/{Env_name}/{Env_name}_INN_100.pth'

        btv = optimize.test(model_name= (dir + name), nrollout = 30, file_ = "",
                            sample_ = False, ModelH = ModelH, agent=agent_)
        rewards.append(btv["Average sum of rewards per episode"])

    stream = open(path_to_data,"w")
    for i,j,k in zip(sizes,rewards,list_scan):
        print(f"{k} {(i)*100.:.1f} & {j}")
        stream.write(f"{(i)*100.:.1f} & {j}\n")

    stream.close()

    plt.plot(sizes,rewards,"*-")
    plt.xlabel("compression rate")
    plt.ylabel("mean reward")
    plt.savefig("Figs/Lunar_compression.png")
    #plt.show()

def plot_reward():
    i = 0
    dfNN = pd.read_csv(f"data/{Env_name}/NN_reward.dat_{i}", sep=" & ", engine="python", names=["size", "AR"])
    dfINN = pd.read_csv(f"data/{Env_name}/INN_reward.dat_{i}", delimiter=" & ", names=["size", "AR"])

    plt.plot(dfNN.iloc[0, 0], dfNN.iloc[0, 1], label="trained NN", marker="*", markersize=10, color="black", lw=0)
    plt.plot(dfINN.iloc[0, 0], dfINN.iloc[0, 1], label="trained INN", marker="*", markersize=10, color="blue", lw=0)
    plt.plot(dfNN.iloc[1:, 0], dfNN.iloc[1:, 1], "x--", label="resized NN", markersize=4, color="black")
    plt.plot(dfINN.iloc[1:, 0], dfINN.iloc[1:, 1], "o--", label="resized INN", markersize=4, color="blue")

    for i in range(1,10):
        dfNN = pd.read_csv(f"data/{Env_name}/NN_reward.dat_{i}", sep=" & ", engine="python", names = ["size", "AR"])
        dfINN = pd.read_csv(f"data/{Env_name}/INN_reward.dat_{i}", delimiter=" & ", names = ["size", "AR"])
        plt.plot(dfNN.iloc[0, 0], dfNN.iloc[0, 1], marker="*", markersize=10, color="black", lw=0)
        plt.plot(dfINN.iloc[0, 0], dfINN.iloc[0, 1], marker="*", markersize=10, color="blue", lw=0)
        plt.plot(dfNN.iloc[1:, 0], dfNN.iloc[1:, 1], "x--", markersize=4, color="black")
        plt.plot(dfINN.iloc[1:, 0], dfINN.iloc[1:, 1], "o--", markersize=4, color="blue")


    plt.legend()
    plt.xlabel("Compression Rate, %", fontsize=18)
    plt.ylabel("Average Reward", fontsize=18)
    plt.grid()
    plt.savefig(f"Figs/{Env_name}_compression.png")
    plt.show()

def get_stat():

    sizes = {'LunarLander-v2':[0.0, 9.2, 17.9, 26.2, 48.4, 61.0, 71.9, 81.0, 88.4, 94.0, 97.9, 99.1],
             "HalfCheetah-v4":[0.0, 8.8, 17.2, 25.2, 46.8, 59.2, 70.0, 79.2, 86.8, 92.7, 97.1, 98.7],
             "Acrobot-v1":[0.0,9.3, 18.1, 26.5, 48.9, 61.6, 72.5, 81.6, 88.9, 94.4, 98.1, 99.2],
             "Ant-v4":[0.0, 8.5, 16.6, 24.3, 45.3, 57.5, 68.2, 77.5, 85.3, 91.6, 96.5, 98.4]
             }
    files = range(10)
    i = 0
    for size in sizes[Env_name]:
        arr = []
        arrNN = []
        for file in files:
            df = pd.read_csv(f"data/{Env_name}/INN_reward.dat_{file}", sep=" & ", engine="python", names=["size", "AR"])
            arr.append(df[df["size"] == size]["AR"].iloc[0])
        for file in files:
            df = pd.read_csv(f"data/{Env_name}/NN_reward.dat_{file}", sep=" & ", engine="python", names=["size", "AR"])
            arrNN.append(df[df["size"] == size]["AR"].iloc[0])
        #print(arr)
        arr = np.array(arr)
        arrNN = np.array(arrNN)
        Nsigma = abs(arr.mean() - arrNN.mean())/((arr.std(ddof=1)**2 + arrNN.std(ddof=1)**2)**0.5)
        pvalue = stats.ttest_ind(np.array(arrNN), np.array(arr), equal_var = False).pvalue
        print(f"{list_scan[i]} & {size} &  & {arr.mean():.0f} , {arr.std(ddof=1):.0f} &  & {arrNN.mean():.0f} , {arrNN.std(ddof=1):.0f} & {pvalue:.1E}")
        i+=1


if __name__ == "__main__":
    #run_scan(test_type="INN")
    #run_scan(test_type="NN")
    plot_reward()
    #get_stat()
