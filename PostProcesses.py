from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
from sympy.abc import alpha

plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
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

def run_scan_(list_scan, test_type = "NN", mode = "BIC"):
    sizes = []
    rewards = []
    dir = f"data/{Env_name}/"
    os.makedirs(dir, exist_ok = True)
    path_to_data = f"data/{Env_name}/" + test_type +"_reward.dat" + mode
    agent_ = optimize.make_agent(n_neurons=(100,100))
    if test_type == "KAN":
        agent_ = optimize.make_agent(n_neurons=(100,100), isKAN = True)
    for el in list_scan:
        print(el)
        if mode == "":
            name = f"{Env_name}_{test_type}_100.pth" + el
        else:
            name = f"{Env_name}_{test_type}_100.pth" + el + "_" + mode
        print(f"start test of the model {name}")
        print(f"{torch.load(dir + name).size = }")
        btv = optimize.test(model_name= (dir + name),
                            nrollout = 30,
                            file_ = "",
                            sample_ = False,
                            ModelH = None,
                            agent=agent_)
        rewards.append(btv["Average sum of rewards per episode"])
        sizes.append(torch.load(dir + name).size)
    stream = open(path_to_data,"w")
    for j,k in zip(rewards,sizes):
        print(f"{k} & {j}")
        stream.write(f"{k} & {j}\n")
    stream.close()

def run_scan(test_type = "NN"):
    list_scan = ["100_100_0", "100_95_0", "100_90_0", "100_85_0",
                 "100_70_0", "100_60_0", "100_50_0", "100_40_0", "100_30_0", "100_20_0", "100_10_0",
                 "100_5_0"]
    if not test_type == "KAN":
        run_scan_(list_scan, test_type, mode="BICUBIC0")
        run_scan_(list_scan, test_type, mode="BICUBIC1")
        run_scan_(list_scan, test_type, mode="BICUBIC2")
    list_scan = ["100_0_0", "100_0_1", "100_0_2", "100_0_3", "100_0_5", "100_0_10",
                 "100_0_100", "100_0_900", "100_0_2000", "100_0_5000", "100_0_10000", "100_0_50000", "100_0_75000"]
    run_scan_(list_scan, test_type, mode="")

def plot_reward():
    i = 0
    dfNN = pd.read_csv(f"data/{Env_name}/NN_reward.datNEAR", sep=" & ", engine="python", names=["size", "AR"])
    dfINN = pd.read_csv(f"data/{Env_name}/INN_reward.datNEAR", delimiter=" & ", names=["size", "AR"])
    dfKAN = pd.read_csv(f"data/{Env_name}/INN_reward.datNEAR", delimiter=" & ", names=["size", "AR"])

    plt.plot(dfNN.iloc[0, 0], dfNN.iloc[0, 1], label="trained NN", marker="*", markersize=10, color="black", lw=0)
    plt.plot(dfINN.iloc[0, 0], dfINN.iloc[0, 1], label="trained INN", marker="*", markersize=10, color="blue", lw=0)
    plt.plot(dfKAN.iloc[0, 0], dfKAN.iloc[0, 1], label="trained KAN", marker="*", markersize=10, color="red", lw=0)

    plt.plot(dfNN.iloc[1:, 0], dfNN.iloc[1:, 1], "x--", label="resized NN", markersize=4, color="black")
    plt.plot(dfINN.iloc[1:, 0], dfINN.iloc[1:, 1], "o--", label="resized INN", markersize=4, color="blue")
    plt.plot(dfKAN.iloc[1:, 0], dfKAN.iloc[1:, 1], "x--", label="resized KAN", markersize=4, color="red")
    plt.xticks(rotation=45, ha='right')
    for i in range(1,1):
        dfNN = pd.read_csv(f"data/{Env_name}/NN_reward.dat_{i}", sep=" & ", engine="python", names = ["size", "AR"])
        dfINN = pd.read_csv(f"data/{Env_name}/INN_reward.dat_{i}", delimiter=" & ", names = ["size", "AR"])
        plt.plot(dfNN.iloc[0, 0], dfNN.iloc[0, 1], marker="*", markersize=10, color="black", lw=0)
        plt.plot(dfINN.iloc[0, 0], dfINN.iloc[0, 1], marker="*", markersize=10, color="blue", lw=0)
        plt.plot(dfNN.iloc[1:, 0], dfNN.iloc[1:, 1], "x--", markersize=4, color="black")
        plt.plot(dfINN.iloc[1:, 0], dfINN.iloc[1:, 1], "o--", markersize=4, color="blue")


    plt.legend()
    plt.xlabel("Compression Rate, %", fontsize=25)
    plt.ylabel("Average Reward", fontsize=25)
    plt.grid()
    plt.savefig(f"Figs/{Env_name}_compression.png")
    plt.show()



def plot_one(fig, ax, type = "BIL", plotstars = False, meccolor = 'black',title = "",hatch = "/",
             color = "black"):

    files = range(5,10)
    i = 0
    massINN, massNN, Size = [],[],[]
    for size in range(0,10):
        arr = []
        arrNN = []
        sizes = []
        for file in files:
            df = pd.read_csv(f"data/{Env_name}/INN_reward.dat{type}_{file}", sep=" & ", engine="python", names=["size", "AR"])
            arr.append(df.iloc[size,1])
            sizes.append(100-df.iloc[size,0])
        for file in files:
            df = pd.read_csv(f"data/{Env_name}/NN_reward.dat{type}_{file}", sep=" & ", engine="python", names=["size", "AR"])
            arrNN.append(df.iloc[size,1])
        #print(arr)
        arr = np.array(arr)
        massINN.append(arr)
        arrNN = np.array(arrNN)
        massNN.append(arrNN)
        Size.append(np.array(sizes))
        Nsigma = abs(arr.mean() - arrNN.mean())/((arr.std(ddof=1)**2 + arrNN.std(ddof=1)**2)**0.5)
        pvalue = stats.ttest_ind(np.array(arrNN), np.array(arr), equal_var = False).pvalue
        i+=1

    massINN = np.array(massINN)
    massNN = np.array(massNN)
    Size = np.array(Size)

    if plotstars:
        plt.plot(Size.mean(axis=1)[0], massINN.mean(axis=1)[0], marker="^",
                 markersize=12, color="black", lw=0, label="trained INN")
        plt.plot(Size.mean(axis=1)[0], massNN.mean(axis=1)[0],
                 marker="*", markersize=12,
                 color="black", lw=0, label="trained NN")


    ax.fill_between(Size.mean(axis=1)[1:], massINN.mean(axis=1)[1:] - massINN.std(axis=1)[1:],
                    massINN.mean(axis=1)[1:] + massINN.std(axis=1)[1:], color=color, alpha=0.6,
                    hatch = hatch)
    ax.fill_between(Size.mean(axis=1)[1:], massNN.mean(axis=1)[1:] - massNN.std(axis=1)[1:],
                    massNN.mean(axis=1)[1:] + massNN.std(axis=1)[1:], color=color, alpha=0.2,
                    hatch = hatch)
    ax.plot(Size.mean(axis=1)[1:], massINN.mean(axis=1)[1:], "o-", color=color,markersize= 12,
            mec = meccolor, label="resized INN " + title,lw = 2)
    ax.plot(Size.mean(axis=1)[1:], massNN.mean(axis=1)[1:], "s--", color=color, markersize= 12, label="resized NN " + title,
            alpha = 0.3,lw=4)

def draw_activation():
    fig, axs = plt.subplots()
    df = pd.read_csv("activationKAN.txt", delimiter=" ")
    plt.imshow(df.iloc[:100:2],label="KAN")
    plt.xlabel("2nd layer output", fontsize=20)
    plt.ylabel("# observation frame", fontsize=20)
    axs.set_title(f'{"KAN"}', fontsize=20)
    plt.colorbar()
    plt.show()
    plt.clf()
    fig, axs = plt.subplots()
    df = pd.read_csv("activationNN.txt", delimiter=" ")
    plt.imshow(df.iloc[:100:2],label="NN")
    axs.set_title(f'{"NN"}', fontsize=20)
    plt.colorbar()
    plt.xlabel("2nd layer output", fontsize=20)
    plt.ylabel("# observation frame", fontsize=20)
    plt.show()
    plt.clf()
    fig, axs = plt.subplots()
    df = pd.read_csv("activationINN.txt", delimiter=" ")
    plt.imshow(df.iloc[:100:2],label="INN")
    axs.set_title(f'{"INN"}', fontsize=20)
    plt.colorbar()
    plt.xlabel("2nd layer output", fontsize=20)
    plt.ylabel("# observation frame", fontsize=20)
    plt.show()



def plot_one_CF(fig, ax, type = "BIL", line_ = "--"):

    files = range(5,10)
    i = 0
    massINN, massNN, massKAN, Size = [],[],[],[]
    for size in range(0,13):
        arr = []
        arrNN = []
        arrKAN = []
        sizes = []
        for file in files:
            df = pd.read_csv(f"data/{Env_name}/INN_reward.dat{type}_{file}", sep=" & ", engine="python", names=["size", "AR"])
            arr.append(df.iloc[size,1])
            sizes.append(df.iloc[size,0])
        for file in files:
            df = pd.read_csv(f"data/{Env_name}/NN_reward.dat{type}_{file}", sep=" & ", engine="python", names=["size", "AR"])
            arrNN.append(df.iloc[size,1])
        for file in files:
            df = pd.read_csv(f"data/{Env_name}/KAN_reward.dat{type}_{file}", sep=" & ", engine="python", names=["size", "AR"])
            arrKAN.append(df.iloc[size,1])
        #print(arr)
        arr = np.array(arr)
        massINN.append(arr)
        arrNN = np.array(arrNN)
        massNN.append(arrNN)
        arrKAN= np.array(arrKAN)
        massKAN.append(arrKAN)
        Size.append(np.array(sizes))
        Nsigma = abs(arr.mean() - arrNN.mean())/((arr.std(ddof=1)**2 + arrNN.std(ddof=1)**2)**0.5)
        pvalue = stats.ttest_ind(np.array(arrNN), np.array(arr), equal_var = False).pvalue
        i+=1

    massINN = np.array(massINN)
    massNN = np.array(massNN)
    massKAN = np.array(massKAN)
    Size = [[0.1], [1], [2], [3], [5], [10], [100], [900], [2000], [5000], [10000], [50000], [75000]]
    Size = np.array(Size)

    ax.fill_between(Size.mean(axis=1), massINN.mean(axis=1) - massINN.std(axis=1),
                    massINN.mean(axis=1) + massINN.std(axis=1), color='blue', alpha=0.6, label="retrained INN")
    ax.fill_between(Size.mean(axis=1), massNN.mean(axis=1) - massNN.std(axis=1),
                    massNN.mean(axis=1) + massNN.std(axis=1), color='green', alpha=0.6, label="retrained NN")
    ax.fill_between(Size.mean(axis=1), massKAN.mean(axis=1) - massKAN.std(axis=1),
                    massKAN.mean(axis=1) + massKAN.std(axis=1), color='red', alpha=0.6, label="retrained KAN")
    ax.plot(Size.mean(axis=1), massINN.mean(axis=1), line_, color='blue', mec = 'black')
    ax.plot(Size.mean(axis=1), massNN.mean(axis=1), line_, color='green', mec = 'black')
    ax.plot(Size.mean(axis=1), massKAN.mean(axis=1), line_, color='red', mec = 'black')
    plt.plot(Size.mean(axis=1)[0], massINN.mean(axis=1)[0], marker="*",
             markersize=16, color="blue", lw=0, label="trained INN", mec = 'black')
    plt.plot(Size.mean(axis=1)[0], massNN.mean(axis=1)[0],
             marker="*", markersize=17,
             color="green", lw=0, label="trained NN", mec = 'black')
    plt.plot(Size.mean(axis=1)[0], massKAN.mean(axis=1)[0], marker="*",
             markersize=25, color="red", lw=0, label="trained KAN", mec = 'black')

def get_stat():

    fig, ax = plt.subplots(figsize=(1, 1))
    plot_one(fig, ax, type="BICUBIC0",plotstars=True, meccolor = "black", title = "BICUBIC",hatch = "/", color = "red")
    plot_one(fig, ax, type="BICUBIC1",meccolor = "black", title = "BILINEAR",hatch = '/', color = "green")
    plot_one(fig, ax, type="BICUBIC2", meccolor = "black", title = "NEAREST",hatch = "/", color = "blue")
    plt.legend(prop={'size': 22}) #, loc = 3)
    plt.xlabel("Compression Rate, %", fontsize=25)
    plt.ylabel("Average Reward", fontsize=25)
    #plt.ylim(bottom = -950)
    plt.title(Env_name.split("-")[0], fontdict={'fontsize':25})
    plt.grid()
    plt.savefig(f"Figs/{Env_name}_compression.png")
    plt.show()

def get_stat_CF():
    fig, ax = plt.subplots(figsize=(1, 1))
    plot_one_CF(fig, ax, type="",line_="o--")
    plt.legend(prop={'size': 22}, loc = 3)
    plt.xlabel("N events used for domain shifted training", fontsize=25)
    plt.ylabel("Average Reward", fontsize=25)
    plt.title(Env_name.split("-")[0], fontdict={'fontsize':25})
    plt.grid()
    plt.xscale('log')
    plt.savefig(f"Figs/{Env_name}_compression_CF.png")
    plt.show()


if __name__ == "__main__":
    #run_scan(test_type="INN")
    #run_scan(test_type="NN")
    #plot_reward()
    get_stat()
    get_stat_CF()
    #draw_activation()
