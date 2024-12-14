import matplotlib.pyplot as plt
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
import pandas as pd
from src.TRPO.optimize import Env_name
from networkx.algorithms.bipartite.basic import color

for el in [100]:
    df = pd.read_csv(f"data/{Env_name}/metric_TRPO_{el}.csv", delimiter=' ', names = ["N episodes", "mean reward", "mean std reward"])
    print(df)
    #plt.yscale('log')
    line = plt.plot(df.iloc[:,0].rolling(5).mean(), df.iloc[:,1].rolling(5).mean(),label=f"Expert's Average Reward")
    plt.plot(df.iloc[:,0].rolling(5).mean(), df.iloc[:,2].rolling(5).mean(),"--",label=f"STD",color=line[0].get_color())

plt.xlabel("N episodes", fontsize=12)
plt.legend()
plt.savefig("Figs/Lunar_TRPO_training.png")
plt.show()