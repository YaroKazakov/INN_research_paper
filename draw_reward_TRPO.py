import matplotlib.pyplot as plt
import pandas as pd
from networkx.algorithms.bipartite.basic import color

for el in [100,20,15,5,2]:
    df = pd.read_csv(f"data/Lunar/metric_TRPO_{el}.csv", delimiter=' ', names = ["N episodes", "mean reward", "mean std reward"])
    print(df)
    #plt.yscale('log')
    line = plt.plot(df.iloc[:,0].rolling(5).mean(), df.iloc[:,1].rolling(5).mean(),label=f"mean reward {el}")
    plt.plot(df.iloc[:,0].rolling(5).mean(), df.iloc[:,2].rolling(5).mean(),"--",label=f"mean std reward {el}",color=line[0].get_color())

plt.xlabel("N episodes", fontsize=18)
plt.legend()
plt.savefig("Figs/Lunar_TRPO_training.png")
plt.show()