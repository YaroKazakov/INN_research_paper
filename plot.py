import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("data/data_training.csv",delimiter=' ')
plt.hist(df.iloc[:,-1])
plt.show()