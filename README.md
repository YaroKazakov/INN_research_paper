# Kolmogorov–Arnold and Integral Neural Networks in the context of Imitation Learning: robustness to network resizing and catastrophic interference

The project allows to represent an agent’s policy models as INN and KAN. By utilizing open
RL benchmarks (Acrobot, Lunar Lander and on continuous control tasks Swimmer, Walker2D, Half Cheetah and Ant), we have demonstrated that INN representation provides flexible control over the degree of network
discretization without necessitating retraining across a broad
spectrum of model compression levels. This allows users to
directly tailor the balance between network size and accuracy. On the orher hand we showed that KAN representation of the agent policy allows the agent to delay the onset of
catastrophic forgetting to a later stage.

# Building
```. setup.sh```

# Usage examples
#### # run TRPO to get expert::

```python main.py --train_TRPO_expert 1```

#### # Convert expert to INN agent:

```python main.py --train_INN_agent 1``` 

#### # Plot rewards vs compression rate:
```python PostProcesses.py```

#### # Run Lunar Lander environment with INN
```python main.py --run_INN_test 1```


# Demo with INN policy at original size
![myfile](Ant-v4_original.gif)

# Demo with INN policy 24.3\% compressed
![myfile](Ant-v4_24.3percentage_compressed.gif)
