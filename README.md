# Kolmogorov–Arnold and Integral Neural Networks in the context of Imitation Learning: robustness to network resizing and catastrophic interference

This is the repo for our paper that examines the issue of catastrophic forgetting and online resizing in the context of imitation learning.

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
