# INN_RL

The repository is dedicated to learn at benchmark problem Lunar Lander v3 using integral neural networks (INN) and trust region policy optimization (TRPO).

# Building
```. setup.sh```

#### # run TRPO to get expert::

```python main.py --train_TRPO_expert 1```

#### # Convert expert to INN agent:

```python get_INN_Lunar_Landing.py --train_INN_agent 1``` 

#### # Plot rewards vs compression rate:
```python PostProcesses.py```

#### # Run Lunar Lander environment with INN
```python get_INN_Lunar_Landing.py --run_INN_test 1```


# Demo with INN policy at original size
![myfile](Ant-v4_original.gif)

# Demo with INN policy 24.3\% compressed
![myfile](Ant-v4_24.3percentage_compressed.gif)