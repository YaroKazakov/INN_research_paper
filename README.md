# INN_RL

The repository is dedicated to solving a class of benchmark problems in the field of RL using integrated neural networks (INN) and trust region policy optimization (TRPO).

# Building
```python3.8 -m venv venv```

```source venv/bin/activate```

```pip install --upgrade pip```

```pip install --upgrade setuptools wheel```

```pip install "gymnasium[classic-control]"```

```pip install "gymnasium[box2d]"```

```pip install git+https://github.com/TheStageAI/TorchIntegral.git```

```sudo apt-get install -y libxcb-cursor-dev```

```pip install -r requirements.txt```

# Examples
#### # Train TRPO based model, convert to INN:
```python get_INN_Lunar_Landing.py``` 

#### # Compute rewards vs compression rate:
```python PostProcesses.py```

# Tests
