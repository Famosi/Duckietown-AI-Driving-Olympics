# Reinforcement Learning - Decision Tree
This folder contains all the scripts required for the Reinforcement Learning (Descion Tree) based approach.

## Table of contents
* [Description](#description)
* [Getting Started](#getting-started)
    * [Prerequisite](#prerequisite)
    * [Run The Expert](#run-the-expert) 
    * [Evaluation](#evaluation)
* [Folder Details](#folder-details)
* [Author](#author)

## Description
The `expert`, at each step, computes the prediction tree 
and uses information from the simulated environment (e.g. the distance from the center of the right lane) 
to calculate the Q-value of each state. Then, it takes the action that leads to the state that maximize the [reward](https://github.com/FaMoSi/Duckietown-Aido4/blob/6d05e3ef26ccde7283a6f4d97e3ace311565865a/exper_RL/expert.py#L164) 
(check the [expert.py](expert_RL/expert.py) file for more info).

**Note:** To produce rollouts, the `expert` modify the environment (e.g. position and angle of the agent). 
Thus, when the agent has to take the "best" action, the environment is modified.
There are two main solutions:
* **For each rollout create a new environment**: 
to produce rollouts the `expert` uses a different environment and the original one remains unchanged. 
This solution is **too slow**, because at each step, we have to instantiate a new `environment` (you can try). 
* **Reset the environment after each rollout prediction**: store the environment parameters (e.g. position and angle of the agent)
before the rollout prediction and, at the end of the computation,
the environment is restored. This is a **faster** solution. 
To do so, you'll use the [set_env_params](https://github.com/FaMoSi/Duckietown-Aido4/blob/6d05e3ef26ccde7283a6f4d97e3ace311565865a/expert_RL/gym_duckietown/simulator.py#L609) 
function.

You can see here below the `expert` running in the [`zigzag_dists` map](https://github.com/FaMoSi/Duckietown-Aido4/blob/master/expert_RL/maps/zigzag_dists.yaml):

<img width="350" height="350" src="./media/gifs/duckie.gif">

## Getting Started

### Prerequisite
Remember to [install all the dependencies](../README.md#prerequisite)

### Run The Expert
:construction_worker:

**Description...**

``` 
python run_expert.py
```

### Evaluation
:construction_worker:

**Description...**

``` 
python evaluation.py
```

## Folder Details
<details>
<summary><b><i>gym_duckietown/</i></b></summary>
</details>

<details>
<summary><b><i>maps/</i></b></summary>
</details>

<details>
<summary><b><i>utils/</i></b></summary>
</details>

<details>
<summary><b><i>env.py</i></b></summary>
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-b0Ng1A5C34iSTqXOxWZufNmpImWMCUb?usp=sharing)
</details>

<details>
<summary><b><i>expert.py.py</i></b></summary>
</details>

<details>
<summary><b><i>collect_data.py</i></b></summary>
</details>

<details>
<summary><b><i>evaluation.py.py</i></b></summary>
</details>

<details>
<summary><b><i>manual_control.py</i></b></summary>
</details>

## Author
* **[Simone Faggi](https://github.com/FaMoSi)**
* Email: **simone.faggi@yahoo.it**
