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
(check the [expert.py](expert.py) file for more info).

**Note:** To produce rollouts, the `expert` modify the environment (e.g. position and angle of the agent). 
Thus, when the agent has to take the "best" action, the environment is modified.
There are two main solutions:
* **For each rollout create a new environment**: 
to produce rollouts the `expert` uses a different environment and the original one remains unchanged. 
This solution is **too slow**, because at each step, we have to instantiate a new `environment` (you can try). 
* **Reset the environment after each rollout prediction**: store the environment parameters (e.g. position and angle of the agent)
before the rollout prediction and, at the end of the computation,
the environment is restored. This is a **faster** solution. 
To do so, you'll use the [set_env_params](https://github.com/FaMoSi/Duckietown-Aido4/blob/6d05e3ef26ccde7283a6f4d97e3ace311565865a/duckietown_RL/gym_duckietown/simulator.py#L609) 
function.

You can see here below the `expert` running in the [`zigzag_dists` map](https://github.com/FaMoSi/Duckietown-Aido4/blob/master/duckietown_RL/maps/zigzag_dists.yaml):

<img width="350" height="350" src="./media/gifs/duckie.gif">

## Getting Started

### Prerequisite
Remember to [install all the environment dependencies](../README.md#install-the-environment)

### Run The Expert
You can run the Expert using the [run_expert.py](./run_expert.py) script:
``` 
python run_expert.py
```
If you want to change the number of episodes and steps, you can simply
do it using the  [`EPISODES`](https://github.com/FaMoSi/Duckietown-Aido4/blob/4b3033f6037e201a2c2f8a46507f72094847dfc5/duckietown_rl/run_expert.py#L9) 
and [`STEPS`](https://github.com/FaMoSi/Duckietown-Aido4/blob/4b3033f6037e201a2c2f8a46507f72094847dfc5/duckietown_rl/run_expert.py#L8) variables.

You can also change environment settings (e.g. top-down view) in the [env.py](./env.py) file.

### Evaluation
The evaluation script allows to evaluate the performance of the Expert on different maps.
You can run the evaluation using the [evaluation.py](./evaluation.py) script:
``` 
python evaluation.py
```

This script evaluates and shows:
* The **reward** taken by the agent.
* The **distance** traveled by the agent.

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
