# Duckietown - AI Driving Olympics
<a href="http://aido.duckietown.org"><img width="200" src="https://www.duckietown.org/wp-content/uploads/2018/12/AIDO_no_text-e1544555660271.png"/></a>

The [“AI Driving Olympics” (AI-DO)](http://aido.duckietown.org/) is a competition with the objective of 
evaluating the state of the art in machine learning and artificial intelligence for mobile robotics.
The goal of the competition is to build a machine learning (ML) model that allows a self-driving "car", called `Duckiebot`, to drive on streets within `Duckietown`.

The AI Driving Olympics competition is structured into the following separate challenges:
* Lane Following - `LF` 
* Lane Following with Vehicles - `LFV`
* Lane following with Vehicles and Intersections- `LFVI`
* Autonomous Mobility-on-Demand - `AMoD`

This project is a solution for the `aido-LF` challenge: 
* **Control of a Duckiebot to drive on the right lane on streets within Duckietown without other moving Duckiebots present.**

More info about the Duckietown Project and aido-LF challenge [here](http://aido.duckietown.org/).

## Table of contents
* [Overview](#overview)
* [Prerequisite](#prerequisite) 
* [Getting Started](#getting-started)
    * [Install](#install)
    * [Collect Data](#collect-data) 
    * [Train The Model](#train-the-model) :construction_worker:
* [Submit](#submit) :construction_worker:
* [Author](#author)

## Overview

The approach is to train a Reinforcement Learning (RL) agent to build an `expert` that drives 
perfectly within an environment, then use this `expert` to collect data. 
Data are pairs `<observation, actions>`collected in different maps/environments used to train an agent that imitates the expert's behaviour (Imitation Learning / Behavioural Cloning) .
Finally, you have a self-driving car that navigates within Duckietown using only one single sensor, the camera.

The `expert`, at each step, computes a prediction tree 
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
To do so, I've implemented the [set_env_params](https://github.com/FaMoSi/Duckietown-Aido4/blob/6d05e3ef26ccde7283a6f4d97e3ace311565865a/expert_RL/gym_duckietown/simulator.py#L609) 
function.

You can see here below the `expert` running in the `zigzag_dists` [map](https://github.com/FaMoSi/Duckietown-Aido4/blob/master/expert_RL/maps/zigzag_dists.yaml):

<img width="350" height="350" src="expert_RL/media/gifs/duckie.gif">

Trough this tutorial you will:
* Use a [Reinforcement Learning](expert_RL) approach (Decision Tree) to build an `expert`.
* Use the `expert` to [collect data](duckiebot_IL/collect_data.py).
* Use an [Imitation Learning](duckiebot_IL) approach to train the `Duckiebot`. (:construction_worker: **Work in progres...**)
* Submit the solution to the [“AI Driving Olympics” (AI-DO)](http://aido.duckietown.org/). (:construction_worker: **Work in progres...**)

## Prerequisite
**It's highly recommended to create a virtual environment using `virtualenv` or `anaconda`**

Before proceeding:
* Make sure you have `pip` installed.
* This project is tested on `Python 3.7`.
* Install the `duckietown-shell` following [the official guide](https://github.com/duckietown/duckietown-shell/blob/daffy-aido4/README.md).

## Getting Started
### Install The Environment

Clone this repository:
```
git clone https://github.com/FaMoSi/duckietown_aido4.git
```

Change into it:
```
cd duckietown_aido4
```

Install all the dependencies using the [requirements.txt](requirements.txt) file:

```
pip install -r requirements.txt
```

That's all, everything is installed and you can run all the scripts in this project!

## Author
* **[Simone Faggi](https://github.com/FaMoSi)**
* Email: **simone.faggi@yahoo.it**




