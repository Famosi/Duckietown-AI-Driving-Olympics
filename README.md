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
* [Description](#descritpion)
* [Prerequisite](#prerequisite) 
* [Getting Started](#getting-started)
    * [Install](#install)
* [Author](#author)

## Description

The approach is to train a Reinforcement Learning (RL) agent to build an `expert` that drives 
perfectly within an environment, then use this `expert` to collect data. 
Data are pairs `<observation, actions>`collected in different maps/environments used to train an agent that imitates the expert's behaviour (Imitation Learning / Behavioural Cloning) .
Finally, you have a self-driving car that navigates within Duckietown using only one single sensor, the camera.

Trough this tutorial you will:
* Use a [Reinforcement Learning](expert_RL) approach (Decision Tree) to build an `expert`.
* Use the `expert` to [collect data](duckiebot_IL/collect_data.py).
* Use an [Imitation Learning](duckiebot_IL) approach to train the `Duckiebot`. (:construction_worker: **Work in progres...**)
* [Submit](submission) the solution to the [“AI Driving Olympics” (AI-DO)](http://aido.duckietown.org/). (:construction_worker: **Work in progres...**)

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




