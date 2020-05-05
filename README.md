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

This project is a solution for the `LF challenge`: 
* **Control of a Duckiebot to drive on the right lane on streets within Duckietown without other moving Duckiebots present.**

More info about the Duckietown Project [here](http://aido.duckietown.org/).

## Table of contents
* [Overview](#overview)
* [Prerequisite](#prerequisite)
* [Getting Started](#getting-started)
    * [Install](#install)
    * [Collect Data](#collect-data)
    * [Train The Model](#train-the-model)
* [Submit](#submit)
* [Author](#author)

## Overview

The approach is to build an `expert` and use it to collect data.
Collected data are used to train a neural network and build a model to predict which action to take.
The `expert`, at each step, produces a prediction tree and takes the action that maximize the reward.

You can see here below the `expert` running:

<img width="350" height="350" src="gifs/topview.gif">

Trough this tutorial you will:
* Use the expert to collect data.
* Train a neural network and build the final model.
* Submit the solution to the [“AI Driving Olympics” (AI-DO)](http://aido.duckietown.org/).
  
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
cd duckietown_aido4/learning
```

Install all the dependencies using the [requirements.txt](learning/requirements.txt) file:

```
pip install -r requirements.txt
```

That's all, everything is installed and you can run the agent!

### Collect data
Let's collect data using the [collect_data.py](learning/collect_data.py) script.

What this is script does is:
* Run an `expert` on a variety of gym-duckietown `maps` (see [maps](learning/maps)).  
* Record the actions it takes and save them (pairs `<observation, action>`) in the `train.log` file.

An important aspect is the number of samples. 
It is possible to increase/decrease the number of samples increasing/decreasing 
the value of `STEPS` and/or `EPISODES` in the [collect_data.py](learning/collect_data.py) file.

Run the expert and collect data:

``` 
python collect_data.py
```

### Train the model 
Given the `expert` trajectories (which we recorded in the previous [section](#collect-data)), 
we want to learn a policy that learns from those trajectories.

To do so, you will train a neural network running the [train.py](learning/train.py) script 
that uses the collected trajectories.

``` 
python train.py
```

:construction_worker:

Work in progres...

## Submit
**Remember to install the `duckietown-shell` in order to make a submission.**

To submit the `agent` for the evaluation you need to copy the relevant files from the `learning/` directory to the `submission/` one. 
In particular, you will need to overwrite `submission/model.py` to match any update you’ve made to the model, 
and place your final model inside of `submission/tf_models/` so you can load it correctly. 

Now you are ready to make a submission for the `LF` challenge:

```
dts challengs submit
```

Congreats! You made a submission for the [“AI Driving Olympics” (AI-DO)](http://aido.duckietown.org/) `LF challange`s

## Author
* **[Simone Faggi](https://github.com/FaMoSi)**
* Email: **simone.faggi@yahoo.it**




