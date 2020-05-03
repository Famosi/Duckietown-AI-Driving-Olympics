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

This project is a solution for the `LF` challenge: 
* **control of a Duckiebot to drive on the right lane on streets within Duckietown without other moving Duckiebots present.**

More info about the Duckietown Project [here](http://aido.duckietown.org/).

## Table of contents
* [Introduction](#introduction)
* [Overview](#overview)
* [Prerequisite](#prerequisite)
* [Getting Started](#getting-started)
    * [Install](#install)
    * [Run](#run-the-agent)
* [Submit](#submit)
* [Author](#author)

## Overview

The approach is to build an "expert" using reinforcement learning (RL) and use this expert to collect data.
Collected data are used to train a neural network and build a model to predict, at each step, which action to take.

The expert, at each step, produces a prediction tree and takes the action that maximize the reward.

```
# Compute reward
reward = (
        + speed * self.cof_speed
        + align * self.cof_align
        - dist * self.cof_dist
        + not_derivable
)
```

![topview-gif]()

## Prerequisite
* Make sure you have `pip` installed
* This project is tested on `Python 3.7`
* Install the `duckietown-shell` following [this guide](https://github.com/duckietown/duckietown-shell/blob/daffy-aido4/README.md)

**It's highly recommended to create a virtual environment using `virtualenv` or `anaconda`**

## Getting Started
Now you have all the requisite to continue.

### Install
First of all, clone this repository:

```
git clone https://github.com/FaMoSi/duckietown_aido4.git
```

Then, install all the required dependencies using the `requirements.txt` file:

```
cd learning && pip install -r requirements.txt
```

That's all, everything is installed and you can run the agent!

## Run the agent

### Collect data
Let's use the `collect_data.py` script to collect data.

``` 
python collect_data.py
```

What this is script does is use the `expert` to collect observations from the environment.
The script saves these observations in the `train.log` file.

### Train the model 

## Submit
**Remember to install the `duckietown-shell` in order to make a submission.**

You need to copy the relevant files from the `learning/` directory to the `submission/` one. 
In particular, you will need to overwrite `submission/model.py` to match any update you’ve made to the model, 
and place your final model inside of `submission/tf_models/` so you can load it correctly. 

Now you are ready to make a submission for the `LF` challenge:

```
dts challengs submit
```

## Author

* **[Simone Faggi (GitHub)](https://github.com/FaMoSi)**
* Email: simone.faggi@yahoo.it




