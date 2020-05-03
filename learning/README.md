# Duckietown - AI Driving Olympics
<a href="http://aido.duckietown.org"><img width="200" src="https://www.duckietown.org/wp-content/uploads/2018/12/AIDO_no_text-e1544555660271.png"/></a>

The [“AI Driving Olympics” (AI-DO)](http://aido.duckietown.org/) is a competition with the objective of evaluating the state of the art in machine learning and artificial intelligence for mobile robotics.

The AI Driving Olympics competition is structured into the following separate challenges:
* Lane Following - `LF`
* Lane Following + Vehicles - `LFV`
* Lane following with vehicles and intersections- `LFVI`
* Autonomous Mobility-on-Demand - `AMoD`

This project is a solution for the `LF` challenge: control of a Duckiebot to drive on the right lane on streets within Duckietown without other moving Duckiebots present. 

## Table of contents
* [Introduction](#introduction)
* [Overview](#overview)

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

## Prerequisite
* Make sure you have `pip` installed (follow [this guide]())
* This project is tested on `Python 3.7`

... duckietown-shell

**It's highly recommended to create a virtual environment `  a `conda` environment**

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
See deployment for notes on how to deploy the project on a live system.

### Installing
First of all clone this repository:

```
git clone https://github.com/FaMoSi/duckietown_aido4.git
```

Then, install all the required dependencies using the `requirements.txt` file:

```
cd learning && pip install -r requirements.txt
```

That's all, everything is installed and you can run the agent!

## Running the agent

### Collect data
Let's use the `collect_data.py` script to collect data.

``` 
python collect_data.py
```

What this is script does is use the `expert` to collect observations from the environment.
The script save this observations in the `train.log` file.

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

**[Simone Faggi](https://github.com/FaMoSi)**




