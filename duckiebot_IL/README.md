# Imitation Learning - Behavioral Cloning
This folder contains all the scripts required for the Imitation Learning approach.

* [Description](#description)
* [Getting Started](#getting-started)
    * [Prerequisite](#prerequisite)
    * [Collect Data](#collect-data) 
    * [Train](#train-the-model)
* [Folder Details](#folder-details)
* [Author](#author)

## Description
:construction_worker:

**Work in progres...**

## Getting Started

### Prerequisite
Remember to [install all the dependencies](../README.md#prerequisite)

### Collect data
Let's collect data using the [collect_data.py](duckiebot_IL/collect_data.py) script.

What this is script does is:
* Run an `expert` on a variety of `maps` (see [maps](duckiebot_IL/maps)).  
* Record the actions it takes and save them (pairs `<observation, action>`) in the `train.log` file.

An important aspect is the number and the variety of samples:
* To increase/decrease the number of samples you can increase/decrease 
the value of `STEPS` and/or `EPISODES` in the [collect_data.py](duckiebot_IL/collect_data.py) file.
* It is possible to run the `expert` on a single/variety of gym-duckietown `maps`. 
To do so use the `randomize_maps_on_reset` parameters of the class `Simulator` (see [env.py](duckiebot_IL/env.py)).

In general, you can use the parameters of the `Simulator` class
to change the environment settings.

Run the expert and collect data:
``` 
python collect_data.py
```

### Train the model 

:construction_worker:

**Work in progres...**

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

## Author
* **[Simone Faggi](https://github.com/FaMoSi)**
* Email: **simone.faggi@yahoo.it**