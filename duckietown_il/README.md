# Imitation Learning - Behavioral Cloning
This folder contains all the scripts required for the Imitation Learning approach.

* [Description](#description)
* [Getting Started](#getting-started)
    * [Prerequisite](#prerequisite)
    * [Collect Data](#collect-data) 
    * [Train](#train-the-model)
    * [Evaluate](#evaluate-the-model)
* [Folder Details](#folder-details)
* [Author](#author)

## Description
:construction_worker:

**Work in progres...**

You can see here below the trained `Duckiebot` running in two different [maps](https://github.com/FaMoSi/Duckietown-Aido4/blob/master/duckietown_RL/maps):

<img width="350" height="350" alt="Trained Duckiebot" src="../media/gifs/duckiebot.gif">

## Getting Started

### Prerequisite
Remember to [install all the dependencies](../README.md#prerequisite)

### Collect data
Let's collect data using the [collect_data.py](collect_data.py) script.

What this is script does is:
* Run an `expert` on a variety of `maps` (see [maps](maps)).  
* Record the actions it takes and save pairs `<observation, action>` in the `.log` file.

An important aspect is the number and the variety of samples:
* To increase/decrease the number of samples you can increase/decrease 
the value of `STEPS` and/or `EPISODES` in the [collect_data.py](collect_data.py) file.
* It is possible to run the `expert` on a single/variety of gym-duckietown `maps`. 
To do so use the `randomize_maps_on_reset` parameters of the class `Simulator` (see [env.py](duckiebot_IL/env.py)).

In general, you can use the parameters of the `Simulator` class
to change the environment settings.

Run the expert and collect data:
``` 
python collect_data.py
```

### Train the model 
Now that you've collected data using the `Expert`, you'll train an agent that drives safely on the streets within Duckietown.
The model you'll train is a Neural Network (NN) that takes as input an observation and learn to gives in output the right action t
he Duckiebot has to perform to drive safely.

The architecture of the NN you'll train is taken from the paper 

In the [model.py](./model.py) file you can find three different models:
* **VGG16_model** is the implementation of the VGG16 architecture.
* **NVIDIA_model** is taken from the paper ["End-to-End Deep Learning for Self-Driving Cars" by By M. Bojarski (2016)](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/). 
This is the architecture you'll use to train the model.
* **Model_a_d** is the architecture to train a NN that predicts `angles` and `displacement` given an `observation`.

Train the model:
``` 
python train_actions.py
```

### Evaluate the model
Once you've trained the model you can evaluate its performance. 
You can do that using the [eval_actions.py](./eval_actions.py) script. With this script you will:
* see the Duckiebot that drives using the trained model.
* check the reward taken by the agent during its driving.
* plot images to check Predictions VS. Ground Truth. In this case you need to uncomment this [lines](https://github.com/FaMoSi/Duckietown-Aido4/blob/869d1bb9198a7cbdd5da0c0603677c722706a6b7/duckietown_il/eval_actions.py#L67):
``` 
# # Plot Predictions VS. Ground Truth
# fig = plt.figure(figsize=(40, 30))
# i = 0
# for prediction, gt, img in zip(predictions_25Ep, gts, observations):
#     fig, ax = plt.subplots(1, 1, constrained_layout=True)
#     ax.imshow(img)
#     ax.set_title(f"Pred: [{prediction[0]:.3f}, {prediction[1]:.3f}]\n"
#                  f"GT: [{gt[0]:.3f}, {gt[1]:.3f}]")
#     i += 1
#
# plt.show()
```

Evaluate the model:
``` 
python eval_actions.py
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