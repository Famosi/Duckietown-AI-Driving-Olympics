import sys

sys.path.append("../")
import numpy as np
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from duckietown_il._loggers import Reader
import pandas as pd
from duckietown_il.model_keras import VGG16_model, NVIDIA_model, NVIDIA_model_2
from keras.optimizers import SGD, Adam
from keras.models import Model, Input
from keras.layers import Concatenate
from keras.losses import mean_squared_error as MSE
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


# Function to plot model's validation loss and validation accuracy
def plot_model_history(model_history, path_to_save, model_name):
    fig, axs = plt.subplots(1, 2, figsize=(25, 8))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
    axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['accuracy']) + 1),
                      len(model_history.history['accuracy']) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.savefig(path_to_save + '/' + model_name + '_model_history.png')

    plt.show()


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--data", required=True, type=str, help="name of the data to learn from (without .log)")
ap.add_argument("-e", "--epoch", required=True, type=int, help="number of epochs")
ap.add_argument("-b", "--batch-size", required=True, type=int, help="batch size")
args = vars(ap.parse_args())
DATA = args["data"]

# configuration zone
BATCH_SIZE = args["batch_size"]  # define the batch size
EPOCHS = args["epoch"]  # how many times we iterate through our data
STORAGE_LOCATION = "trained_models/"  # where we store our trained models
reader = Reader(f'../{DATA}.log')  # where our data lies
MODEL_NAME = "01_NVIDIA"
# MODEL_NAME = "VGG_16"


#################### Data PRE-PROCESSING ####################
observations, _, angles, info = reader.read()  # read the observations from data
observations = np.array(observations)
angles = np.array(angles)
dist = np.array([i['Simulator']['lane_position']['dist'] for i in info])

df = pd.DataFrame({'angles': angles, 'dists': dist})


def process_data(dataframe, arg, dict, where_to_cut, label_names):
    # Handle missing values
    dataframe[dict[arg]] = pd.cut(dataframe[arg], where_to_cut, labels=label_names)
    return dataframe


# Angle: 0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2
# Displacement: 0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90

labels_dist = ["-0.2/-0.18", "-0.18/-0.16", "-0.16/-0.14", "-0.14/-0.12", "-0.12/0.1",
               "-0.1/-0.08", "-0.8/-0.6", "-0.6/-0.4", "-0.4/-0.2", "-0.2/0.0",
               "0.0/0.02", "0.02/0.04", "0.04/0.06", "0.06/0.08", "0.08/0.1",
               "0.1/0.12", "0.12/0.14", "0.14/0.16", "0.16/0.18", "0.18/0.2"]
cut_points_dist = [-0.2, -0.18, -0.16, -0.14, -0.12, -0.1,
                   -0.08, -0.06, -0.04, -0.02, 0.0,
                   0.02, 0.04, 0.06, 0.08, 0.1,
                   0.12, 0.14, 0.16, 0.18, 0.2]

labels_angle = ["-90/-81", "-81/-72", "-72/-63", "-63/-54", "-54/-45",
                "-45/-36", "-36/-27", "-27/-18", "-18/-9", "-9/0",
                "0/9", "9/18", "18/27", "27/36", "36/45",
                "45/54", "54/63", "63/72", "72/81", "81/90"]
cut_points_angle = [-90, -81, -72, -63, -54, -45,
                    -36, -27, -18, -9, 0,
                    9, 18, 27, 36, 45,
                    54, 63, 72, 81, 90]

dict = {
    "angles": "angles_cat",
    "dists": "dists_cat"
}

df = process_data(df, "angles", dict, cut_points_angle, labels_angle)
df = process_data(df, "dists", dict, cut_points_dist, labels_dist)


def create_dummies(dataframe, column_name):
    dummies = pd.get_dummies(dataframe[column_name], prefix=column_name)
    dataframe = pd.concat([dataframe, dummies], axis=1)
    return dataframe


df = create_dummies(df, dict["dists"])
df = create_dummies(df, dict["angles"])

targets_dist = [dict["dists"] + '_' + col_name for col_name in labels_dist]
targets_angle = [dict["angles"] + '_' + col_name for col_name in labels_angle]

x_train = observations
y_label_dists = df[targets_dist]
y_label_angle = df[targets_angle]

# # Split the data: Train and Test
x_train, x_test, y_train_dists, y_test_dists, y_train_angle, y_test_angle = \
    train_test_split(
        x_train, y_label_dists, y_label_angle, test_size=0.2, random_state=2
    )

# #################### BUILD the model ####################
inputs = Input(shape=(60, 120, 3))
dist_model = NVIDIA_model_2(inputs, "dist_output")
angle_model = NVIDIA_model_2(inputs, "angle_output")

model = Model(
    inputs=inputs,
    outputs=[dist_model, angle_model]
)

optimizer = Adam(lr=1e-3, decay=1e-3 / EPOCHS)
losses = {
    "dist_output": "categorical_crossentropy",
    "angle_output": "categorical_crossentropy",
}
lossWeights = {"dist_output": 1.0, "angle_output": 1.0}

model.compile(optimizer=optimizer,
              loss=losses,
              loss_weights=lossWeights,
              metrics=["accuracy"])

model.summary()

# #################### TRAIN the model ####################

history = model.fit(x=x_train,
                    y={"dist_output": y_train_dists, "angle_output": y_train_angle},
                    validation_data=(x_test,
                                     {"dist_output": y_test_dists, "angle_output": y_test_angle}),
                    epochs=EPOCHS,
                    verbose=1)

# #################### PLOT AND SAVE the model ####################

plot_model_history(history, path_to_save=STORAGE_LOCATION, model_name=MODEL_NAME)
# Test the model on the test set
# test_result = model.evaluate(x_test, y_train_dists, y_test_angle)

test_result = model.evaluate(x=x_test,
               y={"dist_output": y_test_dists, "angle_output": y_test_angle},
               batch_size=BATCH_SIZE)

print(f"Test loss: {test_result[0]:.3f}\t | Test accuracy: %{test_result[1]:.2f}")
