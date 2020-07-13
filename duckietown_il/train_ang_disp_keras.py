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
from duckietown_il.model_keras import VGG16_model, NVIDIA_model, Model_a_d
from keras.optimizers import SGD, Adam
from keras.models import Model, Input
from keras.layers import Concatenate
from keras.losses import mean_squared_error as MSE
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from random import randrange


# Function to plot model's validation loss and validation accuracy
def plot_model_history(model_history, path_to_save, model_name):
    fig, axs = plt.subplots(2, 2, figsize=(25, 8))
    # summarize history for DIST accuracy
    axs[0][0].plot(range(1, len(model_history.history['displacement_accuracy']) + 1),
                   model_history.history['displacement_accuracy'])
    axs[0][0].plot(range(1, len(model_history.history['val_displacement_accuracy']) + 1),
                   model_history.history['val_displacement_accuracy'])
    axs[0][0].set_title('Model Accuracy DIST')
    axs[0][0].set_ylabel('Accuracy')
    axs[0][0].set_xlabel('Epoch')
    axs[0][0].set_xticks(np.arange(1, len(model_history.history['displacement_accuracy']) + 1),
                         len(model_history.history['displacement_accuracy']) / 10)
    axs[0][0].legend(['train', 'val'], loc='best')

    # summarize history for ANGLE accuracy
    axs[0][1].plot(range(1, len(model_history.history['angle_accuracy']) + 1),
                   model_history.history['angle_accuracy'])
    axs[0][1].plot(range(1, len(model_history.history['val_angle_accuracy']) + 1),
                   model_history.history['val_angle_accuracy'])
    axs[0][1].set_title('Model Accuracy ANGLE')
    axs[0][1].set_ylabel('Accuracy')
    axs[0][1].set_xlabel('Epoch')
    axs[0][1].set_xticks(np.arange(1, len(model_history.history['angle_accuracy']) + 1),
                         len(model_history.history['angle_accuracy']) / 10)
    axs[0][1].legend(['train', 'val'], loc='best')

    # summarize history for DIST loss
    axs[1][0].plot(range(1, len(model_history.history['displacement_loss']) + 1),
                   model_history.history['displacement_loss'])
    axs[1][0].plot(range(1, len(model_history.history['val_displacement_loss']) + 1),
                   model_history.history['val_displacement_loss'])
    axs[1][0].set_title('Model Loss DIST')
    axs[1][0].set_ylabel('Loss')
    axs[1][0].set_xlabel('Epoch')
    axs[1][0].set_xticks(np.arange(1, len(model_history.history['displacement_loss']) + 1),
                         len(model_history.history['displacement_loss']) / 10)
    axs[1][0].legend(['train', 'val'], loc='best')

    # summarize history for ANGLE loss
    axs[1][1].plot(range(1, len(model_history.history['angle_loss']) + 1),
                   model_history.history['angle_loss'])
    axs[1][1].plot(range(1, len(model_history.history['val_angle_loss']) + 1),
                   model_history.history['val_angle_loss'])
    axs[1][1].set_title('Model Loss ANGLE')
    axs[1][1].set_ylabel('Loss')
    axs[1][1].set_xlabel('Epoch')
    axs[1][1].set_xticks(np.arange(1, len(model_history.history['angle_loss']) + 1),
                         len(model_history.history['angle_loss']) / 10)
    axs[1][1].legend(['train', 'val'], loc='best')

    fig.tight_layout(pad=3.0)
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
reader = Reader(f'{DATA}.log')  # where our data lies
MODEL_NAME = "01_NVIDIA"
# MODEL_NAME = "VGG_16"


#################### Data PRE-PROCESSING ####################
observations, _, _, info = reader.read()  # read the observations from data
observations = np.array(observations)
angles = np.array([i['Simulator']['lane_position']['angle_deg'] for i in info])
dists = np.array([i['Simulator']['lane_position']['dist'] for i in info])

df = pd.DataFrame({'angles': angles, 'dists': dists})


def hot_encoding(dataframe, arg, dict, where_to_cut, label_names):
    dataframe[dict[arg]] = pd.cut(dataframe[arg], where_to_cut, labels=label_names)
    return dataframe


# Displacement: 0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2 (and negatives)
# Angle: 0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90 (and negatives)

labels_dist = ["-0.15/-0.135", "-0.135/-0.12", "-0.12/-0.105", "-0.105/-0.09", "-0.09/0.075",
               "-0.075/-0.06", "-0.06/-0.045", "-0.045/-0.03", "-0.03/-0.015", "-0.015/0",
               "0/0.015", "0.015/0.03", "0.03/0.045", "0.045/0.06", "0.06/0.075",
               "0.075/0.09", "0.09/0.105", "0.105/0.12", "0.12/0.135", "0.135/0.15"]
cut_points_dist = [-0.15, -0.135, -0.12, -0.105, -0.09,
                   -0.075, -0.06, -0.045, -0.03, -0.015, 0.0,
                   0.015, 0.03, 0.045, 0.06, 0.075,
                   0.09, 0.105, 0.12, 0.135, 0.15]

labels_angle = ["-45/-41.5", "-41.5/-36", "-36/-31.5", "-31.5/-27", "-27/-22.5",
                "-22.5/-18", "-18/-13.5", "-13.5/-9", "-9/-4.5", "-4.5/0",
                "0/4.5", "4.5/9", "9/13.5", "13.5/18", "18/22.5",
                "22.5/27", "27/31.5", "31.5/36", "36/41.5", "41.5/45"]
cut_points_angle = [-45, -41.5, -36, -31.5, -27, -22.5,
                    -18, -13.5, -9, -4.5, 0,
                    4.5, 9, 13.5, 18, 22.5,
                    27, 31.5, 36, 41.5, 45]

dict = {
    "angles": "angles_cat",
    "dists": "dists_cat"
}

df = hot_encoding(df, "angles", dict, cut_points_angle, labels_angle)
df = hot_encoding(df, "dists", dict, cut_points_dist, labels_dist)


def create_dummies(dataframe, column_name):
    dummies = pd.get_dummies(dataframe[column_name], prefix=column_name)
    dataframe = pd.concat([dataframe, dummies], axis=1)
    return dataframe


df = create_dummies(df, dict["dists"])
df = create_dummies(df, dict["angles"])

targets_dist = [dict["dists"] + '_' + col_name for col_name in labels_dist]
targets_angle = [dict["angles"] + '_' + col_name for col_name in labels_angle]

# ################## DISTRIBUTE DATA ####################
def sort_samples(df, targets):
    data_sorted = []
    for label in targets:
        obs_category = []
        idxs = df.index[df[label] == 1].tolist()
        for i in idxs:
            obs_category.append([observations[i], i])
        data_sorted.append(obs_category)

    return data_sorted


sort_angle = sort_samples(df, targets_angle)
sort_dist = sort_samples(df, targets_dist)


def sample_for_training(arr):
    samples = []
    for entry in arr:
        for _ in range(0, 60):
            try:
                idx = entry[randrange(len(entry))][1]
                samples.append([observations[idx], df['angles'][idx], df['dists'][idx]])
            except:
                continue
    return samples


data = np.concatenate((np.array(sample_for_training(sort_angle)), np.array(sample_for_training(sort_dist))), axis=0)

df = pd.DataFrame({'angles': data[:, 1], 'dists': data[:, 2]})

df = hot_encoding(df, "angles", dict, cut_points_angle, labels_angle)
df = hot_encoding(df, "dists", dict, cut_points_dist, labels_dist)

df = create_dummies(df, dict["dists"])
df = create_dummies(df, dict["angles"])

x_train = np.stack(data[:, 0])
y_dist = df[targets_dist]
y_angle = df[targets_angle]

# # Split the data: Train and Test
x_train, x_test, y_train_dist, y_test_dist, y_train_angle, y_test_angle = \
    train_test_split(
        x_train, y_dist, y_angle, test_size=0.2, random_state=2
    )

# Split Train data once more for Validation data
val_size = int(len(x_train) * 0.1)
# DIST
x_validate, y_validate_dist = x_train[:val_size], y_train_dist[:val_size]
y_train_dist = y_train_dist[val_size:]
# ANGLE
y_validate_angle = y_train_angle[:val_size]
x_train, y_train_angle = x_train[val_size:], y_train_angle[val_size:]


# prepare data augmentation configuration
train_datagen = ImageDataGenerator(rescale=1./255,              # rescaling factor
                                   width_shift_range=0.2,       # float: fraction of total width, if < 1
                                   height_shift_range=0.2,      # float: fraction of total height, if < 1
                                   brightness_range=None,       # Range for picking a brightness shift value from
                                   zoom_range=0.0,              # Float or [lower, upper]. Range for random zoom
                                   )

train_datagen.fit(x_train)

# this is the augmentation configuration we will use for validating: only rescaling
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen.fit(x_validate)

# this is the augmentation configuration we will use for testing: only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)
test_datagen.fit(x_test)

# #################### BUILD the model ####################
inputs = Input(shape=(60, 120, 3))
angle_model, dist_model = Model_a_d(inputs)

model = Model(
    inputs=inputs,
    outputs=[angle_model, dist_model]
)

optimizer = Adam(lr=1e-4, decay=1e-4 / EPOCHS)
# losses = {
#     "angles": "categorical_crossentropy",
#     "displacement": "categorical_crossentropy",
# }
# lossWeights = {"angles": 1.0, "displacement": 1.0}

model.compile(optimizer=optimizer,
              loss=['categorical_crossentropy', 'categorical_crossentropy'],
              metrics=["accuracy"])

model.summary()

# #################### TRAIN AND SAVE the model ####################

es = EarlyStopping(monitor='val_loss', verbose=1, patience=30)
mc = ModelCheckpoint(STORAGE_LOCATION + MODEL_NAME + '.h5', monitor='val_loss', save_best_only=True)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb = TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(x=x_train,
                    y={"angle": y_train_angle, "displacement": y_train_dist},
                    validation_data=(x_validate,
                                     {"angle": y_validate_angle, "displacement": y_validate_dist}),
                    epochs=EPOCHS,
                    verbose=2,
                    callbacks=[es, mc, tb],
                    shuffle=True
                    )

# #################### PLOT AND SAVE ####################

plot_model_history(history, path_to_save=STORAGE_LOCATION, model_name=MODEL_NAME)
# Test the model on the test set
# test_result = model.evaluate(x_test, y_train_dist, y_test_angle)

test_result = model.evaluate(x=x_test,
                             y={"angle": y_test_angle, "displacement": y_test_dist},
                             batch_size=BATCH_SIZE)

print(f"Test loss: {test_result[0]:.3f}\t | Test accuracy: %{test_result[1]:.2f}")
