import os
import json
import numpy as np
import pandas as pd
from collections import Counter

from PIL import Image
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras
from tensorflow.keras.applications import EfficientNetB3
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=tf_config)

workdir = "/home/rakesh/leaf_detection"
train_path = workdir + "/train_images"

data = pd.read_csv(workdir + '/train.csv')
f = open(workdir + '/label_num_to_disease_map.json')
real_labels = json.load(f)
real_labels = {int(k):v for k,v in real_labels.items()}

data['class_name'] = data.label.map(real_labels)

train, val = train_test_split(data, test_size=0.2, random_state=42, stratify=data['class_name'])

IMG_SIZE = 244
size = (IMG_SIZE, IMG_SIZE)
n_class = 5

datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
            rotation_range=40,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            vertical_flip = True,
            fill_mode = 'nearest'
)

train_set = datagen.flow_from_dataframe(
                train,
                directory=train_path,
                seed=42,
                x_col="image_id",
                y_col="class_name",
                target_size=size,
                class_mode="categorical",
                interpolation="nearest",
                shuffle=True,
                batch_size=32
)

val_set = datagen.flow_from_dataframe(val,
                         directory = train_path,
                         seed=42,
                         x_col = 'image_id',
                         y_col = 'class_name',
                         target_size = size,
                         class_mode = 'categorical',
                         interpolation = 'nearest',
                         shuffle = True,
                         batch_size = 32
)

model = Sequential()
model.add(EfficientNetB3(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"))
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(512, activation="relu", bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)))
model.add(Dropout(0.5))
model.add(Dense(n_class, activation="softmax"))

model.summary()

EPOCHS = 50
STEP_SIZE_TRAIN = train_set.n//train_set.batch_size
STEP_SIZE_VALID = val_set.n//val_set.batch_size
print(STEP_SIZE_TRAIN)
print(STEP_SIZE_VALID)

keras.utils.plot_model(model)

EPOCHS = 50
STEP_SIZE_TRAIN = train_set.n//train_set.batch_size
STEP_SIZE_VALID = val_set.n//val_set.batch_size

loss = tf.keras.losses.CategoricalCrossentropy()

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss=loss,
              metrics=['categorical_accuracy']
             )

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   patience=5,
                   restore_best_weights=True,
                   verbose=1
                  )

checkpoint_cb = ModelCheckpoint("best_model.h5",
                                save_best_only=True,
                                monitor='val_loss',
                                mode='min',
                                save_freq='epoch'
                               )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.3,
                              patience=3,
                              min_lr=1e-5,
                              mode='min',
                              verbose=1
                             )

history = model.fit(train_set,
                   validation_data=val_set,
                   epochs=EPOCHS,
                   batch_size=64,
                   steps_per_epoch=STEP_SIZE_TRAIN,
                   validation_steps=STEP_SIZE_VALID,
                   callbacks=[es, checkpoint_cb, reduce_lr]
                   )

model.save('final_model.h5')

with open('trainingHistory.json', 'w') as hist_file:
    json.dump(history.history, hist_file)