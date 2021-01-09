import json
import os

import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.python.keras.applications.efficientnet import EfficientNetB5

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=tf_config)

arg = argparse.ArgumentParser()
arg.add_argument("-m", default="no", help="set yes to use mixed precision training")
args = vars(arg.parse_args())

if args['m'] == 'yes':
    # enable XLA
    tf.config.optimizer.set_jit(True)

    # enable AMP
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

df = pd.read_csv('train.csv')
df['path'] = '../train_images/' + df['image_id']
f = open('label_num_to_disease_map.json')
real_labels = json.load(f)
real_labels = {int(k): v for k, v in real_labels.items()}
df['class_name'] = df.label.map(real_labels)
train, valid = train_test_split(df, test_size=0.2, random_state=42, shuffle=True, stratify=df['class_name'])

N_CLASS = 5
IMAGE_SIZE = 256
BS = 32
VBS = 16
EPOCHS = 50
STEPS_PER_EPOCH = len(df) * 0.8 // BS
VALIDATION_STEPS = len(df) * 0.2 // VBS

AUTOTUNE = tf.data.experimental.AUTOTUNE


def process_data_train(image_path, label):
    # load the raw data from the file as a string
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.random_brightness(img, 0.3)
    img = tf.image.random_flip_left_right(img, seed=None)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_crop(img, size=[IMAGE_SIZE, IMAGE_SIZE, 3])
    return img, label


def process_data_valid(image_path, label):
    # load the raw data from the file as a string
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    return img, label


train_ds = tf.data.Dataset.from_tensor_slices((train.path.values, train.label.values))
valid_ds = tf.data.Dataset.from_tensor_slices((valid.path.values, valid.label.values))

train_ds = train_ds.shuffle(len(train.path.values))
valid_ds = valid_ds.shuffle(len(valid.path.values))

train_ds = train_ds.map(process_data_train, num_parallel_calls=AUTOTUNE)
valid_ds = valid_ds.map(process_data_valid, num_parallel_calls=AUTOTUNE)


def configure_for_performance(ds, batch_size):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


train_ds_batch = configure_for_performance(train_ds, BS)
valid_ds_batch = configure_for_performance(train_ds, VBS)

model = Sequential()
model.add(EfficientNetB5(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(GlobalAveragePooling2D())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(N_CLASS, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['categorical_accuracy']
              )

rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, verbose=1, mode='min', min_lr=1e-4)
mc = ModelCheckpoint('./models/effB5/best.h5', verbose=1, save_best_only=True, monitor='val_loss')
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)

model.fit(train_ds_batch,
          validation_data=valid_ds_batch,
          epochs=EPOCHS,
          steps_per_epoch=STEPS_PER_EPOCH,
          validation_steps=VALIDATION_STEPS,
          callbacks=[rlr, mc, es],
          workers=12,
          use_multiprocessing=True
          )

model.save('./models/effB5/weights.h5')
