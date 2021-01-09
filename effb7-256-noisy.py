# benchmark with simple baseline model

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
import efficientnet.keras as eff
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.python.keras import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=tf_config)

df = pd.read_csv('train.csv')
f = open('label_num_to_disease_map.json')
real_labels = json.load(f)
real_labels = {int(k): v for k, v in real_labels.items()}
df['class_name'] = df.label.map(real_labels)
train, valid = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class_name'])

size = 600
n_class = 5
EPOCHS = 50
BS = 16
VBS = 16
STEPS_PER_EPOCH = len(df)*0.8 // BS
VALIDATION_STEPS = len(df)*0.2 // VBS

prep_func = tf.keras.applications.efficientnet.preprocess_input

train_datagen = ImageDataGenerator(
    preprocessing_function=prep_func,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(
    preprocessing_function=prep_func,
)

train_datagenerator = train_datagen.flow_from_dataframe(
    train,
    directory='../train_images/',
    seed=42,
    x_col='image_id',
    y_col='class_name',
    target_size=(size, size),
    class_mode='categorical',
    shuffle=True,
    batch_size=BS,
)

valid_datagenerator = valid_datagen.flow_from_dataframe(
    valid,
    directory='../train_images/',
    seed=42,
    x_col='image_id',
    y_col='class_name',
    target_size=(size, size),
    class_mode='categorical',
    shuffle=False,
    batch_size=VBS,
)

model = Sequential()
model.add(EfficientNetB4(weights='models/efficientnet-b7_noisy-student_notop.h5', input_shape=(size, size, 3)))
model.add(GlobalAveragePooling2D())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(n_class, activation='softmax'))

model.summary()

# ---

# model = Sequential()
# for layer in base_model.layers:
#     layer.trainable = False
#     model.add(layer)
# model.add(GlobalAveragePooling2D())
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dense(n_class, activation='softmax'))

# model = Sequential([
#     base_model,
#     GlobalAveragePooling2D(),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dense(n_class, activation='softmax')
# ])
#model = EfficientNetB4(weights='', include_top=False, input_shape=(size, size, 3), classifier_activation='softmax')
# model = Sequential()
# model.add(EfficientNetB4(include_top=False, weights='models/efficientnet-b4_noisy-student_notop.h5', input_shape=(size, size, 3)))


model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['categorical_accuracy']
              )

rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=1, mode='min', min_lr=1e-4)
mc = ModelCheckpoint('./models/effB7/best.h5', verbose=1, save_best_only=True, monitor='val_loss')
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)

model.fit(train_datagenerator,
          validation_data=valid_datagenerator,
          epochs=EPOCHS,
          steps_per_epoch=STEPS_PER_EPOCH,
          validation_steps=VALIDATION_STEPS,
          callbacks=[rlr, mc, es],
          workers=12,
          use_multiprocessing=True
          )

model.save('./models/effB7/weights.h5')
