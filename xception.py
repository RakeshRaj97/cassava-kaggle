# benchmark with simple baseline model

import json
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import Xception

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=tf_config)

df = pd.read_csv('train.csv')
f = open('label_num_to_disease_map.json')
real_labels = json.load(f)
real_labels = {int(k): v for k, v in real_labels.items()}
df['class_name'] = df.label.map(real_labels)
train, valid = train_test_split(df, test_size=0.1, random_state=42, stratify=df['class_name'])

size = 256
n_class = 5
EPOCHS = 50
BS = 96

prep_func = tf.keras.applications.xception.preprocess_input

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
    directory='train256/',
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
    directory='train256/',
    seed=42,
    x_col='image_id',
    y_col='class_name',
    target_size=(size, size),
    class_mode='categorical',
    shuffle=False,
    batch_size=32,
)

model = Sequential()
model.add(Xception(include_top=False, weights='imagenet', input_shape=(256, 256, 3), classifier_activation='softmax'))
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(512, activation='relu', bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)))
model.add(Dropout(0.5))
model.add(Dense(n_class, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['categorical_accuracy']
              )

rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, verbose=1, mode='min')
mc = ModelCheckpoint('./models/xception-256/latest.h5', verbose=1, save_best_only=True, monitor='val_loss')
es = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)
model.fit(train_datagenerator,
          validation_data=valid_datagenerator,
          epochs=EPOCHS,
          callbacks=[rlr, mc, es],
          workers=12,
          use_multiprocessing=True
          )

model.save('./models/xception-256/baseline.h5')
