# benchmark with simple baseline model

import json
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=tf_config)

df = pd.read_csv('train.csv')
f = open('label_num_to_disease_map.json')
real_labels = json.load(f)
real_labels = {int(k):v for k,v in real_labels.items()}
df['class_name'] = df.label.map(real_labels)
train, valid = train_test_split(df, test_size=0.1, random_state=42, stratify=df['class_name'])

size = 256
n_class = 5
EPOCHS = 50
BS = 256

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(
    rescale=1./255,
)

train_datagenerator = train_datagen.flow_from_dataframe(
    train,
    directory='train256/',
    seed = 42,
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
model.add(Conv2D(64, (3, 3), input_shape=(size, size, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(n_class))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['categorical_accuracy']
)

rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, verbose=1, mode='min')
mc = ModelCheckpoint('./models/latest.h5', verbose=1, save_best_only=True, monitor='val_loss')
es = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)
model.fit(train_datagenerator,
          validation_data=valid_datagenerator,
          epochs=EPOCHS,
          callbacks=[rlr, mc, es],
          workers=12,
          use_multiprocessing=True
          )

model.save('./models/baseline.h5')