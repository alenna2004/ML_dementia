"""
This code is used for creating neuro net for detecting dementia using Tensorflow.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


#Here you have to place the path to folder with images
PATH = ''

#here we taking our dataset from directory
img_height, img_width = 224, 224
train_ds = tf.keras.utils.image_dataset_from_directory(
  '/content/drive/MyDrive/dementia/',
  validation_split=0.2,
  subset="training",
  seed=42,
  color_mode = 'grayscale',
  image_size=(img_height, img_width),
  batch_size=32)
val_ds = tf.keras.utils.image_dataset_from_directory(
  '/content/drive/MyDrive/dementia/',
  validation_split=0.2,
  subset="validation",
  seed=42,
  color_mode = 'grayscale',
  image_size=(img_height, img_width),
  batch_size=32)
class_names = train_ds.class_names
num_classes = len(class_names)

#creating cache
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


#defining a sequential model
model_seq = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 1)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(5)
])


#working with pretrained model resnet50
def process(image,label):
    image = tf.cast(image/255. ,tf.float32)
    return image,label
train_ds = train_ds.map(process)
val_ds = val_ds.map(process)
inputs = resnet.input
x = resnet.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.6)(x)
outputs = Dense(5, activation ='softmax')(x)
model_resnet = Model(inputs=inputs, outputs=outputs)


#working with pretrained model mobilenet
i = tf.keras.layers.Input([None, None, 3], dtype = tf.uint8)
x = tf.cast(i, tf.float32)
x = tf.keras.applications.mobilenet.preprocess_input(x)
core = tf.keras.applications.MobileNet()
x = core(x)
model_mobilenet = tf.keras.Model(inputs=[i], outputs=[x])


#working with pretrained model densenet
i = tf.keras.layers.Input([None, None, 3], dtype = tf.uint8)
x = tf.cast(i, tf.float32)
x = tf.keras.applications.densenet.preprocess_input(x)
core = tf.keras.applications.DenseNet201()
x = core(x)
model_densenet = tf.keras.Model(inputs=[i], outputs=[x])


#compiling a model and training it, code bellow is the same for all models, replace "model" with apropriate model name 
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001,beta_1=0.9, beta_2=0.999,epsilon=1e-07,),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


#visualazing our results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
loss, acc = model.evaluate(val_ds, verbose=0)
print('Accuracy: %.f' % acc)
print('Loss: %.3f' % loss)
