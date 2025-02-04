# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1VVZBKseDzCG-j3qB1T9VmqqrpTcUw8r1
"""

import tensorflow_datasets as tfds
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Processamento de dados
(train_set, valid_set, test_set), info = tfds.load('tf_flowers',     split=[
        'train[:70%]',  # 70% for training
        'train[70%:85%]',  # 15% for validation (from remaining train split)
        'train[85%:]'     # 15% for testing (from remaining train split)
    ], shuffle_files=True, with_info=True)

dataset_size = info.splits['train'].num_examples
class_names = info.features['label'].names
n_classes = info.features['label'].num_classes

print('train_set: ', len(train_set))
print('valid_set', len(valid_set))
print('test_set', len(test_set))
print(dataset_size)
print(class_names)
print(n_classes)

batch_size = 32
SIZE=224
def preprocess(image):
  rescalling = tf.keras.Sequential([
    tf.keras.layers.Resizing(SIZE, SIZE),
    # tf.keras.layers.Rescaling(1./255),
  ])
  image_rescalling = rescalling(image['image'])

  final_image = tf.keras.applications.xception.preprocess_input(image_rescalling)
  return final_image, image['label']

train_set = train_set.shuffle(1000)
train = train_set.map(preprocess).batch(batch_size).prefetch(1)
valid = valid_set.map(preprocess).batch(batch_size).prefetch(1)
test = test_set.map(preprocess).batch(batch_size).prefetch(1)


# Carregamento da arquitetura Xception
base_model = tf.keras.applications.xception.Xception(weights="imagenet", include_top=False)

avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

model.summary()


# Treinamento do modelo com as camadas da arquitetura Xception travadas
for layer in base_model.layers:
  layer.trainable = False

optimizer = tf.keras.optimizers.SGD(learning_rate=0.2, momentum=0.9)

model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

history = model.fit(train, epochs=5, validation_data=valid)

import matplotlib.pyplot as plt

loss = history.history['accuracy']
val_loss = history.history['val_accuracy']

epochs = range(1, len(loss)+ 1)

plt.plot(epochs, loss, 'y', label = "Treinamento")
plt.plot(epochs, val_loss, 'r', label = "Validação")

plt.title("Treinamento versus validação")
plt.xlabel("Epocas")
plt.ylabel("Acurácia Global")
plt.legend()
plt.show()

_,score = model.evaluate(test)

print(score)

# Treinamento do modelo utilizando as Camadas da arquitetura Xception

for layer in base_model.layers:
  layer.trainable = True


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

history = model.fit(train, epochs=5, validation_data=valid)

import matplotlib.pyplot as plt

loss = history.history['accuracy']
val_loss = history.history['val_accuracy']

epochs = range(1, len(loss)+ 1)

plt.plot(epochs, loss, 'y', label = "Treinamento")
plt.plot(epochs, val_loss, 'r', label = "Validação")

plt.title("Treinamento versus validação")
plt.xlabel("Epocas")
plt.ylabel("Acurácia Global")
plt.legend()
plt.show()

_,score = model.evaluate(test)

print(score)

# Salvando modelo 
model.save('flowers_model.h5')

# Carregamento modelo e predizendo imagens da internet
from tensorflow.keras.models import load_model

model = load_model('flowers_model.h5')

print(class_names)

from PIL import Image
import numpy as np
from skimage import transform
from io import BytesIO
import requests

def load(filename):
  response = requests.get(filename)
  np_image = Image.open(BytesIO(response.content))
  np_image = np.array(np_image).astype('float32')/255
  np_image = transform.resize(np_image, (256, 256, 3))
  np_image = np.expand_dims(np_image, axis=0)
  return np_image


image = load('https://www.selectseeds.com/cdn/shop/products/239-1-zoom_grande.jpg?v=1687464500')
result = model.predict(image)

predicted_class_index = np.argmax(result[0])

# Access the class name using the index
predicted_class_name = class_names[predicted_class_index]

print("Predicted class:", predicted_class_name)