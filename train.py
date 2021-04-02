"""This module implements data feeding and training loop to create model
to classify X-Ray chest images as a lab example for BSU students.
"""

__author__ = 'Alexander Soroka, soroka.a.m@gmail.com'
__copyright__ = """Copyright 2020 Alexander Soroka"""


import argparse
import glob
import numpy as np
import tensorflow as tf
import time
from tensorflow.python import keras as keras
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
import tensorflow.keras.applications
from tensorflow.keras.models import Sequential
import math

# Avoid greedy memory allocation to allow shared GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


LOG_DIR = 'logs'
BATCH_SIZE = 64
NUM_CLASSES = 20
RESIZE_TO = 224
TRAIN_SIZE = 12786

# img_augmentation = keras.Sequential(
#     [
#         preprocessing.RandomRotation(factor=0.15),
#         preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
#         preprocessing.RandomFlip(),
#         preprocessing.RandomContrast(factor=0.1),
#     ]
# )

# def augment(image, label):
#   bright = tf.image.adjust_brightness(image, delta=0.1)
#   contrast = tf.image.adjust_contrast(image, 2)
#   return image, label


# resize_and_rescale = tf.keras.Sequential([
#   layers.experimental.preprocessing.Resizing(RESIZE_TO, RESIZE_TO),
#   layers.experimental.preprocessing.Rescaling(1./255)
# ])

# def augment(image_label, seed):
#   image, label = image_label
#   #image, label = resize_and_rescale(image, label)
# #   tf.image.stateless_random_brightness(image, max_delta, seed)
# #   tf.image.stateless_random_contrast(image, lower, upper, seed)
#   image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
#   # Make a new seed
#   new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
#   # Random crop back to the original size
# #   image = tf.image.stateless_random_crop(
# #       image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
#   # Random brightness
#   image = tf.image.stateless_random_brightness(
#       image, max_delta=0.5, seed=new_seed)
# #   image = tf.clip_by_value(image, 0, 1)
#   return image, label

# counter = tf.data.experimental.Counter()
# train_ds = tf.data.Dataset.zip((train_datasets, (counter, counter)))

def parse_proto_example(proto):
  keys_to_features = {
    'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
    'image/label': tf.io.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
  }
  example = tf.io.parse_single_example(proto, keys_to_features)
  example['image'] = tf.image.decode_jpeg(example['image/encoded'], channels=3)
  example['image'] = tf.image.convert_image_dtype(example['image'], dtype=tf.uint8)
  example['image'] = tf.image.resize(example['image'], tf.constant([RESIZE_TO, RESIZE_TO]))
  return example['image'], tf.one_hot(example['image/label'], depth=NUM_CLASSES)


def normalize(image, label):
  return tf.image.per_image_standardization(image), label


def create_dataset(filenames, batch_size):
  """Create dataset from tfrecords file
  :tfrecords_files: Mask to collect tfrecords file of dataset
  :returns: tf.data.Dataset
  """
  #.map(augment, parse_proto_example, num_parallel_calls=tf.data.AUTOTUNE)
  return tf.data.TFRecordDataset(filenames)\
    .map(parse_proto_example, num_parallel_calls=tf.data.AUTOTUNE)\
    .cache()\
    .batch(batch_size)\
    .prefetch(tf.data.AUTOTUNE)


def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  #x = img_augmentation(inputs)
  #x = tf.keras.preprocessing.image.random_brightness(x, 3.0)
  x = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")
  x.trainable = False
  x = layers.GlobalAveragePooling2D()(x.output)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)

# def exp_decay(epoch):
#    initial_lrate = 0.01
#    k = 0.6
#    lrate = initial_lrate * math.exp(-k*t)
#    return lrate

def step_decay(epoch):
   initial_lrate = 0.1
   drop = 0.6
   epochs_drop = 5.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate
  
lrate = LearningRateScheduler(step_decay)

def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True


def main():
  args = argparse.ArgumentParser()
  args.add_argument('--train', type=str, help='Glob pattern to collect train tfrecord files, use single quote to escape *')
  args = args.parse_args()

  dataset = create_dataset(glob.glob(args.train), BATCH_SIZE).shuffle(8)
  train_size = int(TRAIN_SIZE * 0.7 / BATCH_SIZE)
  train_dataset = dataset.take(train_size)
  validation_dataset = dataset.skip(train_size)

  model = build_model()

  model.compile(
    optimizer=tf.optimizers.Adam(),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.categorical_accuracy],
  )

  log_dir='{}/owl-{}'.format(LOG_DIR, time.time())
  model.fit(
    train_dataset,
    epochs=20,
    validation_data=validation_dataset,
    callbacks=[
      tf.keras.callbacks.TensorBoard(log_dir), lrate
    ]
  )
  
  unfreeze_model(model)
  
  model.compile(
    optimizer=tf.optimizers.Adam(),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.categorical_accuracy],
  )
  
  model.fit(
    train_dataset,
    epochs=10,
    validation_data=validation_dataset,
    callbacks=[
      tf.keras.callbacks.TensorBoard(log_dir), lrate
    ]
  )
    


if __name__ == '__main__':
    main()
