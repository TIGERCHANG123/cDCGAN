import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class generator(tf.keras.Model):
  def __init__(self, noise_shape, img_shape):
    super(generator, self).__init__()
    self.img_shape = img_shape
    self.noise_shape = noise_shape

    self.model = tf.keras.Sequential()

    self.model.add(tf.keras.layers.Dense(256, input_shape=noise_shape))
    self.model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    self.model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    self.model.add(tf.keras.layers.Dense(512))
    self.model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    self.model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    self.model.add(tf.keras.layers.Dense(1024))
    self.model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    self.model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    self.model.add(tf.keras.layers.Dense(np.prod(self.img_shape), activation='tanh'))
    self.model.add(tf.keras.layers.Reshape(self.img_shape))
  def call(self, x):
    x=self.model(x)
    return x

class discriminator(tf.keras.Model):
  def __init__(self, img_shape):
    super(discriminator, self).__init__()
    self.img_shape = img_shape
    self.model = tf.keras.Sequential()

    self.model.add(tf.keras.layers.Flatten(input_shape=self.img_shape))
    self.model.add(tf.keras.layers.Dense(512))
    self.model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    self.model.add(tf.keras.layers.Dense(256))
    self.model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
  def call(self, x):
    return self.model(x)

def get_gan(noise_shape, img_shape):
  Generator = generator(noise_shape, img_shape)
  Discriminator = discriminator(img_shape)
  gen_name = 'gan'
  return Generator, Discriminator, gen_name

