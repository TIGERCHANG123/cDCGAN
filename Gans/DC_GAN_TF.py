import tensorflow as tf
from tensorflow.keras import layers

class generator(tf.keras.Model):
  def __init__(self, noise_shape, img_shape):
    super(generator, self).__init__()
    self.noise_shape = noise_shape
    self.img_shape=img_shape

    self.model = tf.keras.Sequential()
    self.model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=noise_shape))
    self.model.add(layers.BatchNormalization())
    self.model.add(layers.LeakyReLU())
    self.model.add(layers.Reshape((7, 7, 256)))
    self.model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    self.model.add(layers.BatchNormalization())
    self.model.add(layers.LeakyReLU())
    self.model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    self.model.add(layers.BatchNormalization())
    self.model.add(layers.LeakyReLU())
    self.model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
  def call(self, x, output):
    return self.model(x)

class discriminator(tf.keras.Model):
  def __init__(self, img_shape):
    super(discriminator, self).__init__()
    self.img_shape = img_shape

    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=img_shape))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    self.model = model
  def call(self, x):
    return self.model(x)

def get_gan(noise_shape, img_shape):
  Generator = generator(noise_shape, img_shape)
  Discriminator = discriminator(img_shape)
  gen_name = 'dc_gan'
  return Generator, Discriminator, gen_name
