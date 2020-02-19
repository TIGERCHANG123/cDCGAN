import tensorflow as tf
import numpy as np

def discriminator_loss(real_real_output, real_fake_output, fake_real_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = cross_entropy(tf.ones_like(real_real_output), real_real_output)
    fake_loss1 = cross_entropy(tf.zeros_like(real_fake_output), real_fake_output)
    fake_loss2 = cross_entropy(tf.zeros_like(fake_real_output), fake_real_output)
    total_loss = real_loss + fake_loss1 + fake_loss2
    return total_loss

def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    return cross_entropy(tf.ones_like(fake_output), fake_output)

class train_one_epoch():
    def __init__(self, model, train_dataset, optimizers, metrics, noise_gen):
        self.generator, self.discriminator = model
        self.generator_optimizer, self.discriminator_optimizer = optimizers
        self.gen_loss, self.disc_loss = metrics
        self.train_dataset = train_dataset
        self.noise_gen = noise_gen

    def fit_label(self, image, label):
        label = tf.expand_dims(label, axis=1)
        label = tf.expand_dims(label, axis=1)
        shape = image.shape
        label = label * tf.ones([shape[0], shape[1], shape[2], 10])
        concat = tf.concat([image, label], axis=-1)
        return concat
    def train_step(self, noise, labels, images):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            real_label = tf.one_hot(labels, depth=10)
            gen_input = tf.concat([noise, real_label], axis=-1)
            fake_images = self.generator(gen_input, training=True)

            fake_label = np.random.multinomial(1, 10 * [float(1.0 / 10)], size=[images.shape[0]])
            fake_label = tf.convert_to_tensor(fake_label)
            fake_label = tf.cast(fake_label, tf.float32)

            real_real_label = self.fit_label(images,  real_label)
            real_fake_label = self.fit_label(images, fake_label)
            fake_real_label = self.fit_label(fake_images,  real_label)

            disc_real_real = self.discriminator(real_real_label, training=True)
            disc_real_fake = self.discriminator(real_fake_label, training=True)
            disc_fake_real = self.discriminator(fake_real_label, training=True)

            gen_loss = generator_loss(disc_fake_real)
            disc_loss = discriminator_loss(disc_real_real, disc_real_fake, disc_fake_real)
        self.gen_loss(gen_loss)
        self.disc_loss(disc_loss)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
    def train(self, epoch,  pic):
        self.gen_loss.reset_states()
        self.disc_loss.reset_states()

        for (batch, (images, labels)) in enumerate(self.train_dataset):
            noise, auxi_dict = self.noise_gen.get_noise()
            self.train_step(noise, labels, images)
            pic.add([self.gen_loss.result().numpy(), self.disc_loss.result().numpy()])
            pic.save()
            if batch % 500 == 0:
                print('epoch: {}, gen loss: {}, disc loss: {}'.format(epoch, self.gen_loss.result(), self.disc_loss.result()))