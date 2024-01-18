import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Sequential, layers
from IPython import display
from matplotlib import pyplot as plt
import os
import time
import PIL
import numpy as np
import gdmodel_pre as gdm

# ------------------ #
g_loss = []
d_loss = []
BATCH_SIZE = 256  # 每批次的样本数量，用于模型训练
EPOCHS = 60  # 训练的轮数，表示模型将遍历整个训练数据的次数
IMG_SCALE = 255.0 // 2  # 图像缩放因子，用于将像素值缩放到[-1, 1]范围
NOISE_DIM = 64  # 噪声向量的维度，用于生成器的输入
TRAIN_SIZE = 60000  # 训练集的大小，即MNIST数据集中的样本数量
BUFFER_SIZE = 60000
TEST_SIZE = 10000  # 测试集的大小，即MNIST数据集中的样本数量
LEARNING_RATE = 1e-4  # 模型的学习率，表示模型在每次参数更新时的步长

# ------------------ #

def discriminator_loss(out_real, out_fake, d_loss):
    real_loss = cross_entropy(tf.ones_like(out_real), out_real)
    fake_loss = cross_entropy(tf.zeros_like(out_fake), out_fake)
    total_loss = real_loss + fake_loss
    d_loss.append(total_loss)
    return total_loss

def generator_loss(out_fake, g_loss):
    fake_loss = cross_entropy(tf.ones_like(out_fake), out_fake)
    g_loss.append(fake_loss)
    return fake_loss

def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      out_real = discriminator(images, training=True)
      out_fake = discriminator(generated_images, training=True)

      gen_loss = generator_loss(out_fake, g_loss)
      disc_loss = discriminator_loss(out_real, out_fake, d_loss)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
  
def sav_img(model, epoch, test_input):
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(3, 3, i+1)
      plt.imshow(predictions[i, :, :, 0] * IMG_SCALE + IMG_SCALE, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - IMG_SCALE) / IMG_SCALE

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

generator = gdm.def_generator(NOISE_DIM)

noise = tf.random.normal([1, NOISE_DIM])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')

discriminator = gdm.def_discriminator()
decision = discriminator(generated_image)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

seed = tf.random.normal([9, NOISE_DIM])

for epoch in range(EPOCHS):
  start = time.time()

  for image_batch in train_dataset:
    train_step(image_batch)

  display.clear_output(wait=True)
  sav_img(generator, epoch + 1, seed)

  if (epoch + 1) % 20 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)

  print ('epoch {} 耗时 {} sec'.format(epoch + 1, time.time()-start))

display.clear_output(wait=True)
sav_img(generator, EPOCHS, seed)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

discriminator.summary()
