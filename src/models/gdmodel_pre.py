import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

def def_generator(noise_dim):
    out = tf.keras.Sequential()
    # best_dim = estimate_best_noise(min=64, max=512) 

    # 添加第一层，全连接层，将噪声映射到高维空间
    out.add(layers.Dense(7*7*256, use_bias=False, input_shape=(noise_dim,)))
    # out.add(layers.Dense(7*7*512, input_shape=(256,))) 
    out.add(layers.BatchNormalization())
    out.add(layers.LeakyReLU())

    # 将输出形状重新调整为3D张量，准备进行转置卷积操作
    out.add(layers.Reshape((7, 7, 256)))
    #print(out.output_shape)

    # 添加第一个转置卷积层，将图像上采样
    out.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert out.output_shape == (None, 7, 7, 128)
    out.add(layers.BatchNormalization())
    out.add(layers.LeakyReLU())

    out.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert out.output_shape == (None, 14, 14, 64)
    out.add(layers.BatchNormalization())
    out.add(layers.LeakyReLU())

    out.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert out.output_shape == (None, 28, 28, 1)

    return out

def def_discriminator():
    out = tf.keras.Sequential()

    # 添加第一个卷积层，用于处理图像的特征提取
    out.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    out.add(layers.LeakyReLU())
    out.add(layers.Dropout(0.3))

    out.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    out.add(layers.LeakyReLU())
    out.add(layers.Dropout(0.3))

    out.add(layers.Flatten())
    out.add(layers.Dense(1))

    return out