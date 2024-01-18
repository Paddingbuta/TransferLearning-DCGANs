import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

def def_generator(noise_dim):
    out = tf.keras.Sequential()
    # best_dim = estimate_best_noise(min=64, max=512) 

    # 添加第一层，全连接层，将噪声映射到高维空间
    out.add(Dense(7*7*512, input_shape=(noise_dim,)))
    # out.add(layers.Dense(7*7*512, input_shape=(256,)))  
    out.add(layers.LayerNormalization()) # 添加层标准化层
    #out.add(layers.BatchNormalization())
    out.add(layers.ReLU())  # 添加ReLU激活函数

    # 将输出形状重新调整为3D张量，准备进行转置卷积操作
    out.add(layers.Reshape((7, 7, 512)))
    print(out.output_shape)
    
    # 添加第一个转置卷积层，将图像上采样
    out.add(layers.Conv2DTranspose(256, (5,5), strides=(1,1), padding='same'))
    print(out.output_shape)
    out.add(layers.LayerNormalization())
    #out.add(layers.BatchNormalization())
    out.add(layers.ReLU())
    print(out.output_shape)
    out.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same'))
    out.add(layers.LayerNormalization())
    #out.add(layers.BatchNormalization())
    out.add(layers.ReLU())
    out.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'))
    out.add(layers.LayerNormalization())
    #out.add(layers.BatchNormalization())
    out.add(layers.ReLU())

    # 添加最终转置卷积层，输出生成的图像
    out.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh'))
    out.add(layers.LayerNormalization())
    print(out.output_shape)
    return out

def def_discriminator():
    # 创建一个Sequential模型，用于构建鉴别器
    out = tf.keras.Sequential()
    
    # 添加第一个卷积层，用于处理图像的特征提取
    out.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same', input_shape=[28, 28, 1])) 
    out.add(layers.LayerNormalization())
    out.add(layers.LeakyReLU())
    out.add(layers.Dropout(0.3))

    out.add(layers.Conv2D(256, (5,5), strides=(2,2), padding='same'))
    out.add(layers.LayerNormalization())
    out.add(layers.LeakyReLU())
    out.add(layers.Dropout(0.3))
    
    # 添加第三个卷积层
    out.add(layers.Conv2D(512, kernel_size=4, strides=1, padding='valid'))
    out.add(layers.LayerNormalization())
    out.add(layers.Flatten())
    out.add(layers.Dense(1))
    print(out.output_shape)
    return out