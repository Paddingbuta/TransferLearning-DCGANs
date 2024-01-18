import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential, layers
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os
import time
import numpy as np
import gdmodel as gdm

# ------------------ #
BATCH_SIZE = 256  # 每批次的样本数量，用于模型训练
EPOCHS1, EPOCHS2 = 10, 10  # 训练的轮数，表示模型将遍历整个训练数据的次数
IMG_SCALE = 255.0 // 2  # 图像缩放因子，用于将像素值缩放到[-1, 1]范围
NOISE_DIM = 64  # 噪声向量的维度，用于生成器的输入
TRAIN_SIZE = 60000  # 训练集的大小，即MNIST数据集中的样本数量
TEST_SIZE = 10000  # 测试集的大小，即MNIST数据集中的样本数量
LEARNING_RATE = 1e-4  # 模型的学习率，表示模型在每次参数更新时的步长

# ------------------ #

'''
def discriminator_loss(real_output, fake_output):
    real_loss = tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output) 
    loss = -real_loss + fake_loss
    return loss

def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output, g_loss)
      disc_loss = discriminator_loss(real_output, fake_output, d_loss)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_opt.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_opt.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
'''

checkpoint_dir = './training_checkpoints'  # 检查点保存的目录路径
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")  # 检查点文件的前缀，用于保存模型参数

generator = gdm.def_generator(NOISE_DIM)  # 生成器模型
generator_opt = tf.keras.optimizers.Adam(LEARNING_RATE)  # 生成器模型优化器

discriminator = gdm.def_discriminator()  # 判别器模型
discriminator_opt = tf.keras.optimizers.Adam(LEARNING_RATE)  # 判别器模型优化器

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_opt,
                                 discriminator_optimizer=discriminator_opt,
                                 generator=generator,
                                 discriminator=discriminator)  # 创建检查点对象，用于保存和恢复模型参数

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))  # 从最新的检查点文件中恢复模型参数

checkpoint.discriminator.summary()  # 打印判别器模型的摘要信息

# 生成一个随机噪声向量，通过生成器生成样本，然后通过判别器进行判别
val = checkpoint.discriminator(checkpoint.generator(tf.random.normal([1, NOISE_DIM]), training=False))
print(val)  # 打印判别器对生成的样本的输出值

model = Sequential()  # 创建一个新的顺序模型

# 将判别器模型的前几层添加到新模型中
for i in range(len(checkpoint.discriminator.layers) - 1):
    model.add(checkpoint.discriminator.layers[i])

# 冻结所有层，使它们在训练中不可更新权重
for layer in model.layers:
    layer.trainable = False

model.add(Dense(128))
# 添加一个新的全连接层，作为多类别分类器输出10个类别
model.add(Dense(10, activation='softmax'))

model.summary()  # 打印新模型的摘要信息

# 查看最后一层全连接层的权重和偏置，这些参数在训练过程中会被更新
model.layers[-2].weights

# 加载 MNIST
(train_img, train_lb), (test_img, test_lb) = tf.keras.datasets.mnist.load_data()

# 将加载的数据作为训练和测试数据
X_train, y_train = train_img, train_lb
X_test, y_test = test_img, test_lb


'''
# set random seed
np.random.seed(100)

randtrain = np.random.choice(train_img.shape[0], TRAIN_SIZE, replace=False)
X_train, y_train = train_img[randtrain], train_lb[randtrain]

np.random.seed(200)

randtest = np.random.choice(test_img.shape[0], TEST_SIZE, replace=False)
X_test, y_test = test_img[randtest], test_lb[randtest]
'''
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# 将训练数据 reshape 成适用于模型的形状，灰度图像通道为1
train_images = X_train.reshape((TRAIN_SIZE, 28, 28, 1))
# 将像素值缩放到[-1, 1]的范围内
train_images = (train_images.astype('float32') - IMG_SCALE) / IMG_SCALE

# 将测试数据 reshape 成适用于模型的形状，灰度图像通道为1
test_images = X_test.reshape((TEST_SIZE, 28, 28, 1))
# 将像素值缩放到[-1, 1]的范围内
test_images = (test_images.astype('float32') - IMG_SCALE) / IMG_SCALE

# 对训练和测试标签进行 one-hot 编码
train_labels = to_categorical(y_train)
test_labels = to_categorical(y_test)

# 编译模型，使用随机梯度下降作为优化器，交叉熵作为损失函数
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型，使用批量梯度下降，迭代 EPOCHS 次，同时在验证集上进行验证
history1 = model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS1,
                    validation_data=(test_images, test_labels))

# 解冻所有层
for layer in model.layers:
    layer.trainable = True
    
# 重新编译模型
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 使用解冻后的模型进行训练
history2 = model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS2,
                      validation_data=(test_images, test_labels))

train_accuracy = history1.history['accuracy'] + history2.history['accuracy']
test_accuracy = history1.history['val_accuracy'] + history2.history['val_accuracy']

plt.figure()
plt.plot(train_accuracy, label = 'train_accuracy')
plt.plot(test_accuracy, label = 'test_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy(%)')
plt.show()

# 预测第 i 个测试数字
i = 100
prediction = model.predict(test_images[i].reshape(1, 28, 28, 1))

# 获取预测概率分布和分类结果
print(prediction)
print('预测结果:', np.argmax(prediction))

# 绘制条形图展示预测概率分布
plt.figure()
plt.bar(np.arange(0, 10).astype('str'), prediction[0, :])
plt.show()

# 展示第 i 个数字的实际标签
print('实际标签:', np.argmax(test_labels[i]))
plt.figure()
plt.imshow(test_images[i, :, :, 0], cmap='gray')
plt.show()

'''
# 通过模型预测整个测试集，并计算混淆矩阵
pred_labels = model.predict(test_images).argmax(axis=1)
true_labels = test_labels.argmax(axis=1)

confusion_mat = tf.math.confusion_matrix(labels=true_labels, predictions=pred_labels).numpy()
print(confusion_mat)

confusion_mat_norm = np.around(confusion_mat.astype('float') / np.atleast_2d(confusion_mat.sum(axis=1)).T, decimals=2)

classes = np.arange(0,10).astype('str')
confusion_mat_df = pd.DataFrame(confusion_mat_norm,
                                index = classes, 
                                columns = classes)

figure = plt.figure()
sns.heatmap(confusion_mat_df, annot=True, cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
'''

