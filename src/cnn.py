import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from matplotlib import pyplot as plt
import numpy as np

# ------------------ #
#g_loss = []
#d_loss = []
BATCH_SIZE = 256  # 每批次的样本数量，用于模型训练
EPOCHS = 15  # 训练的轮数，表示模型将遍历整个训练数据的次数
IMG_SCALE = 255.0 // 2  # 图像缩放因子，用于将像素值缩放到[-1, 1]范围
NOISE_DIM = 64  # 噪声向量的维度，用于生成器的输入
TRAIN_SIZE = 60000  # 训练集的大小，即MNIST数据集中的样本数量
BUFFER_SIZE = TRAIN_SIZE
TEST_SIZE = 10000  # 测试集的大小，即MNIST数据集中的样本数量
LEARNING_RATE = 1e-4  # 模型的学习率，表示模型在每次参数更新时的步长

# ------------------ #

(train_img, train_lb), (test_img, test_lb) = tf.keras.datasets.mnist.load_data()
print(train_img.shape, train_lb.shape, test_img.shape, test_lb.shape)

'''
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
'''

out = Sequential()
out.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
out.add(layers.MaxPooling2D(2, 2))
out.add(layers.Conv2D(64, (3, 3), activation='relu'))
out.add(layers.MaxPooling2D(2, 2))
out.add(layers.Flatten())
out.add(layers.Dense(128))
out.add(layers.Dense(10, activation='softmax'))
out.summary()

train_labels = to_categorical(train_lb)
test_labels = to_categorical(test_lb)

train_images = train_img.reshape((TRAIN_SIZE, 28, 28, 1))
train_images = (train_images.astype('float32') - IMG_SCALE) / IMG_SCALE

test_images = test_img.reshape((TEST_SIZE, 28, 28, 1))
test_images = (test_images.astype('float32') - IMG_SCALE) / IMG_SCALE

out.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = out.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_data=(test_images, test_labels))


plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.3, 1])
plt.legend(loc='lower right')
plt.show()
# print('Test accuracy:', test_acc)

# demonstration: predict the ith test digit
i = 15
prediction = out.predict(test_images[i].reshape(1,28,28,1))

# get probability distribution and classification of the test digit
print(prediction)
print('prediction:', np.argmax(prediction))

# draw the barplot
plt.figure()
plt.bar(np.arange(0,10).astype('str'), prediction[0,:])
plt.show()

# show the actual ith digit
print('actual label:', np.argmax(test_labels[i]))
plt.figure()
plt.imshow(test_images[i,:,:,0], cmap='gray')
plt.show()

pred_labels = out.predict(test_images).argmax(axis=1)
true_labels = test_labels.argmax(axis=1)

confusion_mat = tf.math.confusion_matrix(labels=true_labels, predictions=pred_labels).numpy()
confusion_mat

import pandas as pd
import seaborn as sns

# normalize
confusion_mat_norm = np.around(confusion_mat.astype('float') / np.atleast_2d(confusion_mat.sum(axis=1)).T, decimals=2)

classes = np.arange(0,10).astype('str')
confusion_mat_df = pd.DataFrame(confusion_mat_norm,
                                index = classes, 
                                columns = classes)

# generate heatmap
figure = plt.figure()
sns.heatmap(confusion_mat_df, annot=True, cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()