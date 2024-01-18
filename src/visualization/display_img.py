from matplotlib import pyplot as plt
import os
import time
import PIL

g_loss = []
d_loss = []

def display_image(epoch):
  return PIL.Image.open('img_epoch{:04d}.png'.format(epoch))
display_image(5)
display_image(20)
display_image(40)
display_image(50)
plt.figure()
plt.plot(g_loss, label='generator')
plt.plot(d_loss, label = 'discriminator')
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.show()
