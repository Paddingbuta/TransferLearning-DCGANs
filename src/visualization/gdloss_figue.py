import matplotlib.pyplot as plt

loss = [0.8804, 0.4143, 0.3642, 0.3318, 0.3147, 0.2996, 0.8294, 0.1953, 0.1483, 0.1215, 0.1067, 0.0937, 0.0868, 0.0805, 0.0777, 0.0711, 0.0709, 0.0669, 0.0640, 0.0633]
accuracy = [0.7831, 0.8749, 0.8901, 0.8997, 0.9053, 0.9090, 0.8805, 0.9415, 0.9563, 0.9625, 0.9665, 0.9717, 0.9731, 0.9748, 0.9761, 0.9777, 0.9774, 0.9791, 0.9803, 0.9804]
val_loss = [0.3753, 0.3063, 0.2810, 0.2483, 0.2375, 0.2470, 0.2402, 0.1284, 0.1229, 0.0878, 0.0851, 0.0681, 0.0502, 0.0688, 0.0519, 0.0497, 0.0529, 0.0518, 0.0573, 0.0463]
val_accuracy = [0.9019, 0.9183, 0.9268, 0.9346, 0.9380, 0.9364, 0.9360, 0.9629, 0.9675, 0.9752, 0.9731, 0.9783, 0.9842, 0.9794, 0.9832, 0.9837, 0.9849, 0.9845, 0.9833, 0.9854]


x_values = list(range(1, 21))
plt.plot(x_values, loss, label='TRAIN_LOSS')
plt.plot(x_values, val_loss, label='TEST_LOSS')
plt.plot(x_values, accuracy, label='TRAIN_ACCURACY')
plt.plot(x_values, val_accuracy, label='TEST_ACCURACY')
ax = plt.gca()
y_ticks = ax.get_yticks()

# 在每个y轴刻度位置添加水平线
for y_tick in y_ticks:
    plt.axhline(y=y_tick, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.xlabel('Epochs')
plt.title('Accuracy Over Epochs')
plt.legend()
plt.show()
