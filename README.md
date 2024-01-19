# CNN-GAN-DCGAN-TransferLearning-Enhancement

* This project enhances GAN and DCGAN models for image generation and transformation. Improvements include optimizing parameters, introducing novel algorithms, and achieving accuracy levels comparable to standard CNN models.

## Key Contributions
* **Improvement of Original GAN Network:** Considering factors such as learning rate, activation function, optimizer, and network structure. The performance in terms of accuracy and D_loss is significantly better than the original algorithm.

* **Enhanced DCGAN Algorithm:** Introducing an enhanced DCGAN algorithm. This improvement incorporates Wasserstein distance, spectral normalization, and noise input dimension estimation, resulting in significantly better performance compared to a standard GAN network.

* **Transformation into a 10-Class Classifier:** The DCGAN model is further enhanced by transforming its discriminator into a 10-class classifier. Through transfer learning and fine-tuning strategies, the model's performance is optimized, achieving accuracy levels very close to the original CNN model.

## Development Environment

* Windows 10
* Python 3.11.5
* tensorflow-gpu 2.15.0
* numpy 1.24.3
* matplotlib 3.7.2

## Dataset Download
### MNIST Dataset:
- Official Website: [MNIST](http://yann.lecun.com/exdb/mnist/)
- TensorFlow provides a convenient API to directly load the MNIST dataset. You can use the `tfds` module from TensorFlow Datasets: [MNIST on TensorFlow](https://www.tensorflow.org/datasets/catalog/mnist)

### CIFAR-10 Dataset:
- Official Website: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- TensorFlow also offers the `tfds` API to load the CIFAR-10 dataset: [CIFAR-10 on TensorFlow](https://www.tensorflow.org/datasets/catalog/cifar10)
## Main References

[1] Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.

(Full paper: http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

[2] GAN, Lan, et al. "Data augmentation method based on improved deep convolutional generative adversarial networks." Journal of Computer Applications 41.5 (2021): 1305.

[3] Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434

(Full paper: https://arxiv.org/pdf/1511.06434.pdf)

