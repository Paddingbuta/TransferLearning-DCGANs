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

## Results
### Improvement of Original GAN Network
<table>
  <tr>
    <th colspan="2">Accuracy(%)</th>
    <th colspan="2">Discriminator Loss</th>
    <th colspan="2">Generator Loss</th>
  </tr>
  <tr>
    <td>pre</td>
    <td>new</td>
    <td>pre</td>
    <td>new</td>
    <td>pre</td>
    <td>new</td>
  </tr>
  <tr>
    <td>59.8</td><td>53.7</td><td>0.69</td><td>1.29</td><td>0.53</td><td>1.42</td>
  </tr>
  <tr>
    <td>60.3</td><td>59.8</td><td>0.62</td><td>0.18</td><td>0.81</td><td>0.86</td>
  </tr>
  <tr>
    <td>61.5</td><td>62.4</td><td>0.64</td><td>0.25</td><td>0.93</td><td>1.13</td>
  </tr>
  <tr>
    <td>62.7</td><td>63.4</td><td>0.59</td><td>0.29</td><td>0.98</td><td>0.96</td>
  </tr>
  <tr>
    <td>63.5</td><td>65.7</td><td>0.60</td><td>0.31</td><td>0.96</td><td>0.84</td>
  </tr>
  <tr>
    <td>61.9</td><td>68.2</td><td>0.57</td><td>0.24</td><td>1.02</td><td>0.95</td>
  </tr>
  <tr>
    <td>65.7</td><td>66.7</td><td>0.61</td><td>0.32</td><td>1.06</td><td>0.91</td>
  </tr>
  <tr>
    <td>62.8</td><td>68.1</td><td>0.51</td><td>0.29</td><td>1.03</td><td>0.86</td>
  </tr>
  <tr>
    <td>63.9</td><td>69.0</td><td>0.49</td><td>0.28</td><td>1.04</td><td>0.95</td>
  </tr>
  <tr>
    <td>66.5</td><td>69.3</td><td>0.46</td><td>0.34</td><td>1.05</td><td>0.89</td>
  </tr>
  <tr>
    <td>66.3</td><td>70.2</td><td>0.50</td><td>0.27</td><td>0.97</td><td>0.90</td>
  </tr>
  <!-- ... (remaining rows) ... -->
</table>

### Enhanced DCGAN Algorithm

<table align='center'>
<tr align='center'>
<td>Generated Digits in GAN</td>
<td>Generated Digits in Improved DCGAN</td>
</tr>
<tr>
<td><img src = 'img/Generated Digits Evolution in GAN, MNIST dataset.png'>
<td><img src = 'img/Generated Digits Evolution in DCGAN, MNIST dataset.png'>
</tr>
</table>

### Transfer Learning and Fine-tune

<table align='center'>
<tr align='center'>
<td>Training & Validation Accuracy & Loss in 3.3.3</td>
</tr>
<tr>
<td><img src = 'img/Training & Validation Accuracy & Loss in 3.3.3.png'  width='400px'>
</tr>
</table>

## Main References

[1] Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.

(Full paper: http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

[2] GAN, Lan, et al. "Data augmentation method based on improved deep convolutional generative adversarial networks." Journal of Computer Applications 41.5 (2021): 1305.

[3] Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434

(Full paper: https://arxiv.org/pdf/1511.06434.pdf)

