# Interesting-Works
<!-- https://latex.codecogs.com/gif.latex? -->

## SGD
- [On the Origin of Implicit Regularization in Stochastic Gradient Descent](https://arxiv.org/pdf/2101.12176.pdf), 2021 ICLR
- [On the Validity of Modeling SGD with Stochastic Differential Equations (SDEs)](https://arxiv.org/pdf/2102.12470.pdf), 2021 NeurIPS
- [Rethinking the Limiting Dynamics of SGD: Modified Loss, Phase Space Oscillations, and Anomalous Diffusion](https://arxiv.org/pdf/2107.09133.pdf), 2021 NeurIPS
  1) Other works have questionedthe correctness of the using the central limit theorem to model the gradient noise as Gaussian [20], arguing that the weak dependence between batches and heavy-tailed structure in the gradient noiseleads the CLT to break down.
  2) The two conditions needed for the CLT (Central Limit Theorem) to hold are not exactly met in the setting of SGD. Independent and identically distributed. Generally we perform SGD by making a complete pass through the entire dataset before using a sample again which introduces a weak dependence between samples. **While the covariance matrix without replacement more accurately models the dependence between samples within a batch, it fails to account for the dependence between batches.** *Finite variance*. A differentline of work has questioned the Gaussian assumption entirely because of the need for finite variance random variables. This work instead suggests using the generalized central limit theorem implying the noise would be a heavy-tailed α-stable random variable. Thus, the previous assumption is implicitly assuming the i.i.d. and finite variance conditions apply for large enough datasets and small enough batches.
  3) Reference: [What is the difference between finite and infinite variance](https://stats.stackexchange.com/questions/94402/what-is-the-difference-between-finite-and-infinite-variance/100161)

- [On the Interplay between Noise and Curvature and its Effecton Optimization and Generalization](https://arxiv.org/pdf/1906.07774.pdf), 2020 AISTATS
  1) The relationship between the Hessian and the covariance matrix.  
  ![](https://latex.codecogs.com/gif.latex?\Sigma(\theta)%20\approx%20\frac{\sigma^2}{N}%20\sum_{i=1}^{N}%20x_i%20x_i^T%20=%20\frac{\sigma^2}{N}%20X^T%20X%20=%20\sigma^2%20H)
  2) The speed at which one can minimize an expected loss using stochastic methods dependson two properties: the curvature of the loss and the variance of the gradients.
  3) The distinction between the Fisher matrix, the Hessian, and the covariance matrixof the gradients.
  
- [Three Factors Influencing Minima in SGD](https://arxiv.org/pdf/1711.04623.pdf), 2018 ICANN
- [Entropy-SGD: Biasing Gradient Descent Into Wide Valleys](https://arxiv.org/pdf/1611.01838.pdf), 2017 ICLR
- [Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/pdf/1606.04838.pdf), 2018 SIAM Review


## Information Bottleneck
- [Disentangled Information Bottleneck](https://arxiv.org/pdf/2012.07372.pdf), 2021 AAAI


## Batch Normalization
- [Cross-Iteration Batch Normalization](https://arxiv.org/pdf/2002.05712.pdf), 2021 CVPR


## Physics & ML
- [Hidenori Tanaka](https://sites.google.com/view/htanaka/home)


## Random Matrix Theory
- [Geometry of Neural Network Loss Surfaces via Random Matrix Theory](https://dl.acm.org/doi/pdf/10.5555/3305890.3305970), 2017 ICML
- [Random Matrix Theory Proves that Deep Learning Representations of GAN-data Behave as Gaussian Mixtures](http://proceedings.mlr.press/v119/seddik20a/seddik20a.pdf), 2020 ICML
- [Applicability of Random Matrix Theory in Deep Learning](https://arxiv.org/pdf/2102.06740v1.pdf), 2021 arXiv


## Covariance Matrix
- [Hidenori Tanaka](https://sites.google.com/view/htanaka/home)
- [Matrix Power Normalized Covariance Pooling For Deep Convolutional Networks](http://peihuali.org/iSQRT-COV/index.html)
