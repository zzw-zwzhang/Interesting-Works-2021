# Interesting-Works
<!-- https://latex.codecogs.com/gif.latex? -->

## SGD
- [Communication Efficient SGD via Gradient Sampling with Bayes Prior](https://openaccess.thecvf.com/content/CVPR2021/papers/Song_Communication_Efficient_SGD_via_Gradient_Sampling_With_Bayes_Prior_CVPR_2021_paper.pdf), 2021 CVPR  
- [On the Origin of Implicit Regularization in Stochastic Gradient Descent](https://arxiv.org/pdf/2101.12176.pdf), 2021 ICLR
- [A Diffusion Theory For Deep Learning Dynamics: Stochastic Gradient Descent Exponentially Favors Flat Minima](https://arxiv.org/pdf/2002.03495.pdf), 2021 ICLR
- [On the Validity of Modeling SGD with Stochastic Differential Equations (SDEs)](https://arxiv.org/pdf/2102.12470.pdf), 2021 NeurIPS
- [Rethinking the Limiting Dynamics of SGD: Modified Loss, Phase Space Oscillations, and Anomalous Diffusion](https://arxiv.org/pdf/2107.09133.pdf), 2021 NeurIPS
  1) Other works have questionedthe correctness of the using the central limit theorem to model the gradient noise as Gaussian [20], arguing that the weak dependence between batches and heavy-tailed structure in the gradient noiseleads the CLT to break down.
  2) The two conditions needed for the CLT (Central Limit Theorem) to hold are not exactly met in the setting of SGD. Independent and identically distributed. Generally we perform SGD by making a complete pass through the entire dataset before using a sample again which introduces a weak dependence between samples. **While the covariance matrix without replacement more accurately models the dependence between samples within a batch, it fails to account for the dependence between batches.** *Finite variance*. A differentline of work has questioned the Gaussian assumption entirely because of the need for finite variance random variables. This work instead suggests using the generalized central limit theorem implying the noise would be a heavy-tailed α-stable random variable. Thus, the previous assumption is implicitly assuming the i.i.d. and finite variance conditions apply for large enough datasets and small enough batches.
  3) Reference: [What is the difference between finite and infinite variance](https://stats.stackexchange.com/questions/94402/what-is-the-difference-between-finite-and-infinite-variance/100161)  
     &emsp; &emsp; &emsp;&emsp; [ANITA: An Optimal Loopless Accelerated Variance-Reduced Gradient Method](https://arxiv.org/pdf/2103.11333.pdf), 2021 arXiv  
     &emsp; &emsp; &emsp;&emsp; [Accelerating Stochastic Gradient Descent using Predictive Variance Reduction](https://proceedings.neurips.cc/paper/2013/file/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Paper.pdf), 2013 NeurIPS  
   4) Large variance slows down the convergence.

- [On the Interplay between Noise and Curvature and its Effecton Optimization and Generalization](https://arxiv.org/pdf/1906.07774.pdf), 2020 AISTATS
  1) The relationship between the Hessian and the covariance matrix.  
  ![](https://latex.codecogs.com/gif.latex?\Sigma(\theta)%20\approx%20\frac{\sigma^2}{N}%20\sum_{i=1}^{N}%20x_i%20x_i^T%20=%20\frac{\sigma^2}{N}%20X^T%20X%20=%20\sigma^2%20H)
  2) The speed at which one can minimize an expected loss using stochastic methods dependson two properties: the curvature of the loss and the variance of the gradients.
  3) The distinction between the Fisher matrix, the Hessian, and the covariance matrix of the gradients.
  
- [Three Factors Influencing Minima in SGD](https://arxiv.org/pdf/1711.04623.pdf), 2018 ICANN
- [Entropy-SGD: Biasing Gradient Descent Into Wide Valleys](https://arxiv.org/pdf/1611.01838.pdf), 2017 ICLR
- [Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/pdf/1606.04838.pdf), 2018 SIAM Review
- [On the Relation Between the Sharpest Directions of DNN Loss and the SGD Step Length](https://arxiv.org/pdf/1807.05031.pdf), 2019 ICLR
- [A Bayesian Perspective on Generalization and Stochastic Gradient Descent](https://arxiv.org/pdf/1710.06451.pdf), 2018 ICLR
- [Importance Sampling for Minibatches](https://www.jmlr.org/papers/volume19/16-241/16-241.pdf), 2018 JMLR
- [Variance Reduction for Stochastic Gradient Optimization](https://papers.nips.cc/paper/2013/file/9766527f2b5d3e95d4a733fcfb77bd7e-Paper.pdf), 2013 NIPS*
- [A Study of Gradient Variance in Deep Learning](https://arxiv.org/pdf/2007.04532.pdf), 2020 arXiv [[code](https://github.com/fartashf/gvar_code)]
- [Lecture 10: Accelerating SGD with Averaging and Variance Reduction](https://www.cs.cornell.edu/courses/cs4787/2021sp/notebooks/Slides10.html), Cornell
- [New Insights and Perspectives on the Natural Gradient Method](https://arxiv.org/pdf/1412.1193.pdf), 2020 JMLR


## Generalization
- [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/pdf/2010.01412.pdf), 2021 ICLR
- [SAM: Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/pdf/2010.01412.pdf), 2021 ICLR
- [Coherent Gradients: An Approach to Understanding Generalization in Gradient Descent-based Optimization](https://arxiv.org/pdf/2002.10657.pdf), 2020 ICLR
  1) We propose an approach to answering this question basedon a hypothesis about the dynamics of gradient descent that we call **Coherent Gradients**: Gradients from similar examples are similar and so the overall gradientis stronger in certain directions where these reinforce each other.  Thus changesto the network parameters during training are biased towards those that (locally)simultaneously benefit many examples when such similarity exists.
  2) Gradients arecoherent, i.e, similar examples (or parts of examples) have similar gradients (or similar components of gradients) and dissimilar examples have dissimilar gradients.
  3) Since the overall gradient is the sum of the per-example gradients, it is stronger in directions where the per-example gradients are similar and reinforce each other and weaker in other directions where they are different and do not add up.
  4) Since network parameters are updated proportionally to gradients, they change faster in the direction of stronger gradients.
  5) Thus the changes to the network during training are biased towards those that simultaneously benefit many examples instead of a few (or one example).
  6) Strong gradient directions are more stable since the presence or absence of a single example does not impact them as much, as opposed to weak gradient directions which may altogether disappear if a specific example is missing from the training set. With this observation, we can reason inductively about the stability of GD: since the initial values of the parameters do not depend on the training data, the initial function mapping examples to their gradients is stable. Now, if all parameter updates are due to strong gradient directions, then stabilityis preserved. However, if some parameter updates are due to weak gradient directions, then stability is diminished.


## Sampling
- [Random Shuffling Beats SGD Only After Many Epochson Ill-Conditioned Problems](https://arxiv.org/pdf/2106.06880.pdf), 2021 arXiv
  1) Recently, there has been much interest in studying the convergence rates of without-replacementSGD, and proving that it is faster than with-replacement SGD in the worst case.
  2) Without-replacement sampling createsstatistical dependencies between the iterations, so the stochastic gradients computed at each iteration can nolonger be seen as unbiased estimates of gradients.
- [Curiously Fast Convergence of someStochastic Gradient Descent Algorithms](https://leon.bottou.org/publications/pdf/slds-2009.pdf), 2009
  1) In fact, the stochastic approximation results rely on randomness assump-tion on the successive choice of examples are independent. Both the cycle and the shuffle break these assumptions but provide a more even coverage of the training set.
  2) What can we prove for thecycleand the shuffle cases?
- [Without-Replacement Samplingfor Stochastic Gradient Methods](https://papers.nips.cc/paper/2016/file/c74d97b01eae257e44aa9d5bade97baf-Paper.pdf), 2016 NIPS
- [Closing the Convergence Gap of SGD without Replacement](https://par.nsf.gov/servlets/purl/10183698), 2020 ICML
- [Open Problem: Can Single-Shuffle SGD be Better thanReshuffling SGD and GD?](http://www.learningtheory.org/colt2021/virtual/static/images/yun21a.pdf), 2021 COLT
  1) With-replacement: In many theoretical studies the indices of component functions are assumed to be chosen with replacement, making the choice at each iteration independent of other iterations.
  2) Without-replacement: all the indices are randomly shuffled and are then visited exactly once per epoch (i.e., one pass through all the components). There are two popular
variants of shuffling schemes: one that reshuffles the components at every epoch and another that shuffles only once at the beginning and reuses that order every epoch.
  3) The without-replacement sampling is considerably trickier than their with-replacement counterparts, because the component chosen at each iteration is dependent on the previous iterates of an epoch.


## 2nd-order Optimization & Hessian
- [Whitening and Second Order Optimization Both Make Information in theDataset Unusable During Training, and Can Reduce or Prevent Generalization](https://arxiv.org/pdf/2008.07545.pdf), 2021 ICML
- [An Investigation into Neural Net Optimization via HessianEigenvalue Density](https://icml.cc/media/Slides/icml/2019/hallb(11-16-00)-11-16-00-4686-an_investigatio.pdf), 2019 Slide
- [Relationship between the Hessian and Covariance Matrix for Gaussian Random Variables](https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470824566.app1), 2010, Appendix
- [RMSprop Converges with Proper Hyper-parameter](https://openreview.net/pdf?id=3UDSdyIcBDA), 2021 ICLR


## Information Bottleneck & Theory
- [Opening the black box of Deep Neural Networks via Information](https://arxiv.org/pdf/1703.00810.pdf), 2017 arXiv
- [On the Information Bottleneck Theory of Deep Learning](https://openreview.net/pdf?id=ry_WPG-A-), 2018 ICLR
- [Disentangled Information Bottleneck](https://arxiv.org/pdf/2012.07372.pdf), 2021 AAAI
- [Mutual Information Neural Estimation](https://arxiv.org/pdf/1801.04062.pdf), 2021 ICML
- [Learning Deep Representations by Mutual Information Estimation and Maximization](https://arxiv.org/pdf/1808.06670.pdf), 2019 ICLR
- [A Unifying Mutual Information View of Metric Learning: Cross-Entropy vs. Pairwise Losses](https://arxiv.org/pdf/2003.08983.pdf), 2020 ECCV
- [Estimating Total Correlation withMutual Information Bounds](https://arxiv.org/pdf/2011.04794.pdf), 2020 NeurIPS-Workshop
- [CLUB: A Contrastive Log-ratio Upper Bound of Mutual Information](https://arxiv.org/pdf/2006.12013.pdf), 2020 ICML
- [Mutual Information really invariant to invertible transformations?](https://stats.stackexchange.com/questions/50184/mutual-information-really-invariant-to-invertible-transformations)
- [A Comparison of Correlation Measures](https://m-clark.github.io/docs/CorrelationComparison.pdf)
- [Estimation of Entropy and Mutual Information](https://www.stat.berkeley.edu/~binyu/summer08/L2P2.pdf), 2003
- [Mutual Information in Learning Feature Transformations](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=C05B2757CD4DE983E56A390332D83860?doi=10.1.1.28.7731&rep=rep1&type=pdf), 2000
- [Does the entropy of a random variable change under a linear transformation?](https://stats.stackexchange.com/questions/517102/does-the-entropy-of-a-random-variable-change-under-a-linear-transformation)


## Batch Normalization
- [Beyond BatchNorm: Towards a GeneralUnderstanding of Normalization in Deep Learning](https://arxiv.org/pdf/2106.05956.pdf), 2021 NeurIPS
- [Metadata Normalization](https://arxiv.org/pdf/2104.09052.pdf), 2021 CVPR
- [Cross-Iteration Batch Normalization](https://arxiv.org/pdf/2002.05712.pdf), 2021 CVPR
- [Stochastic Whitening Batch Normalization](https://arxiv.org/pdf/2106.04413.pdf), 2021 CVPR
- [Learning Intra-Batch Connections for Deep Metric Learning](https://arxiv.org/pdf/2102.07753.pdf), 2021 ICML
- [Padé Activation Units: End-to-end Learning of Flexible Activation Functions in Deep Networks](https://arxiv.org/pdf/1907.06732.pdf), 2020 ICLR
- [Rethinking "Batch" in BatchNorm](https://arxiv.org/pdf/2105.07576.pdf), 2021 arXiv


## Causal Inference
- [Causal Inference and Stable Learning](http://pengcui.thumedialab.com/papers/StableLeaning-ICML19.pdf), 2019 ICML


## Physics & ML
- [Hidenori Tanaka](https://sites.google.com/view/htanaka/home)
- [Physics-based Deep Learning](https://arxiv.org/pdf/2109.05237.pdf), 2021 BOOK


## Random Matrix Theory
- [Geometry of Neural Network Loss Surfaces via Random Matrix Theory](https://dl.acm.org/doi/pdf/10.5555/3305890.3305970), 2017 ICML
- [Nonlinear Random Matrix Theory for Deep Learning](https://papers.nips.cc/paper/2017/file/0f3d014eead934bbdbacb62a01dc4831-Paper.pdf), 2017 NeurIPS
  1) Analysis of the eigenvalues of the data covariance matrix.
- [Random Matrix Theory Proves that Deep Learning Representations of GAN-data Behave as Gaussian Mixtures](http://proceedings.mlr.press/v119/seddik20a/seddik20a.pdf), 2020 ICML
- [Applicability of Random Matrix Theory in Deep Learning](https://arxiv.org/pdf/2102.06740v1.pdf), 2021 arXiv
- [Random Matrices in Machine Learning](https://afia.asso.fr/wp-content/uploads/2018/09/Stats-IA_RCouillet-2.pdf), 2018
- [Why Deep Learning Works:Implicit Self-Regularization in Deep Neural Networks](http://helper.ipam.ucla.edu/publications/mlpws2/mlpws2_16011.pdf), 2019 Slide
- [Recent Advances in Random Matrix Theoryfor Modern Machine Learning](http://cs.if.uj.edu.pl/matrix/files/Liao.pdf), 2019 Slide
  1) Sample covariance matrix in the large n, p regime, at least (p-n) zero eigenvalues.
- [Understanding and Improving Deep Learning with Random Matrix Theory](https://stats385.github.io/assets/lectures/Understanding_and_improving_deep_learing_with_random_matrix_theory.pdf), 2017 Slide


## Covariance Matrix
- [Hidenori Tanaka](https://sites.google.com/view/htanaka/home)
- [Matrix Power Normalized Covariance Pooling For Deep Convolutional Networks](http://peihuali.org/iSQRT-COV/index.html)
- [Estimation of the covariance structure of heavy-taileddistributions](https://arxiv.org/pdf/1708.00502.pdf), 2017 NeurIPS
- [Law of Log Determinant of Sample Covariance Matrix andOptimal Estimation of Differential Entropy forHigh-Dimensional Gaussian Distributions](http://www.stat.yale.edu/~hz68/Covariance-Determinant.pdf) 
- [Shape Matters: Understanding the Implicit Bias of theNoise Covariance](https://arxiv.org/pdf/2006.08680.pdf), 2021 COLT


## Image Generation
- [Infinite Nature: Perpetual View Generation of Natural Scenes from a Single Image](https://infinite-nature.github.io/), 2021 ICCV Oral
- [Repopulating Street Scenes](https://grail.cs.washington.edu/projects/repop/), 2021 CVPR
- [No Shadow Left Behind: Removing Objects and their Shadows using Approximate Lighting and Geometry](http://grail.cs.washington.edu/projects/shadowremoval/), 2021 CVPR
- [Repopulating Street Scenes](https://arxiv.org/pdf/2103.16183.pdf), 2021 CVPR
- [People as Scene Probes](https://arxiv.org/pdf/2007.09209.pdf), 2020 ECCV
- [TediGAN: Text-Guided Diverse Face Image Generation and Manipulation](https://github.com/IIGROUP/TediGAN), 2021 CVPR
- [Object-Centric Learning with Slot Attention](https://arxiv.org/pdf/2006.15055.pdf), 2020 NeurIPS
- [Decomposing 3D Scenes into Objects via Unsupervised Volume Segmentation](https://stelzner.github.io/obsurf/), 2021 arXiv
- [Feature-wise Transformations](https://distill.pub/2018/feature-wise-transformations/), 2018 Blog
- [Concept Grounding with Modular Action-Capsules in Semantic Video Prediction](https://arxiv.org/pdf/2011.11201.pdf), 2021 arXiv
- [Cut-and-Paste Neural Rendering](https://anandbhattad.github.io/projects/reshading/)
- [View Generalization for Single Image Textured 3D Models](https://nv-adlr.github.io/view-generalization)


## Virtual Try-On
- [Self-Supervised Collision Handling via Generative 3D Garment Models for Virtual Try-On](https://arxiv.org/pdf/2105.06462.pdf), 2021 CVPR
- [SMPLicit: Topology-aware Generative Model for Clothed People](https://arxiv.org/pdf/2103.06871.pdf), 2021 CVPR


## Out-of-Distribution Generalization
- [Out-of-Distribution Generalization: Categories and Paper List](http://out-of-distribution-generalization.com/)


## Time Series Data
- [Synthetic Healthcare Data Generation and Assessment: Challenges, Methods, and Impact on Machine Learning](https://www.vanderschaar-lab.com/papers/ICML%202021%20tutorial%20-%20synthetic%20data%20-%20M%20van%20der%20Schaar%20A%20Alaa.pdf)
- [Conditional Independence Testing using GenerativeAdversarial Networks](https://papers.nips.cc/paper/2019/file/dc87c13749315c7217cdc4ac692e704c-Paper.pdf), 2019 NeurIPS
- [Time-series Generative Adversarial Networks](https://papers.nips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf), 2019 NeurIPS
