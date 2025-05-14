## AlexNet Legacy Implementation

This repo is for testing the original AlexNet paper and feeling what its main issues are, this is not for maximizing performance, and is just a small project I am doing to understand it better. If you find this interesting I am glad. 


From initial tests, I noticed that AlexNet is very unstable without the Gaussian init and bias initailization that is said in the paper. You need weight decay and momentum to get the thing to even start leraning. My first thought was that Kaiming Uniform would still be best for the ReLU's used in AlexNet, but i was having convergense issues on my dummy test, I will test again later with CIFAR10 & CIFAR100. 

For CIFAR10 & 100 I will need to tune in the capacity a bit.

I want to test a few things (Almost in an Ablation test, but more just comparison.)
* Augmentations
    * Go heavy with augmentations and see how you do!!!
* Loaders
    * Get num workers working on local machine (Have been having bugs)
* Normalization
    * LRN (like the paper)
    * Batch-Norm
    * Layer-Norm
* Optimizers
    * Tuned SGD from AlexNet Paper
    * Adam -> no tuning
* LR Schedulers: TEST
    * ReduceLROnPlateau
    * StepLR
    * ExponentialLR
    * CosineAnnealingLR
* Activations
    * ReLU
    * LeakyReLU
    * GELU
    * Swish
* Init Strategies
    * Kaiming
    * Gaussian + the bias (1) for the paper layers
* Pooling:
    * Max Pool: Like in the paper
    * Average pool: modern aproach
* Dropout: With and without
* Architecture changes
    * I want to add Global Average Pool to replace the FC layers at the end, see how that effects accuracy.


This is another change


BN
* Crucial for stability: if you don't add it the model will not learn well.


EXPERIMENTS: CIFAR10
* No Augmentation or Dropout: 20 epochs
* Augmentation
* Dropout
* Augmentation & Dropout
* Best combo of Normalization & LR Scheduling like in the paper.


For this model
* You have Conv with LRN and Max Pooling
* To fight overfitting you have capacity, augmentation, and dropout.