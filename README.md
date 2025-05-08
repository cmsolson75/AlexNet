## AlexNet Legacy Implementation

This repo is for testing the original AlexNet paper and feeling what its main issues are, this is not for maximizing performance, and is just a small project I am doing to understand it better. If you find this interesting I am glad. 


From initial tests, I noticed that AlexNet is very unstable without the Gaussian init and bias initailization that is said in the paper. You need weight decay and momentum to get the thing to even start leraning. My first thought was that Kaiming Uniform would still be best for the ReLU's used in AlexNet, but i was having convergense issues on my dummy test, I will test again later with CIFAR10 & CIFAR100. 


Batch Normalization is king for stopping this and I want to test how it differs when you use batch normalization, I would assume it would have less gradient spikes.


For CIFAR10 & 100 I will need to tune in the capacity a bit.

I want to test a few things (Almost in an Ablation test, but more just comparison.)
* Batch Normalization: This is to replace the LocalResponseNorm that was in the original paper.
* Train with Adam instead of the SGD that is in the paper. I forgot how much you have to configure SGD to get it to even learn, I want to see if default Adam deals with issues, and improves over tuned SGD.
* I want to use torch.optim.lr_scheduler.CosineAnnealingLR(...) instead of torch.optim.lr_scheduler.ReduceLROnPlateau(...).
* I want to test with Kaiming Init over Gaussian.
* I want to see if Batch Norm effects dropout.
* I want to add Global Average Pool to replace the FC layers at the end, see how that effects accuracy.
