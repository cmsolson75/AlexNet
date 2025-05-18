## AlexNet Legacy Implementation

This repo is for testing the original AlexNet paper and feeling what its main issues are, this is not for maximizing performance, and is just a small project I am doing to understand it better. If you find this interesting I am glad. 

Tech Plan
* PyTorch: For main components
* Pytorch-Lighting: for Training Wrapping
* W&B: for Logging
* YAML/HYDRA for config
* Optuna for sweeping.


TODO
* Checkpoint Naming and Rate issues: Get a unified name.
* Optuna: I have the ability to sweep so i shall!
    * Review Optuna with PTL
    * Use Hydra-Optuna-Sweeper plugin:
        * Use for optimizing layer size and hyperparameters like LR & more.
        * Run this on CIFAR100 model
* Make TinyImageNet Model: https://paperswithcode.com/dataset/tiny-imagenet
* Update the Hydra Config system to be more Idiomatic with Hydra. I am not using important features that I want to try.
* Improve README.md to make it functional for a archived style repo (I will not come back to this)

EXPERIMENTS: CIFAR10
* Link to model file [file_link](/alexnet/configs/model/alexnet.yaml)
* No Augmentation or Dropout: 20 epochs
    * 74% accuracy
* Augmentation
    * Test Accuracy: 77%
* Dropout
    * Test Accuracy: 78%
* Augmentation & Dropout
    * Test Accuracy: 78%
* Best combo of Normalization & LR Scheduling like in the paper: 40 epochs
    * Test Accuracy: 86%
    ```
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1,)
    ```
