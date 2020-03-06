# Assignment 13

1. Refer to your Assignment 12.
2. Replace whatever model you have there with the ResNet18 model as shown below.
3. Your model must look like Conv->B1->B2->B3->B4 and not individually called Convs. 
4. If not already using, then:
    1. Use Batch Size 128
    2. Use Normalization values of: (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    3. Random Crop of 32 with padding of 4px
    4. Horizontal Flip (0.5)
    5. Optimizer: SGD, Weight-Decay: 5e-4
    6. NOT-OneCycleLR
    7. Save model (to drive) after every 50 epochs or best model till now
5. Describe your blocks, and the stride strategy you have picked
6. Train for 300 Epochs
7. Assignment Target Accuracy is 90%, so exit gracefully if you reach 90% (you can target more, it can go till ~93%)
8. Assignment has hard deadline and any assignment submitted post deadline will not be accepted. 

The ResNet18 model is build your following layers-
## Input Layer ##
1. 1 Convolution layer with 32 3x3 kernels, stride 1 -> BatchNormalization -> Relu
  (In the original architecture, 7x7 kernel with stride 2 is used. We're using 3x3 kernel with stride 1, because the input 
  resolution is just 32x32. Use use 32 kernels because this is enough capacity for this resolution)
2. MaxPooling with kernel 2x2, stride 2 (Original architecture uses 3x3 kernel with stride 2. Again, due to small resolution, 
  we stick with 2x2, stride 2. This gives an output resolution of 16x16. The resolution stays **16x16 hereafter**)
  
## Resnet blocks ##
The number of channels is reduced to provide enough capacity for learning for CIFAR10 dataset. Also, we stick with 3x3 kernels and
stride 1 and apply padding to our images to keep the same resolution. Since our image is already 16x16, we do not want to reduce the 
resolution again as we might lose spatial information.

BLock 1 - 4 Convolution layers with 32 3x3 kernel, stride 1 -> BatchNormalization -> Relu\
BLock 2 - 4 Convolution layers with 64 3x3 kernel, stride 1 -> BatchNormalization -> Relu\
Block 3 - 4 Convolution layers with 128 3x3 kernel, stride 1 -> BatchNormalization -> Relu\
BLock 4 - 4 Convolution layers with 256 3x3 kernel, stride 1 -> BatchNormalization -> Relu

There is a shortcut connection after every 2 convolution layer as in the original architecture.

## Output layers ##

1. GLobalAveragePooling2D layer
2. Dense layer with 10 units to match the output number of classes

We train the model for 50 epochs and achieve a maximum validation accuracy of 87.71% at the 43rd epoch. Both the train and validation
accuracy saturates for the last 10 epochs. 
