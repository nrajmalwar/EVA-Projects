## Advanced Convolutions

### Model A - Train a classification model for CIFAR10 dataset

1. Design the network based on Receptive Field. For this model, we go beyond the receptive field of 32 till 44 in order to learn the background as well.
2. Sequentially increase the number of channels in each layer, followed by a 1x1 convolution to decrease the number of channels and feed this to MaxPooling.
3. Use Dropout at three locations with a value of 0.1 to regularize the network.
4. Do not use dense layers, instead average the value of the matrix using GlobalAveragePooling2D. Pass this final layer to softmax activation.
5. Use BatchNormalization after every Convolution layer to normalize the features.
6. Use border_mode='same' to add padding to our image. This ensures that the size of the matrix remains the same after every Convolution. The size is reduced only in the MaxPooling layer.

### Model B - Use 5 different types of convolutions on Model A

1. Normal Convolution
2. Spatially Separable Convolution  (Conv2d(x, (3,1)) followed by Conv2D(x,(3,1))
3. Depthwise Separable Convolution
4. Grouped Convolution (use 3x3, 5x5 only)
5. Grouped Convolution (use 3x3 only, one with dilation = 1, and another with dilation = 2) 
