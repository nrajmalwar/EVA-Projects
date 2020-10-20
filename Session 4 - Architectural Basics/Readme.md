# Architectural Basics

## Understanding Data

Before we start  building our model, it is very important that we understand our data. Without any knowledge on how our data looks like, it is pointless to think about what layers we should add. 

Firstly, we display a random batch of images from our dataset. We look at images from all the classes to get an idea of what kind of images each class contains and understand the details of the object (type, variations, size etc.). Then we do **image normalization [10]** in order to bring all the pixel values between 0 and 1. This is important because there could be some images where the pixel values are very skewed (for eg. some images can have 90% pixel values close to 255). We do not want our model to be biased towards any image based on shear intensity of the pixel.

## Building Basic Model Architecture - First Model

When we first start building our model, we use only the basic functions and build a vanilla model. We also overlook on some of the constraints we might have like number of parameters, training time, target validation accuracy etc. The aim of this simple model is to act as a good baseline to build a more complex model that solves our model.

We start by deciding the **number of layers [1]** our model will have based on the **Receptive Field [5]** of the object in our images. We start by adding **3x3 Convolution [4]** layers in the sequence of increasing channels and calculate the output size and receptive field alongside. Once the low level features are obtained (like edges, gradients or textures) we merge the channels of our kernels using a  **MaxPooling [2]** layers followed by a **1x1 Convolution [3]** to reduce the number of channels. This together is known as a Convolution Block and a Transition Block. We add several of these blocks till we reach an output image size of 9x9 or 7x7. After this point, adding a Convolution layer would convolve more number of times on the central pixels as compared to the boundary pixels. This is when we **stop Convolution and go ahead with a larger kernel or some other alternative [19].** 

We bring down the number of channels to the output number of classes using a 1x1 Convolution. We flatten this result and feed it to a **Softmax [6]** activation. Softmax creates a large separation between the final predictions which can be used by Backpropagation to learn faster. However, the results of a softmax function are only probability like and may not be suitable in some critical cases (like medical data). 

We compile the model and then **add validation checks [22]** while fitting the model. The validation checks are crucial and required to be added from the beginning because our model can behave very differently on the training and the test data. It can start overfitting from the very beginnning in some cases. We check the first two epochs of our model and **know whether the network is going well very early [20]**. The value of training/validation accuracy and also the gap between gives a fair idea of how this model is going to behave eventually. A network which starts with a low accuracy cannot be expected to pick up and have a higher accuracy than a network which has a higher accuracy to begin with.

With a model in place, we aim to achieve 99% validation accuracy in 10 epochs.

## Overcome Parameter Contraints and Improve Architecture - Second Model

We right away follow the architecture from our first model and decrease the number of parameters to less than 15k. We aim for a validation accuracy of 99% which is a good benchmark for a simple model like this.

We take the **channels and the number of kernels [8]** into consideration to capture the expressivity of our images and the inter/intra class variations. Based on how many kernels we add, we can now **position the MaxPooling [11]** layer and inturn decide the **position of transition layer [13]**. Deep neural networks have the ability to capture features at different levels of abstraction from edges, gradients, textures to patterns, parts of objects, object. This is where the **concept of transition layer [12]** comes in which aims to sum up one level of features and move to the next. 

The **distance of MaxPooling [17]** is at least 3-4 Convolution layer from the prediction. By the time we reach the prediction layers, the final layers should be capable of predicting our images, so we want this whole information to be passed to the softmax layer without adding a MaxPooling layer.

## Achieve Target Accuracy - Third Model

Now, we add additional features to our model to achieve an accuracy of 99.4%. We first add **Batch Normalization [9]** after every Convolution layer. Batch Normalization brings the kernel values with mean 0 and standard deviation 1. So the values lie anywhere between -1 and 1. Somehwat similar to MaxPooing, we keep **Batch Normalization at some distance [18]** from the prediction layer. We do not add Batch Normalization after the last convolution layer, as we want to feed raw values to our softmax function. 

If we run our model now, we can see that the training accuracy saturates to a higher value compared to the validation accuracy. If we continue to train further, only the training keeps increasing whereas the validation accuracy has saturated at some lower value. This is **when we introduce DropOut [16]** in our network when we know that the model is overfitting on the training set. We initially add a **DropOut [15]** layer after every Convolution layer, keep a minimal DropOut rate of 0.1. However, this strategy did not improve our results drastically so we removed and kept the DropOut layers at only two places and increased the rate to 0.25. Now the network trains better and does not overfit so easily.

Due to the addition of DropOuts, our model now trains rather slowly as some kernels are getting shut off with every iteration. So we increase the **number of epochs [14]** at this point to allow it time to learn the dataset completely.

## Further Improvements - Fourth Model

In this model, we try to decrease the computution time for model training and try to further improve the validation accuracy. 

Till now we ran our model with the default batch size of 32. We can now try a higher **Batch size [21]** which  will train the model faster. However, a higher batch requires the right value of **learning rate [7]** otherwise it may not be useful. There's still a limit to how big a batch size can be and we will try to reach this limit. It was found that with batch size of 256 and 512, the model trains extremely fast, with just 3-4 seconds per epoch. But no matter how small learning rate we choose, it is unable to perform well. We go ahead with 128 batch which works best with a learning rate of 0.003.

We also used **LRScheduler [23]** where we can define how the learning rate change with time. Here, we decay the learning rate with each epoch because as the model reaches higher accuracy, we want to get hit a better minima in our loss function. A smaller learning rate helps to achieve this.

Next, we try different optimizers like **Adam vs. SGD [24]**. One approach towards choosing different optimizers would be to read literature on this particular problem, and see if there is a consensus to choosing a optimizer, we can try that. Otherwise, we just try the optimizers and compare the results. In our case, Adam optimizer performed far better compared to SGD.

