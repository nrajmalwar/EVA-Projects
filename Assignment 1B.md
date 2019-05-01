# Assignment 1B

## 1.  What are Channels and Kernels? ##

When we extract a particular feature from an image (or data), we call this a channel. These channels represent some basic features of the image such as edges, gradients, textures and patterns, and can further represent complex features such parts of object and the object itself. The basic features can be combined to build the more complex features.

To give an example, consider you are painting a landscape picture. You have your painting canvas in place and you are ready to paint. So you quickly grab your basic ingredients - 10 bottles of paint (for different colors) and 10 painting brushes (for different strokes). Here, your bottles of paint and brushes are your basic channels. Some simple or complex combination of these tools will give some more channels. Like for example, a horizontal stroke can be called a channel. You can have this stroke in 10 different colors and 10 different brushes, so you add some more channels of horizontal strokes.

![picture alt](https://github.com/nrajmalwar/Project-1/blob/master/Images/Oil%20Paint%20Bottles.jpg)

![picture alt](https://github.com/nrajmalwar/Project-1/blob/master/Images/Paint%20brushes.jpg)

![picture alt](https://github.com/nrajmalwar/Project-1/blob/master/Images/Brush%20Strokes.jpg)

Similarly, when we do convolution on an image, we first extract some low level features and put them into a layer of channels. Then we do convolution again, and these channels give us more channels containing mid level features. We put these channels into a separate bucket. Similarly, with every convolution we get higher level features which can be subsequently used to construct our image.

A kernel is a matrix which can be used to extract features from an image. It is also called a feature extractor or a filter. An image is basically a matrix of numbers containing the pixel intensities as values. In order to extract useful information from the image, we can scan the entire image with a kernel, say a 3x3 matrix. The values of the kernel could be designed in a specific way to extract a particular type of feature, like a vertical edge. Similarly, we can have numerous such kernels which can extract different features on its own.

When we scan the entire image with a kernel, the output of the kernel is called a channel. So if we use 32 different kernels on our image, we get 32 different channels as the output.

![picture alt](https://github.com/nrajmalwar/Project-1/blob/master/Images/Filters.gif)

In the image above, the matrix is orange is called a kernel and the matrix is red is called a channel.

## 2.  Why should we only (well mostly) use 3x3 Kernels? ##

A  3x3 or an odd sized kernel has several advantages when compared to an even sized kernel.

First of all, a 3x3 kernel has a central value. This  central value can be used as a reference point when convolving over an image and thus gives us symmetry. With this as reference point, we can also now say what is left, right, up and down corresponding to this point. We can pad an image uniformly so that the central value convolves over the entire image exactly once.

A 3x3 Kernel with a central value

![picture alt](https://github.com/nrajmalwar/Project-1/blob/master/Images/3x3.png)

A 2x2 Kernel with no central value

![picture alt](https://github.com/nrajmalwar/Project-1/blob/master/Images/2x2.png)

Also, if we need to identify the neighbours of a single pixel in our image, we can only use a 3x3 matrix.

We can calculate the centered gradient of a 3x3 matrix, which uses both forward and backward values for calculating the gradient. Let us explain this through Taylor's Approximation-

![picture alt](https://github.com/nrajmalwar/Project-1/blob/master/Images/Taylor's%20Approximation.PNG)

When we take a forward or one sided gradient, it looks like this

![picture alt](https://github.com/nrajmalwar/Project-1/blob/master/Images/Forward_Numerical_Gradient.PNG)

Whereas a centered gradient or two sided gradient looks like this

![picture alt](https://github.com/nrajmalwar/Project-1/blob/master/Images/Centered_Numerical_Gradient.PNG)

The two sided gradient plays on symmetry, and cancels out the contribution of the square term in Taylor's Expansion. So, when you decrease Δx, the two sided's error diminishes much quicker compared one sided. Hence, a 3x3 gives a better value of gradient since it plays on symmetry.

When it comes to modern GPU resources, they are also optimized for odd sized kernels.

## How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations)? ##

![picture alt](https://github.com/nrajmalwar/Project-1/blob/master/Images/Convolution%20Calculations.png)

We need to perform 99 3x3 Convolution operations in order to reach 1x1 from 199x199.
