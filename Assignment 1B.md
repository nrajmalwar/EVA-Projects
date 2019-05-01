# Assignment 1B

## 1.  What are Channels and Kernels? ##

When we extract a particular feature from an image (or data), we call this a channel. These channels represent some basic features of the image such as edges, gradients, textures and patterns, and can further represent complex features such parts of object and the object itself. The basic features can be combined to build the more complex features.

To give an example, consider you are painting a landscape picture. You have your painting canvas in place and you are ready to paint. So you quickly grab your basic ingredients - 10 bottles of paint (for differnet colors) and 10 painting brushes (for different strokes). Here, your bottles of paint and brushes are your basic channels. Some simple or complex combination of these tools will give some more channels. Like for example, a horizontal stroke can be called a channel. You can have this stroke in 10 different colors and 10 different brushes, so you add some more channels of horizontal strokes.

![picture alt](https://github.com/nrajmalwar/Project-1/blob/master/Images/Oil%20Paint%20Bottles.jpg)

![picture alt](https://github.com/nrajmalwar/Project-1/blob/master/Images/Paint%20brushes.jpg)

![picture alt](https://github.com/nrajmalwar/Project-1/blob/master/Images/Brush%20Strokes.jpg)

Similarly, when we do convolution on an image, we first extract some low level features and put them into a layer of channels. Then we do convolution again, and these channels give us more channels containing mid level features. We put these channels into a separate bucket. Similarly, with every convolution we get higher level features which can be subsequently used to construct our image.

A kernel is a matrix which can be used to extract features from an image. It is also called a feature extractor or a filter. An image is basically is matrix of numbers containing the pixel intensities as values. In order to extract useful information from the image, we can scan the entire image with a kernel, say a 3x3 matrix. The values of the kernel could be designed in a specific way to extract a particular type of feature, like a vertical edge. Similarly, we can have numerous such kernels which can extract different features on its own.

When we scan the entire image with a kernel, the output of the kernel is called a channel. So if we use 32 different kernels on our image, we get 32 different channels as the output.

![picture alt](https://github.com/nrajmalwar/Project-1/blob/master/Images/Filters.gif)

In the image above, the matrix is orange is called a kernel and the matrix is red is called a channel.
