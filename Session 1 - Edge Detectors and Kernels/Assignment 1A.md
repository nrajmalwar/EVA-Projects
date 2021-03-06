# Assignment 1A #

In this assignment, we will manually provide the values of 3x3 kernel in order to see how it functions. The kernels are designed to extract specific features from the image.

First, let's have a look at our image:

![alt text](https://github.com/nrajmalwar/Project-1/blob/master/Images/Helvetica_Normal.png)


If we just look at the edges in the image, we will get the following image:

![picture alt](https://github.com/nrajmalwar/Project-1/blob/master/Images/Helvetica_Edges.png)

Now, we show kernels and the corresponding image for the following feature extraction:

## 1. Vertical Edge Detector ##

A vertical edge is detected when there is a sudden change in pixel intensity along the x-direction.

![picture alt](https://github.com/nrajmalwar/Project-1/blob/master/Images/Vertical_Edge.png)

## 2. Horizontal Edge Detector ##

A horizontal edge is detected when there is a sudden change in pixel intensity along the y-direction.

![picture alt](https://github.com/nrajmalwar/Project-1/blob/master/Images/Horizontal_Edges.png)

## 3. 45 Degree Angle Detector ##

A 45 degree angle is detected when there is a sudden change in pixel intensity along the 45 degree or 135 degree angles.

![picture alt](https://github.com/nrajmalwar/Project-1/blob/master/Images/45_Degrees.png)

## 4. Blur ##

A blur kernel changes a single pixel value to the average of its neighbouring pixel values.

![picture alt](https://github.com/nrajmalwar/Project-1/blob/master/Images/Blur.png)

## 5. Sharpen ##

The sharpen kernel emphasizes difference in adjacent pixel values

![picture alt](https://github.com/nrajmalwar/Project-1/blob/master/Images/Sharpen.png)

## 6. Identity ##

The identity kernel does not modify any pixel values and gives the original image as output.

![picture alt](https://github.com/nrajmalwar/Project-1/blob/master/Images/Identity.png)
