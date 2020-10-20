## First Neural Network

Build a network so that it has-
1. Less than 20k parameters
2. Achieves 99.4% validation accuracy

The model I built has-
1. Total params: 18,602; trainable params: 18,314
2. Achieves 99.5% validation accuracy

The model has the following features due to which it is able to achieve such a high accuracy with very less parameters-
1. Use MaxPooling at the appropriate position keeping the receptive field of edges, gradients, textures and patterns in the images in mind.
2. Add BatchNormalization to stabilize the network and achieve fast convergence.
3. Use dropout to add regularization to the network.
4. Add 1x1 convolution to significantly reduce the number of parameters. Apart from being computationally less expensive, it acts as a filter to pass only the relevant information to the next layer.
