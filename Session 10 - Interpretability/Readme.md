## Interpretability

Grad-CAM localization for "violin, fiddle" category for different rectified convolutional feature maps for VGG16-

![](https://github.com/nrajmalwar/EVA-Projects/blob/master/Session%2010%20-%20Interpretability/GradCAM_violin.png)

The model predicts the class id of 889 which corresponds to the class 'violin, fiddle', which is the correct class for our image.

1. In the first layer image, we can see that the hot spot is at several places in the image. It detects the background but also reads some background.
2. In the second layer image, the hot spot has improved and it is now focused at the bottom of the violin.
3. In the third layer image, the hot spot has further improved by focusing on the violin as well as the bow.

This shows that the GradCAM results show good results for deeper layer as compared to earlier layers.
