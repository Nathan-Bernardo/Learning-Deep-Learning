# Going Deeper with Convolution
*February 2020*

**Overall Impression** <br>
The motivation of the Inception architecture is to prevent the large set of parameters that exist in traditional convoluton neural networks, but have a high-end performance than the convential architectures, like Khrizskey. The Inception model allows for an increase in depth and width.  

**Key Ideas** <br>
* A straightforward approach of improving the performance is increasing the depth and width, but doing would make the model more prone to overfitting and any uniform increase in the number of filters (in the case of 2 chained convolutional layers) results in a quadratic increase of computation.
* 1x1 convolution layers are used for dimensionality reduction before being outputed to the expensive 3x3 and 5x5 convolutions, and are equipped with rectified linear activation (ReLU).
* Inception achieves a top-5 error rate of **6.67**% for classification.
* 22 layers deep including ones with parameters (27 with max poolin layers).
**Technical Details** <br>
* **Training Methodology**
  * Asynchronous stochastic gradient descent with 0.9 momentum.
  * Fixed learning rate schedule (learning rate decreases by 4% every 8 epochs).
  * Polyak averaging was used to creating final model used in inference time.
* Dropout layer with 70% dropout rate.
* **Neural Network Architecture**
![GoogeLeNet](https://programmer.help/images/blog/c6713da92ebca30dcdd6f59e52cacf95.jpg)

**Further Reading** <br>
