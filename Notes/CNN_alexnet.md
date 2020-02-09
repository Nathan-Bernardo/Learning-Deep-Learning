# ImageNet Classification with Deep Convolutional Neural Networks
*February 2020*

**Overal Impression** <br>
Due to the large datasets and existence of GPUS, object recognition had a major improvement in performance when combined with convolutional neural networks (CNN).  Before the large datasets became available to researchers, simple recognition was still achievable with smaller datasets, but the lack of variability introduce inaccuracies in a realist setting.

Fei-Fei Li address the historical problems of object recognition and how their large dataset, ImageNet, gives researchers and practioners more room in developing better network architectures to improve performance and accuracy.  The video can be found [here](https://www.ted.com/talks/fei_fei_li_how_we_re_teaching_computers_to_understand_pictures?language=en).

**Key Ideas** <br>
* CNN are easier to train since their structure contains less connections and parameters, but are still considered optimal in object detection.  Have better **locality of pixel dependencies** and **stationary of statistics**.
* Used Imagenet as their large dataset, containing over 15 million labeled high-resolution images with over 22,000 categories.

**Technical Details** <br>
* Images where downsampled to 256 x 256 for constant input dimensionality.
* **Architecture**: 8 leanred layers - 5 convolutional layers and 3 fully-connected.
* **ReLU** trains several times faster than *tanh*. 
* Two **GTX 580 GPUs** were used for parallel processing, where half of the kernels were split between the two.  Each GPU processed a certain layer.  Reduced top-1 and top-5 error rates by 1.7% and 1.2%, respectively.
* **Local normalization** aids generalization.  Reduced top-1 and top-5 error rates by 1.4% and 1.2%.
* Pooling layers in CNns reduces top-1 and top-5 error rates by 0.4% and 0.3%, respecitvely.
* **Data agumentation** was used to prevent overfitting. Extracted 224 x 224 patches from 256 x 256 images, altered intensities of RGB channels in training set (performed PCA on the set of RGB pixels).
* **Dropout** forced neurons to learn robust features.
* Models were trainined using **stochastic gradient descent**.

**Further Reading** <br>
* [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
* [ML | Stochastic Gradient Descent (SGD)](https://www.geeksforgeeks.org/ml-stochastic-gradient-descent-sgd/)
* [Dropout](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/)
* [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
* [A Practical Guide to ReLU](https://medium.com/@danqing/a-practical-guide-to-relu-b83ca804f1f7)
* [Difference between Local Response Normalization and Batch Normalization](https://towardsdatascience.com/difference-between-local-response-normalization-and-batch-normalization-272308c034ac)



