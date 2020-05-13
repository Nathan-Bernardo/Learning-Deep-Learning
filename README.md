# About
Ever since I picked up computer vision, all I ever wanted to do in my daily routine is code code and code.  I'm currently self-teaching myself Deep Learning and Machine Learning and I'm intrigued by the large potential it has in every occupation.  Inspired by [Adrian Coyler](https://blog.acolyer.org/about/) and [Patrick Lui](https://github.com/patrick-llgc/Learning-Deep-Learning), I brought myself to read papers to keep myself up to date on the development of AI.  And just like Coyler and Lui, I will summarize every paper I read so that not only I have an understanding of the paper, but so that you can understand the topic without in depth reading.  I would love to share my paper notes to those who are curious, but need an overview of the core concepts.  With all said, if you are a researcher, practioner, or undergrad student, I hope you can take something away from my exploration to AI.

# Getting Started
When I began learning the theory and application of Deep Learning, I read the book ***Deep Learning with Python*** by Francois Chollet, the creator of Keras. The book is great for getting started with programming using Tensorflow and Keras, but some concepts didn't seem intuitive (Neural Netowrks in general are just difficult to understand). To pair with the book, I took the the ***Deep Learning Specialization*** course by Andrew Ng, founder of deeplearning.ai. Andrew Ng does a great job in describing the theoretical concepts on hyperparameters, convolutional neural networks (CNN), optimization methods, backpropagation, and other concepts essential for understanding Neural Networks. I personally recommend checking these two out as a starting point: <br>
* [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python)
* [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)

The above two resources requires you to have a basic understanding ***Calculus*** (derivatives specifically) and ***Linear Algebra***. To get deep into the bones of Deep Learning, it is required that you strongly understand ***Linear Algebra***(35%), ***Probability and Statistics*** (25%), ***Multivariate Calculus*** (15%), ***Algorithms and Data Structures*** (15%), and ***Others*** (10%).  Others refer to ***Real and Complex Analysis***, ***Fourier Transforms***, ***Information Theory***, and other topics not covered in the top 4 topics. To get started, I recommend reading ***Deep Learning*** by Ian Goodfellow et al., founder of **Generative Adversial Networks**. He provides you all the math concepts and it's application to machine learning and deep learning all in one book. I will also provide other sources, such as this minibook called *Mathematics for Machine Learning*. <br>  
* [Deep Learning](https://www.deeplearningbook.org/) 
* [Mathematics for Machine Learning](https://mml-book.github.io/)

There are many resources in getting started with Machine Learning and Deep Learning. To explore more of the sources that availabe for you, check out Tensorflow. Tensorflow provides people a guide to how you can go from a beginner to expert in theory and in practice. There are other popular deep learning frameworks that are often used for Deep Learning, such as **Keras**, **Caffe**, and **PyTorch**. Tensorflow, Keras, and Pytorch have been the most popular frameworks used in building NNs. Tensorflow 2.0 actually integrated Keras into their APIs, making the user experience much simpler and easier for generating NNs.  Though, do explore all and see which framework floats your boat. On the other hand, Pytorch now integrates Caffe into their APIs. <br>
* [Tensorflow](https://www.tensorflow.org/resources/learn-ml/theoretical-and-advanced-machine-learning)
* [PyTorch](https://pytorch.org/)

When you begin understanding the different Neural Networks (NN) and their application, you may need a deep understanding of the NN architecture and the math behind the process. In most resources, you'll be presented with popular Neural Network architectures, ranging from it's revolution to Deep Learning to contemporary Deep Learning. Below are the most popular NNs that are mentioned in most literature today, and I encourage you to read these papers: <br>
* [Alex Net](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) [[notes](https://github.com/Nathan-Bernardo/Learning-Deep-Learning/blob/master/Notes/CNN_alexnet.md)] <br>
* [ZF Net](https://arxiv.org/pdf/1311.2901v3.pdf) [[notes](https://github.com/Nathan-Bernardo/Learning-Deep-Learning/blob/master/Notes/CNN_znet.md)] <br>
* [VGG16](https://arxiv.org/abs/1409.1556) [[notes](https://github.com/Nathan-Bernardo/Learning-Deep-Learning/blob/master/Notes/CNN_VGG.md)] <br>
* [GoogleLetNet](https://arxiv.org/abs/1409.4842) [[notes](https://github.com/Nathan-Bernardo/Learning-Deep-Learning/blob/master/Notes/CNN_googleLeNet.md)] <br>
* [Inception-v2](https://arxiv.org/abs/1502.03167) <br>
* [Inception-v3](https://arxiv.org/abs/1512.00567) <br>
* [Inception-ResNet](https://arxiv.org/abs/1602.07261) <br>
* [Microsoft ResNet](https://arxiv.org/pdf/1512.03385v1.pdf) <br>
* [R-CNN](https://arxiv.org/pdf/1311.2524v5.pdf) [[notes](https://github.com/Nathan-Bernardo/Learning-Deep-Learning/blob/master/Notes/R_CNN.md)] <br>
* [Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf) <br>
* [Faster R-CNN](https://arxiv.org/pdf/1506.01497v3.pdf) <br>
* [Xception](https://arxiv.org/pdf/1610.02357.pdf) <br>
* [Generative Adversial Networks](https://arxiv.org/pdf/1406.2661v1.pdf) <br>
* [Generating Image Description](https://arxiv.org/pdf/1412.2306v2.pdf) <br>
* [Spatial Transformer Networks](https://arxiv.org/pdf/1506.02025.pdf) <br>

## Environments to code in
It is always nice to write your code using a notebook, IDE, or a text editor. (Most of the code you will be writing will be in Python). There are a couple notebooks that you can use to write your code: **Google Collab** or **Jupyter Notebooks**. Notebooks are create for organizing your code into blocks, fast prototyping, and reiteration of a specific block of code. Google Collab is a personal favorite because you can save your notebooks in your Google Drive, and you have free access to their powerful GPUs and TPUs. GPUs and TPUs drastically improves the training and test time, especially when working with images and CNNs. With Jupyter Notebooks, you get to work with more coding programs, but have no access to a GPU and TPU.  Though, I encourage you to explore both notebooks.
* [Google Collab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true)
* [Jupyter Notebooks](https://jupyter.org/)

Some of the IDEs that you can use for computer programming are **PyCharm** (developed my JetBrains), **Microsoft Visual Studio Code** (VS Code), or **Spyder**. PyCharm supports virtual environments, allowing you to install scientific packages for certain projects without having to interface with the main system.  My favorite feature of Pycharm is it's scientific tool, which allows you to easily install scientific packages with just a click. This feature is only allowed if you have the professional edition, and students have the opportunity to get a year long license.  On the other hand, VS 
Code and Spyder are free IDEs, and you can make use of Anaconda to simplify package management and deployment. 
* [Pycharm](https://www.jetbrains.com/pycharm/)
* [VS Code](https://code.visualstudio.com/)
* [Spyder](https://www.spyder-ide.org/)


## Github Repos
* [Faster R-CNN and Mask R-CNN in PyTorch 1.0](https://github.com/facebookresearch/maskrcnn-benchmark)
* [LRP (Localization Recall Precision) Performance Metric & Thresholder for Object Detection](https://github.com/cancam/LRP) [[notes](https://github.com/Nathan-Bernardo/Learning-Deep-Learning/blob/master/Notes/od_LRP.md)]
* [Metrics for Object Detection](https://github.com/rafaelpadilla/Object-Detection-Metrics)

# Paper Notes
## 2020-02
* [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) [[notes](https://github.com/Nathan-Bernardo/Learning-Deep-Learning/blob/master/Notes/od_yolo.md)]
* [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) [[notes](https://github.com/Nathan-Bernardo/Learning-Deep-Learning/blob/master/Notes/od_yolo9000.md)]
* [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767) [[notes](https://github.com/Nathan-Bernardo/Learning-Deep-Learning/blob/master/Notes/od_yolov3.md)]

## 2020-03
* [Probabilistic Object Detection: Definition and Evaluation](https://arxiv.org/abs/1811.10800) [[notes](https://github.com/Nathan-Bernardo/Learning-Deep-Learning/blob/master/Notes/od_probabilistic.md)]
* [Localization Recall Precision (LRP): A New Performance Metric for Object Detection](https://arxiv.org/abs/1807.01696) [[notes](https://github.com/Nathan-Bernardo/Learning-Deep-Learning/blob/master/Notes/od_LRP.md)]

## 2020-04
**Computer Vision and Pattern Recognition**
* [Objects as Points](https://arxiv.org/abs/1904.07850)
* [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)

List of articles that I will read <br>
* [Learning to Map Vehicles into Bird's Eye View](https://arxiv.org/abs/1706.08442)
* [VPGNet: Vanishing Point Guided Network for Lane and Road Marking Detection and Recognition](https://arxiv.org/abs/1710.06288)
* [The Matrix Calculus You Need For Deep Learning](https://arxiv.org/abs/1802.01528v2)
* [Soft-NMS -- Improving Object Detection With One Line of Code](https://arxiv.org/abs/1704.04503) [[Github](https://github.com/bharatsingh430/soft-nms)]
* [Weakly- and Semi-Supervised Learning of a DCNN for Semantic Image Segmentation](https://arxiv.org/abs/1502.02734)
* [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1605.06211) [[dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php)]
* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
* [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/abs/1611.09326)
* [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122)
* [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915)
* [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
* [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://paperswithcode.com/paper/encoder-decoder-with-atrous-separable)
* [FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation](https://paperswithcode.com/paper/fastfcn-rethinking-dilated-convolution-in-the)
* [Improving Semantic Segmentation via Video Propagation and Label Relaxation](https://paperswithcode.com/paper/improving-semantic-segmentation-via-video)
* [Gated-SCNN: Gated Shape CNNs for Semantic Segmentation](https://arxiv.org/abs/1907.05740)
* [When Does Labeling Smoothing Help?](https://arxiv.org/abs/1906.02629)
* [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689)
* [YOLACT++: Better Real-time Instance Segmentation](https://arxiv.org/abs/1912.06218)
* [Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530)
* [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142)
* [Dropout Sampling for Robust Object Detection in Open-Set Conditions](https://arxiv.org/abs/1710.06677)
* [MIT Advanced Vehicle Technology Study: Large-Scale Naturalistic Driving Study of Driver Behavior and Interaction with Automation](https://arxiv.org/abs/1711.06976)
* [Multitarget tracking performance metric: deficiency aware subpattern assignment](https://ieeexplore.ieee.org/document/8306032)
* [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
* [Predicting the Generalization Gap in Deep Networks with Margin Distributions](https://arxiv.org/abs/1810.00113)
* [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
* [MMDetection: Open MMLab Detection Toolbox and Benchmark](https://arxiv.org/abs/1906.07155) [[Github](https://github.com/open-mmlab/mmdetection)]
* [Spiking-YOLO: Spiking Neural Network for Energy-Efficient Object Detection](https://arxiv.org/abs/1903.06530)
* [Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression](https://arxiv.org/abs/1911.08287)
* [Gaussian YOLOv3: An Accurate and Fast Object Detector Using Localization Uncertainty for Autonomous Driving](https://arxiv.org/abs/1904.04620)
* [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)
* [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946v1) [[Github](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)]
* [M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network](https://arxiv.org/abs/1811.04533) [[code](https://github.com/qijiezhao/M2Det)]
* [Learning the Depths of Moving People by Watching Frozen People](https://arxiv.org/abs/1904.11111) [[Github](https://github.com/lukemelas/EfficientNet-PyTorch)]
* [Reasoning-RCNN: Unifying Adaptive Global Reasoning Into Large-Scale Object Detection](http://openaccess.thecvf.com/content_CVPR_2019/html/Xu_Reasoning-RCNN_Unifying_Adaptive_Global_Reasoning_Into_Large-Scale_Object_Detection_CVPR_2019_paper.html) [[Github](https://github.com/chanyn/Reasoning-RCNN)]
* [Fixing the train-test resolution discrepancy](https://arxiv.org/abs/1906.06423) [[Github](https://github.com/facebookresearch/FixRes)]
* [Local Aggregation for Unsupervised Learning of Visual Embeddings](https://arxiv.org/abs/1903.12355) [[Github](https://github.com/neuroailab/LocalAggregation)]
* [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)
* [Towards End-to-End Lane Detection: an Instance Segmentation Approach](https://arxiv.org/abs/1802.05591)
* [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) [[Github](https://github.com/davidsandberg/facenet)]
* [Deep Hough Voting for 3D Object Detection in Point Clouds](https://arxiv.org/abs/1904.09664) [[Github](https://github.com/facebookresearch/votenet)]

