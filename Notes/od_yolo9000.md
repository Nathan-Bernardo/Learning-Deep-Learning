# YOLO9000: Better, Faster, Stronger
*February 2020*

**Overall Impression** <br>
YOLO9000 is definetely a better model compared to the original YOLO that Joseph Redmon developed.  Real-time detection systems are taking advantage of YOLOv2 and YOLO9000 because of it's improvement in performance and accuracy, though still has trouble detecting other categories like clothing and equipment.

**Key Ideas** <br>
* Batch normalization and high resolution classifiers improve the mAP by about **6**% in total.
* Due to the convolution with anchor boxes, there are two problems: box dimensions are hand picked and model instability in the early iterations.
  1. K-means clustering is used in the train set bounding boxes to obtain better priors.
  1. For model stability, location coordinates relative to the location of the grid cell are predicted instead of the offsets. Predicting the location coordinates bounds the ground truth to [0 1], making it easier for the network to learn.  Predicting offsets consequentially takes a long time, so model will take a while to stabilize.
* Multi-scale training is used to force network to learn different resolutions, improving the detection.  The tradeoff is between speed and accuracy; smaller the resolution, the faster the detection.
* Data combination, or joint data sets, is accomploshed with WordTree so that the network can simultaneously learn the ImageNet and COCO dataset for detection and classification.
* **Neural network architecture** is called **Darknet-19**, consisting of 19 convolutional layers and 5 maxpooling layers.  Achieves **72.9**% in top-1 accuracy and **91.2**% top-5 accuracy on ImageNet.  Requires only 5.58 million operations to process an image.  This is the base for YOLOv2.

**Technical Details** <br>
* Softmax layer for classification was disregarded for joint datasets. Softmax assumes the classes as mutally exclusive.  A multi-label model was used for joint datasets because it does not assume mutually exclusivity.

**Further Reading**

