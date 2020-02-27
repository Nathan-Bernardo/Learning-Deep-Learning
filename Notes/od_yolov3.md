# YOLOv3: An Incremental Improvement
*February 2020*

**Overal Impression**
YOLOv3 is incrementally better than it's prior network, YOLO9000.  YOLOv3 can detect smaller objects, performs much faster with 320x320, and much more accurate,  

**Key Ideas**
* Compared to the 19-layer convolutional neural network (Darknet-19), YOLOv3 is a 53-layer convolutional neural network (Darknet-53).
* Darknet-53 better utilizes the GPU, and achieves highest-measured floating point operations per second (about 1457 BFLOP/s) cinoared to ResNet-152 that pefrosm 1090 BFLOP/s.

**Technical Details**
* Anchor box x, y offset predictions decreased model stability, therefore was not compatible with YOLOv3.
* Linear x, y predicitons instead of logisitc decreases mAP by a couple points.
* Focal loss dropped mAP by a couple points.

**Further Reading**
