# Rich Features Hierarchies for Accurate Object Detection and Semantic Segmentation
*February 2020*

**Overall Impression** <br>
The combination of region proposals and convolutonal neural networks (CNN) has led the writers of this paper to achieve a mean average precision (mAP) of **54.3**% on the PASCAL VOC dataset. R-CNN has outperformed other algorithms by a large margin, but still poses a few problems in terms of processing speed, cost, and complexity.

**Key Details**
* In the ablation study, the R-CNN achieved an mAP of **31.0**%  with the PASCAL VOC set. Performance may have been affected by overfitting since dataset consisted of 200 classes.
* R-CNN achieved an mAP of **31.4**% on the ILSVRC2013 detection dataset.
![Results](https://i.imgur.com/GFbULx3.png)

**Technical Details**
* *Selective search* algorithm was used for detection work and works with R-CNN's region proposal method.
* 2000 region proposals were extracted from each image, increasing the real-time analysis by a large margin.
* Neural network architecture was taken from Simonyan and Zisserman. [VGG16](https://arxiv.org/abs/1409.1556)

**Further Reading**
* [Step by step VGG16 implementation in Keras for beginners](https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c)
