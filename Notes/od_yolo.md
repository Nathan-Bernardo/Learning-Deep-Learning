# You Only Look Once: Unified, Real-Time Object Detection
*February 2020*

**Overall Impression**
Even though the YOLO algorithm struggles to localize objects correctly, it outperforms other detection systems in real-time detection speed and generates less background errors.  

**Key Ideas**
* Detection systems like deformable parts models (DPM), R-CNN, Fast and Faster R-CNN, and Deep MultiBox all run either in a sliding window over the whole image or generate a subset of regions in the images. Problems introduced:
  * **DPM** uses disjoint pipelines for getting data out of the image, reducing speed and accuracy.  YOLO uses a single CNN to perform a cohort of operations concurrently, leading to a faster and more accurate model.
  * **R-CNN* use region proposals to find objects in images, forcing the research to finely tune indepedentant, complex pipelines which result in a very slow system (about 40 seconds per image).
  * **Fast and Faster R-CNN** offers speed and accuracy improvements, but not sufficient to work with real-time systems.
* YOLO may have accuracy issues in localizing objects, but is fast for real-time systems.

**Technical Details**
* In VOC 2007 dataset, YOLO scores **63.4**%  mean average precision (mAP).  Combined with Fast R-CNN, mAP increases to **75.0**%.
* In VOC 2012 dataset, YOLO scores **57.9**% mAP, taking into account that YOLO struggles in detecting small objects.
* Neural network architecture is inspired by GooglLetNet, but replaces inception modules with 1x1 reduction layers followed by 3x3 convolutional layers.

**Further Reading**
* [YOLO](https://pjreddie.com/darknet/yolo/)
* [VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)
* [VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
