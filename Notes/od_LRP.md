# Localization Recall Precision (LRP): A New Performance Metric for Object Detection
*March 2020*

**Overall Impression** <br>
The average precision (AP) has been the standard performance metric evaulation in object detection.  However, there are a few deficienies that comes from using AP as a performance metric for evalation.  The LRP Error metric is capable of capturing the weaknesses and strengths of the detector by representing the peak values of the recall precision (RP) curves and their localization capability. Another beneift to this new metric is the optimal LRP (oLRP), which determines the optimal confidence score threshold for a class, balancing the trade-offbetween localization and recall-precision.

**Key Ideas** <br>
* Shortcomings of AP:
  * Ap cannot tell the difference between various RP curves, given that multiple object detection results have the same AP.
  * Does not address the localization accuracy.
  * Not confidnece-score sensitive.
  * Does not suggest the "best" confidence scorethreshold for better performance of the object detector.
  * Uses interpolation between beighboring recall values.
* oLRP addresses the AP's deficiencies.  The metric represents the tightness of the bounding-boxes and the shape of the RP curve.  
* LRP outputs the localization accuracy of the true positives (TP), false positives (FP), and false negatives (FN).
* oLRP is suitable for ablation studies.
* LRP in experiments will represent the sharpness 

**Technical Details** <br> 
* F-measure is a popular information theoretic performance measure for object detection.  Derived from the confusion matrix.  But fails as a performance metric because it violates the triangle inequality.  Not symmetric in addition in the positive and negative classes.
* **Hungarian match algorithm** was implemented for updating the confidence score in oLRP.
* **Deficiency aware subpattern assignment (DASA)** is a proven metric that the writer used to prove that LRP is a true,viable metric for evaulation.

**Further Reading** <br>
* [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
* [Hungarian Maximum Matching Algorithm](https://brilliant.org/wiki/hungarian-matching/)
* [Hungarian assignment algorithm](https://www.topcoder.com/community/competitive-programming/tutorials/assignment-problem-and-hungarian-algorithm/)
