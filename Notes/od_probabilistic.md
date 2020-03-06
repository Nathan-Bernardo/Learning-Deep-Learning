# Probabilistic Object Detection: Definition and Evaulation
*March 2020*

**Overall Impression** <br>
Probability-based Object Detection Quality (PDQ) measure a valauble metric for getting a grasp of the model's Label Quality and 
Spatial Quality in object detection.  Unlike convential object detection methods where mean average precision (mAP) and mean optimal
Localization-Recall Precision (moLRP) reply on fixed threshods or tuneable parameters, PDQ provides optimal values for detection by rewarding the model for detecting ground
-truth objects.

**Key Ideas** <br>
* YOLOv3 has a high performance in Labe Quality, but through the PDQ metric the algorithm has a lack 
in performace in Spatial Quality; meaning that the model is more uncertain about the object's location in the image.
* AP-based measures rely on fixed IOUs, leading to the likihood of overfitting.  Model's accuracy can be largely affected
with a small change in the thresholds.
* Unlike mAP and moLRP, PDQ will penalize any false-positive and false-negative detections.

**Technical Details** <br>
* Esimated uncertainty is measured through the Monte Carlo (MC) dropout technique.  A few researchers such as Miller et al. and Kendall et al.
have demostrated benefit of estimating the uncertainity for each class, improving vision tasks.  In addition, the proposed methods deal with
pixel-wise classification.
* *Hungariam Algorithm* was used to provide optimal assigmnent between the Label and Spatial Quality to provide the pair-wise PDQ
score (pPDQ).

**Further Reading**
