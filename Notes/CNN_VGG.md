# Very Deep Convolutional Neural Networks for Large-Scale Image Recognition
*February 2020*

**Overall Impression**
For large data sets, increasing the depth from 16 to 19 weight layers from  the conventional ConvNet architectures (LeCunn et al., Krizhevsky et al.) improves the performance.

**Key Ideas**
* Scale jittering improves the performance of classification.  Multi-scale (scale jittering) outperforms a single-scale evaluation.
* With scale jittering, combining multi-crops and dense evaluation decreases error rate.  Their combination outperforms their own evaluation.
* Ensembling Convnets reduces the error rate.  7 models were ensembled to achieve a **7.3**% error rate, but 2 models ensembled achieved a **6.8**% error rate in the top-5 test.
* *Local Response Normalization* had no improvement on the ILSVRC dataset; led to an increase in memory consumption and computation time.
* Compared to model and data parallism introduced by (Krizhevsky, 2014), off-the-shelf 4-GPU system increased computation time by 3.75 compared to a single GPU. (Four NVIDIA Titan Black GPUs were used).

**Technical Details**
* **Architecture**: 16-19 weight layers with convolutional layer being 3x3.  Max-pooling (5 layers total) is performed over a 2x2 pixel window, with stride 2.
* All hidden layers contained ReLU for non-linearity.

**Further Reading**
* [Activation Functions in Neural Networks](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)
