# Visualizing and Understanding Convolutional Networks
*February 2020*

**Overal Impression**
Being able to visualize the function of each intermediate layer and it's given classifier in your neural network architecture is important for debugging and improving the performance of your model.

**Key Ideas**
* Decovnet is used for visualizing the intermediate layers.  Has the same features as a covnet (filters, pooling), but maps the features onto a pixel map rather than mapping pixels to features.
* Ablation study was performed to test the performance of the neural network.  Removing fully connected layer and one convolutional layer (leaving a total of 4 layers in total) drastically affects the performance.
* Occlusion study was performed for correlation analysis. In result, the neural network implicity computes the correlation between images; highly sensistive to local structures rather than at a borad context.  

**Technical Details**
* Correspondance was computed using *Hamming Distance*.
* With the multi-layer Deconvolutional Network, Krizhevsky et al. 's architecture had various problems: mid-frequencies were not covered very well while the first layer filters had a mixture of low and high frequencies; 2nd layer showed aliasing artifacts caused by their stride of 4 used in the 1st layer convolutons.
* Znet architecture: Similiar to Krizhevsky et al., but (i) 1st layer filter size is 7x7 (not 11x11), and (ii) stride is 2.  New architecture maintains more information in the 1st and 2nd layer features, and imporves classification performance.
* znet has achieved an error rate of **14.8**%.

**Further Reading**
