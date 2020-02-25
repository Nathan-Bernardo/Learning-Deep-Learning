# Types of Layers and There Role

**Convolutional Neural Network** 
* **Convolutional layer** is a layer in the deep neural network in which a convolutional filter passes along an input matrix.

* **Convolutional operation** is a two-step mathematical operation:
  1. Element-wise multiplication of the convolutional filter and a slice of the input matrix.
  1. Summation of all the values in the resulting product matrix.

* **Convolution** mixes the convolutional filter and the input matrix in order to train weights. 
Without convolution, an ML algorithm would have to learn a a separate weight for every cell in a large tensor.
This operation dramatically reduces memory needed to train the model because this algorithm will only find weights for every cell in the convolutional filter.

* **Convolutional filter** is a matrix have the same rank as the input matrix, but a smaller shape.
This is one of the two actors in the convolutional operation; an elemenw-wise multiplication of the convoltutional
filter and a slice of the input matrix.


