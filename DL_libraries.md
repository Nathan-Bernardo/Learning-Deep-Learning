# Dedicated Deep Learning Libraries

**Tensorflow** <br>
A machine learning (ML) library developed by Google. It is GPU accelerated and allows ML implementation on single and multiple architectures, allowing fast deep neural networks on both cloud andfog services.Programming langugaes supported are:
* Python
* Java
* C
* C++
* Go <br>

Tensorflow providesa visualization tool called the *TensorBoard*. Helps users understand how their model works and how data is flowing, while also allowing users to debug their model. Supported higher-level programming itnerfaces include:
* Keras
* Luminoth
* TensorLayer <br>

[Documentation](https://www.tensorflow.org/)

**Theano** <br>
A Python library that allows user to define, optimize, and evaluate numberical computations involving multi-dimensional data. Supports both CPU and GPU modes.  Cons include slow compile time, but has a large community. <br>
[Documentation](http://deeplearning.net/software/theano/)

**(Py)Torch** <br>
A scientific computing framework with wide support for machine learning models and algorithms.  Does lack visualization tools, but easy to debug and well documentented.  Programming langauages supported:
* Lua
* Python
* C
* C++ <br>

Has no higher-level programming interface.

[Documentation](https://pytorch.org/)

# Keras vs tf.keras
Before I give a brief description of the difference, most of my information come from the founder of **pyimagesearch**, Adrian Rosebrock. He provides well-detailed, step-by-step instructions in practicing Deep Learning in general and for Computer Vision.  I am provide a review not only for myself, but those who need a very brief description of the differences between Keras vs. tf.keras.  Let's get started!

**Why Keras?** <br>
In context, Keras was developed by **Francois Chollet**, a Google AI Developer/Researcher. Overtime, researchers picked up Keras because of it was easy to use, compared to Torch, Theano, and Caffe that were required more effort.

**Keras requires a backend** <br>
A backend is a **computational engine**.  Essentially what it does is build the network graph/toplogy, runs the optimizers, and performs the computations.  In the case for Keras, our backend is our database and Keras is the programming language that we use to access the database. <br>

**TensorFlow 2.0**
Keras then began supporting Tensorflow, resulting in the submodule **tf.keras**.  Tensorflow 2.0 and tf.keras pprovides better multi-GPU support and distrubuted training.

Original source => [Keras vs. tf.keras: What’s the difference in TensorFlow 2.0?](https://www.pyimagesearch.com/2019/10/21/keras-vs-tf-keras-whats-the-difference-in-tensorflow-2-0/)








