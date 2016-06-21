tensorflow
===

#### description
  - test code for tensorflow

#### [tensorflow](https://www.tensorflow.org/)
  - how to install in OS X
    - install docker
    - [tensorflow docker installation](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#docker-installation)
	- [installation using pip](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#pip-installation)
    - [installation from source](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html#installing-from-sources)

#### simple test code
```
test.py
```

#### [tutorial video by hunkim(in korean)](http://hunkim.github.io/ml/)
  - linear regression
  ```
  * predict real value
  * code
    linear_regression1.py
    linear_regression2.py
    linear_regression3.py
    linear_regression4.py
    train_linear.txt
  ```
  - logistic regression
  ```
  * binary classification 0 or 1
  * code
    logistic_regression1.py
    train_logistic.txt
  ```
  - softmax regression
  ```
  * multinomial (logistic) classification
  * code
    softmax_regression1.py
    train_softmax.txt
  ```
  - RNN
  ```
  * recurrent neural network
  * code
    simple_rnn1.py
  ```

#### IRIS softmax regression test code
```
* code
  softmax_regression_iris_train.py
  softmax_regression_iris_inference.py
  train_iris.txt
  # training accuracy : around 96%

* for other traning data :
  - convert training data format into the format like train_iris.txt
  - modify softmax_regression3_train.py
  - train and save model
  - modify softmax_regression3_inference.py
  - restore model and test inference
  - these steps are basic usage what you are familar with
```

#### IRIS multi-layer perceptron test code
```
* code
  mlp_iris.py
  # training accuracy : around 96%
```

#### MNIST softmax regression test code
```
* download MNIST data from http://yann.lecun.com/exdb/mnist/

* code
  softmax_regression_mnist.py
  # accuracy : around 92%

* softmax regression 
  = multinomial logistic regression 
  = maximum entropy classifier
  = neural net without hidden layer
```

#### MNIST multi-layer perceptron test code
```
* code
  # train
  mlp_mnist_train.py

  # inference
  mlp_mnist_inference.py

  # accuracy : around 98%

* distributed version
  mlp_mnist_dist.py

  # parameter servers and workers
  ./mlp_mnist_dist.sh

  # worker0 log
  job : worker/0 step :  0 ,training accuracy : 0.9
  job : worker/0 step :  100 ,training accuracy : 0.9
  job : worker/0 step :  200 ,training accuracy : 0.86
  job : worker/0 step :  300 ,training accuracy : 0.9
  ...

  # worker1 log
  job : worker/1 step :  0 ,training accuracy : 0.12
  job : worker/1 step :  0 ,training accuracy : 0.14
  job : worker/1 step :  300 ,training accuracy : 0.82
  job : worker/1 step :  500 ,training accuracy : 0.92
  job : worker/1 step :  600 ,training accuracy : 0.94
  ....

  # inference 
  python mlp_mnist_inference.py

  # accuracy :  0.9604
```

#### MNIST convolutaional neural network test code
```
* code
  conv_mnist.py
  # accuracy : around 99%

* using CPU, training is very slow!
```

#### MNIST LSTM(recurrent neural network) test code
```
* code
  lstm_mnist.py
  # accuracy : around 97%
```

#### RNN
  - simple lstm
  ```
  simple_lstm.py
  ```
  - segmentation(auto-spacing) using lstm
  ```
  * see https://github.com/dsindex/segm-lstm
  ```
  - char-rnn
  ```
  * see https://github.com/sherjilozair/char-rnn-tensorflow
  ```

#### word2vec
```
* build word2vec model
  word2vec.py
* more optimized model
  word2vec_optimized.py
* test word2vec model
  test_word2vec.py
```
![T-SNE sample](https://github.com/dsindex/tensorflow/blob/master/tsne.png)

#### references
  - [Naive Bayesian, HMM, Maximum Entropy, CRF](https://github.com/dsindex/blog/wiki/%5Bstatistics%5D-Naive-Bayesian,-HMM,-Maximum-Entropy-Model,-CRF)
  - [Neural Network and Deep Learning](https://github.com/dsindex/blog/wiki/%5Bneural-network%5D-neural-network-and-deep-learning)
  - [RNN](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)
  - [LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  - [Word2Vec](https://github.com/dsindex/blog/wiki/%5BWord2Vec%5D-Neural-Language-Model-and-Word2Vec)
  - tensorflow
    - [tensorflow mnist inference test using web interface](https://github.com/sugyan/tensorflow-mnist)
    - [distributed tensorflow](https://www.tensorflow.org/versions/r0.8/how_tos/distributed/index.html)
    - [tensorboard]( https://www.tensorflow.org/versions/master/how_tos/summaries_and_tensorboard/index.html)
	- [tensorflow C++ image recognition demo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/label_image)
  - gpu check
    - command
	```
    `lspci` 
	`nvidia-smi`
	`cat /proc/driver/nvidia/gpus/0/information`
	`cat /proc/driver/nvidia/gpus/1/information`
	```
