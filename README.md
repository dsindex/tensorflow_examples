tensorflow
===

### description
- test code for tensorflow
- [tensorflow](https://www.tensorflow.org/)
  - version : tf 1.0 

### simple test code
- [test.py](https://github.com/dsindex/tensorflow/blob/master/test.py)

### [tutorial video by hunkim(in korean)](http://hunkim.github.io/ml/)
- linear regression
  - predict real value
  - code
    - [linear_regression1.py](https://github.com/dsindex/tensorflow/blob/master/linear_regression1.py)
    - [linear_regression2.py](https://github.com/dsindex/tensorflow/blob/master/linear_regression2.py)
    - [linear_regression3.py](https://github.com/dsindex/tensorflow/blob/master/linear_regression3.py)
    - [linear_regression4.py](https://github.com/dsindex/tensorflow/blob/master/linear_regression4.py)
    - [train_linear.txt](https://github.com/dsindex/tensorflow/blob/master/train_linear.txt)
- logistic regression
  - binary classification 0 or 1
  - code
    - [logistic_regression.py](https://github.com/dsindex/tensorflow/blob/master/logistic_regression.py)
    - [train_logistic.txt](https://github.com/dsindex/tensorflow/blob/master/train_logistic.txt)
- softmax regression
  - multinomial (logistic) classification
  - code
    - [softmax_regression.py](https://github.com/dsindex/tensorflow/blob/master/softmax_regression.py)
    - [train_softmax.txt](https://github.com/dsindex/tensorflow/blob/master/train_softmax.txt)
- RNN
  - recurrent neural network
  - code
    - [simple_rnn.py](https://github.com/dsindex/tensorflow/blob/master/simple_rnn.py)
- [tutorial git](https://github.com/hunkim/DeepLearningZeroToAll)

### IRIS softmax regression test code
- code
  - [softmax_regression_iris_train.py](https://github.com/dsindex/tensorflow/blob/master/softmax_regression_iris_train.py)
  - [softmax_regression_iris_inference.py](https://github.com/dsindex/tensorflow/blob/master/softmax_regression_iris_inference.py)
  - [train_iris.txt](https://github.com/dsindex/tensorflow/blob/master/train_iris.txt)
  - training accuracy : around 96%
  - for other traning data
  ```shell
  - convert training data format into the format like train_iris.txt
  - modify softmax_regression3_train.py
  - train and save model
  - modify softmax_regression3_inference.py
  - restore model and test inference
  - these steps are basic usage what you are familar with
  ```

### IRIS multi-layer perceptron test code
- code
  - [mlp_iris.py](https://github.com/dsindex/tensorflow/blob/master/mlp_iris.py)
    - accuracy : around 96%

### MNIST softmax regression test code
- download MNIST data from http://yann.lecun.com/exdb/mnist/
- code
  - [softmax_regression_mnist.py](https://github.com/dsindex/tensorflow/blob/master/softmax_regression_mnist.py)
    - accuracy : around 92%
- softmax regression is same as : 
  - multinomial logistic regression 
  - maximum entropy classifier
  - neural net without hidden layer

### MNIST multi-layer perceptron test code
- code
  - [mlp_mnist_train.py](https://github.com/dsindex/tensorflow/blob/master/mlp_mnist_train.py)
  - [mlp_mnist_inference.py](https://github.com/dsindex/tensorflow/blob/master/mlp_mnist_inference.py)
    - accuracy : around 98%
- distributed version
  - [mlp_mnist_dist.py](https://github.com/dsindex/tensorflow/blob/master/mlp_mnist_dist.py)
  - training using parameter servers and workers
  ```shell
  $ ./mlp_mnist_dist.sh -v -v

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
  ```
  - accuracy :  0.9604
  - if you have a trouble like 'failed to connect...', [read](http://stackoverflow.com/questions/37729746/failed-to-run-tensorflow-distributed-mnist-test)

### MNIST convolutaional neural network test code
- code
  - [conv_mnist1.py](https://github.com/dsindex/tensorflow/blob/master/conv_mnist1.py)
    - accuracy : around 99%
  - [conv_mnist2.py](https://github.com/dsindex/tensorflow/blob/master/conv_mnist2.py)
    - accuracy : around 98.15%

### MNIST LSTM(recurrent neural network) test code
- code
  - [lstm_mnist.py](https://github.com/dsindex/tensorflow/blob/master/lstm_mnist.py)
    - accuracy : around 97%

### RNN
- code
  - [simple_lstm.py](https://github.com/dsindex/tensorflow/blob/master/simple_lstm.py)
- [segmentation(auto-spacing) using lstm](https://github.com/dsindex/segm-lstm)
- [char-rnn](https://github.com/sherjilozair/char-rnn-tensorflow)

### word2vec
- [vector representation of words](https://www.tensorflow.org/tutorials/word2vec)

### tensorflow serving
- [setup tensorflow serving](https://tensorflow.github.io/serving/setup)
  - if you have trouble on installing gRPC, see http://dchua.com/2016/04/08/installing-grpc,-protobuf-and-its-dependencies-for-python-development/
  ```shell
  $ sudo find /usr/lib -name "*protobuf*" -delete
  $ git clone https://github.com/grpc/grpc.git
  $ cd grpc/
  $ git submodule update --init
  $ cd third_party/protobuf
  # install autoconf, libtool (on OS X)
  $ brew install autoconf && brew install libtool
  $ ./autogen.sh
  # if you got an error related to 'libtool' on OS X, edit Makefile to use '/usr/bin/libtool' instead of '/usr/local/bin/libtool'
  $ ./configure; make; sudo make install
  $ cd python
  $ python setup.py build; python setup.py test; sudo python setup.py install --user
  $ cd ../../..
  $ make; sudo make install
  $ which grpc_python_plugin
  $ pip install grpcio --user
  ```
- [serving basic](https://tensorflow.github.io/serving/serving_basic)

### data preprocessing(tf.SequenceExample)
- [RNNS IN TENSORFLOW, A PRACTICAL GUIDE AND UNDOCUMENTED FEATURES](http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/)
  - [in korean](https://tgjeon.github.io/post/rnns-in-tensorflow/)

## model dump and inference
  - simple example
    - [dump_model.py](https://github.com/dsindex/tensorflow/blob/master/dump_model.py)
    - [inference_model.py](https://github.com/dsindex/tensorflow/blob/master/inference_model.py)

### references
- [RNN](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)
- [LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Word2Vec](https://github.com/dsindex/blog/wiki/%5BWord2Vec%5D-Neural-Language-Model-and-Word2Vec)
- tensorflow
  - [tensorflow tutorial](https://github.com/sherrym/tf-tutorial/blob/master/DeepLearningSchool2016.pdf)
  - [tensorflow mnist inference test using web interface](https://github.com/sugyan/tensorflow-mnist)
  - [distributed tensorflow](https://www.tensorflow.org/versions/r0.8/how_tos/distributed/index.html)
  - [distributed tensorflow, a gentle introduction](http://amid.fish/distributed-tensorflow-a-gentle-introduction)
  - [tensorboard]( https://www.tensorflow.org/versions/master/how_tos/summaries_and_tensorboard/index.html)
  - [tensorflow C++ image recognition demo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/label_image)
- gpu check
  - command
  ```shell
  $ lspci
  $ nvidia-smi
  $ cat /proc/driver/nvidia/gpus/0/information
  $ cat /proc/driver/nvidia/gpus/1/information
  $ cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
  ```
