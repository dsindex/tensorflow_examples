# tensorflow

- description
  - test code for tensorflow
- [tensorflow](https://www.tensorflow.org/)
  - how to install in OS X
    - install docker
    - [tensorflow docker installation](https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#docker-installation)
- very simple test code
```
test.py
```
- [tutorial video by hunkim(in korean)](http://hunkim.github.io/ml/)
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
- IRIS softmax regression test code
```
* code
  softmax_regression3_train.py
  softmax_regression3_inference.py
  train_iris.txt
* for other traning data :
  - convert training data format into the format like train_iris.txt
  - modify softmax_regression3_train.py
  - train and save model
  - modify softmax_regression3_inference.py
  - restore model and test inference
  - these steps are basic usage what you are familar with
```
- MNIST softmax regression test code
```
* download MNIST data from http://yann.lecun.com/exdb/mnist/
* code
  softmax_regression2.py
* accuracy is around 92%
```
- MNIST multi-layer perceptron test code
```
* code
  mlp1.py
* accuracy is around 98%
```
- MNIST convolutaional neural network test code
```
* code
  conv1.py
* accuracy is around 99%
```
