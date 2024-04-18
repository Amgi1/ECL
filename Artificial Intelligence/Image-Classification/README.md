# Image classification

## Description
This repository is based on an image classification project at ECL as part of the Artificial Intelligence course. In this repository we implement solutions using the k-nearest neighbours algorithm as well as an artificial neural network consisting of a multi-layer perceptron.

The database used for both training and testing is the CIFAR dataset, which consists of 60 000 color images of size 32x32 divided into 10 classes. The file read_cifar.py contains our methods for initializing the dataset. 

## Usage

### K-Nearest Neighbours

To use the k-nearest-neighbours algorithm one needs to use two separate functions from the knn.py file. First of all, function distance_matrix returns the distance matrix between your training dataset and the dataset containing images whose labels you wish to predict. 

Then, using function knn_predict, you can set hyperparameter k and use the precalculated distance matrix to predict the labels associated to your test dataset.

Finally, function evaluate_knn is used to evaluate the accuracy of the k-nearest neighbours algorithm implemented above on our dataset for a specific value of hyperparameter k. It is later used to study the effect of the value of k on the accuracy of the algorithm, the results being saved in the results/knn.png file. Compared to the demands of the project, the function contains additional arguments 'dists' and 'set_dists' used to input a distance matrix if it had already been calculated, thus saving valuable time when used iteratively. These arguments are taken into account when 'set_dists' is set to True.


### Artificial Neural Network

File 'Artificial Neural Network.md' contains the maths used to write the backpropagation.

The learning steps of the MLP are done at every epoch. Each learning step contains a forward pass that outputs a prediction, loss calculation, a backward pass calculating the gradients needed for backpropagation, and the update of the parameters of the MLP. Functions learn_once_mse and learn_once_cross_entropy both contain an entire learning step, with the difference being the function used for the loss calculation.

Use function train_mlp to train a preexisting model by inputting its own weights and biases. To start training of a new model, use random values for weights and 0 for biases.

Function test_mlp lets a user test his model.

To test the validity of our MLP model, use function run_mlp_training as such:
```python
run_mlp_training(data_train, labels_train, data_test, labels_test, d_h=64, learning_rate=0.1, num_epoch=100)
```
Do note this function does not allow to save trained model. To do so, please use function train_model.

### UnitTest

Some unit tests are included in directory tests. 

Note : Function test_mlp in file mlp.py will return a failed test when running pytest on the entire directory, this error is not to be considered.

## Authors and acknowledgment
Author : Amaury Giard

The project is based on guidelines set in repository MOD_4_6-TD1 published by Emmanuel Dellandrea.
