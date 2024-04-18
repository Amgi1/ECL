import numpy as np

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def learn_once_mse(w1, b1, w2, b2, data, targets, learning_rate):
    N = data.shape[0]  # number of input data

    # Forward pass
    a0 = data # the data are the input of the first layer
    z1 = np.matmul(a0, w1) + b1  # input of the hidden layer
    a1 = 1 / (1 + np.exp(-z1))  # output of the hidden layer (sigmoid activation function)
    z2 = np.matmul(a1, w2) + b2  # input of the output layer
    a2 = 1 / (1 + np.exp(-z2))  # output of the output layer (sigmoid activation function)
    predictions = a2  # the predicted values are the outputs of the output layer

    # Compute loss (MSE)
    loss = np.mean(np.square(predictions - targets))

    # Backward pass (Gradient Descent)
    delta_a2 = 2 * (predictions - targets) / N
    delta_z2 = delta_a2 * (sigmoide(a2) * (1 - sigmoide(a2)))
    delta_w2 = np.matmul(a1.T, delta_z2)
    delta_b2 = np.sum(delta_z2, axis=0)
    delta_a1 = np.matmul(delta_z2, w2.T)
    delta_z1 = delta_a1 * (sigmoide(a1) * (1 - sigmoide(a1)))
    delta_w1 = np.matmul(a0.T, delta_z1)
    delta_b1 = np.sum(delta_z1, axis=0)

    # Update weights and biases
    w1 -= learning_rate * delta_w1
    b1 -= learning_rate * delta_b1
    w2 -= learning_rate * delta_w2
    b2 -= learning_rate * delta_b2

    
    return w1, b1, w2, b2, loss


def one_hot(labels):
    maxi = np.max(labels) + 1
    return np.eye(maxi)[labels]

def softmax(x):
    exps = np.exp(x)
    return(exps/exps.sum())


def learn_once_cross_entropy(w1, b1, w2, b2, data, labels_train, learning_rate):
    N = data.shape[0]

    # Forward pass
    a0 = data
    z1 = np.matmul(a0, w1) + b1
    a1 = 1 / (1 + np.exp(-z1))
    z2 = np.matmul(a1, w2) + b2
    a2 = softmax(z2)
    predictions = a2

    # Compute loss (Cross Entropy)
    targets = one_hot(labels_train)
    loss = np.mean(np.square(predictions - targets))

    # Backward pass (Gradient Descent)
    #delta_a2 = 2 * (predictions - targets) / N
    #delta_z2 = delta_a2 * (sigmoide(a2) * (1 - sigmoide(a2)))
    delta_z2 = (a2 - targets) / N
    delta_w2 = np.matmul(a1.T, delta_z2)
    delta_b2 = np.sum(delta_z2, axis=0)
    delta_a1 = np.matmul(delta_z2, w2.T)
    delta_z1 = delta_a1 * (sigmoide(a1) * (1 - sigmoide(a1)))
    delta_w1 = np.matmul(a0.T, delta_z1)
    delta_b1 = np.sum(delta_z1, axis=0)

    # Update weights and biases
    w1 -= learning_rate * delta_w1
    b1 -= learning_rate * delta_b1
    w2 -= learning_rate * delta_w2
    b2 -= learning_rate * delta_b2
 
    return w1, b1, w2, b2, loss

def accuracy(w1, b1, w2, b2, data, labels):
    # Forward pass
    a0 = data
    z1 = np.matmul(a0, w1) + b1
    a1 = 1 / (1 + np.exp(-z1))
    z2 = np.matmul(a1, w2) + b2
    a2 = softmax(z2)
    predictions = a2
    return np.mean(np.argmax(predictions, axis=1) == labels)


def train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch):
    train_accuracies = []
    for _ in range(num_epoch):
        w1, b1, w2, b2, loss = learn_once_cross_entropy(w1, b1, w2, b2, data_train, labels_train, learning_rate)
        train_accuracies.append(accuracy(w1, b1, w2, b2, data_train, labels_train))
    return w1, b1, w2, b2, train_accuracies
    

def test_mlp(w1, b1, w2, b2, data_test, labels_test):
    N = data.shape[0]

    # Forward pass
    a0 = data_test
    z1 = np.matmul(a0, w1) + b1
    a1 = 1 / (1 + np.exp(-z1))
    z2 = np.matmul(a1, w2) + b2
    a2 = softmax(z2)
    predictions = a2

    return np.mean(np.argmax(predictions, axis=1) == labels_test)


def run_mlp_training(data_train, labels_train, data_test, labels_test, d_h, learning_rate, num_epoch):
    # Input constants
    N = data_train.shape[0]
    d_in = data_train.shape[1]
    d_out = max(labels_train)+1
    
    # Random initialization of the network weights and biaises
    w1 = 2 * np.random.rand(d_in, d_h) - 1  # first layer weights
    b1 = np.zeros((1, d_h))  # first layer biaises
    w2 = 2 * np.random.rand(d_h, d_out) - 1  # second layer weights
    b2 = np.zeros((1, d_out))  # second layer biaises

    # Training
    w1, b1, w2, b2, train_accuracies = train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch)

    # Testing
    test_accuracy = test_mlp(w1, b1, w2, b2, data_test, labels_test)

    return train_accuracies, test_accuracy


if __name__ == "__main__":
    from read_cifar import *
    import matplotlib.pyplot as plt

    data, labels = read_cifar('data/cifar-10-batches-py/')
    data_train, labels_train, data_test, labels_test = split_dataset(data, labels, 0.9)
    train_accuracies, test_accuracy = run_mlp_training(data_train, labels_train, data_test, labels_test, d_h=64, learning_rate=0.1, num_epoch=100)


    plt.plot(train_accuracies, label="Train")
    test_accuracies=[test_accuracy for i in range(len(train_accuracies))]
    plt.plot(test_accuracies, label="Test")
    plt.title('Accuracy of MLP model during training')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.savefig('results/mlp.png')
