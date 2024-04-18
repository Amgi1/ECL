import numpy as np


def distance_matrix(a, b):
    sum_squares_a = np.sum(np.square(a), axis=1, keepdims=True)
    sum_squares_b = np.sum(np.square(b), axis=1, keepdims=True)
    dists = np.sqrt(sum_squares_a + sum_squares_b.T -2*np.dot(a,b.T))
    return dists

def knn_predict(dists, labels_train, k):
    res = []
    for i in range(dists.shape[1]):
        nns = np.argpartition(dists[:, i], k)
        preds = np.zeros((max(labels_train)+1,))
        for j in range(k):
            preds[labels_train[nns[j]]] += 1
        res.append(np.argmax(preds))
    return np.array(res)

def evaluate_knn(data_train, labels_train, data_test, labels_test, k, dists = None, set_dists=False):
    if not set_dists:
        dists = distance_matrix(data_train, data_test)
    predictions = knn_predict(dists, labels_train, k)
    return np.sum(predictions == labels_test)/len(labels_test)


if __name__ == "__main__":
    from read_cifar import *
    import matplotlib.pyplot as plt

    data, labels = read_cifar('data/cifar-10-batches-py/')
    data_train, labels_train, data_test, labels_test = split_dataset(data, labels, 0.9)

    res = []
    dists = distance_matrix(data_train, data_test)
    for k in range(1, 21):
        res.append(evaluate_knn(data_train, labels_train, data_test, labels_test, k, dists=dists, set_dists=True))
        print(res[-1])

    plt.plot(res)
    plt.title('Accuracy of KNN model depending on value of k')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.savefig('results/knn.png')


    