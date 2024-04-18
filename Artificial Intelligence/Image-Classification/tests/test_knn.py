from knn import *


def test_distance_shape(): # Test if output shape corresponds to what was expected
    a = np.random.rand(10, 20)
    b = np.random.rand(6, 20)
    assert distance_matrix(a, b).shape == (10, 6)

def test_predict(): # Test if output is correct, and if equal values algorithm selects first label
    dists = np.array([[1, 2, 3],
                      [5, 1, 3]])
    labels_train = np.array([0, 1])
    assert (knn_predict(dists, labels_train, k=1) == np.array([0, 1, 0])).all()
