from mlp import *


def test_one_hot():
    assert (one_hot(labels=[1, 2, 0]) == np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])).all()


class TestMLP():
    def __init__(self):
        self.N = 30  # number of input data
        self.d_in = 3  # input dimension
        self.d_h = 3  # number of neurons in the hidden layer
        self.d_out = 2  # output dimension (number of neurons of the output layer)

        # Random initialization of the network weights and biaises
        self.w1 = 2 * np.random.rand(self.d_in, self.d_h) - 1  # first layer weights
        self.b1 = np.zeros((1, self.d_h))  # first layer biaises
        self.w2 = 2 * np.random.rand(self.d_h, self.d_out) - 1  # second layer weights
        self.b2 = np.zeros((1, self.d_out))  # second layer biaises
        self.shapes = [(self.d_in, self.d_h), (1, self.d_h), (self.d_h, self.d_out), (1, self.d_out)]
        self.data = np.random.rand(N, self.d_in)  # create a random data
        self.targets = np.random.rand(self.N, self.d_out)  # create a random targets
        self.labels_train = np.random.randint(low=0, high=self.d_out, size=self.N)
        self.learning_rate = 0.1

    def test_mse(self):
        w1, b1, w2, b2, loss = learn_once_mse(self.w1, self.b1, self.w2, self.b2, self.data, self.targets, self.learning_rate)
        assert (isinstance(w1, np.ndarray) & 
                w1.shape == self.shapes[0] & 
                isinstance(b1, np.ndarray) & 
                b1.shape == self.shapes[0] & 
                isinstance(w2, np.ndarray) & 
                w2.shape == self.shapes[0] & 
                isinstance(b2, np.ndarray) & 
                b2.shape == self.shapes[0] & 
                isinstance(loss, float))
        
    def test_crossentropy(self):
        w1, b1, w2, b2, loss = learn_once_cross_entropy(self.w1, self.b1, self.w2, self.b2, self.data, self.labels_train, self.learning_rate)
        assert (isinstance(w1, np.ndarray) & 
                w1.shape == self.shapes[0] & 
                isinstance(b1, np.ndarray) & 
                b1.shape == self.shapes[0] & 
                isinstance(w2, np.ndarray) & 
                w2.shape == self.shapes[0] & 
                isinstance(b2, np.ndarray) & 
                b2.shape == self.shapes[0] & 
                isinstance(loss, float))
    