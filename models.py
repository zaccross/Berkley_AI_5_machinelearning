"""
IMPORTANT NOTE: To run autograder script, need to run python3 autograder.py, otherwise
training the perceptron results in seg fault
"""

import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        # Want to return the dot product of x and w:
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        # default score
        score = -1
        # if dot product >= 0, return 1, else the default
        if nn.as_scalar(self.run(x)) >= 0:
            score = 1
        return score


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        # *** MY CODE HERE *** #
        # Set batch size, so we get 1 val at a time
        batch_size = 1

        # Loop through the training set and update weights until 100 % accuracy
        missed = True
        while missed is True:
            missed = False

            # Iterate through training set 1 (x,y) at a time
            for x, y in dataset.iterate_once(batch_size):
                if float(self.get_prediction(x)) != nn.as_scalar(y):

                    # If it miss classifies, update weights and set missed to True to repeat
                    self.w.update(x, nn.as_scalar(y))
                    missed = True

        return None
        # *** END OF MY CODE *** #



class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.b0 = nn.Parameter(1, 100)
        self.b1 = nn.Parameter(1, 1)
        self.w0 = nn.Parameter(1, 100)
        self.w1 = nn.Parameter(100, 1)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        lin_1 = nn.Linear(x, self.w0)
        relu_1 = nn.ReLU(nn.AddBias(lin_1, self.b0))
        lin_2 = nn.Linear(relu_1, self.w1)

        return nn.AddBias(lin_2, self.b1)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        pred = self.run(x)

        return nn.SquareLoss(pred, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        #grad_dict = {self.a: 10e10, self.b: 10e10, self.c: 10e10, self.d: 10e10}

        batch_size = 50
        cnt = 0
        for x, y in dataset.iterate_forever(batch_size):

            loss = self.get_loss(x, y)
            grad_wrt_b0, grad_wrt_b1, grad_wrt_w0, grad_wrt_w1 = nn.gradients(loss, [self.b0, self.b1, self.w0, self.w1])

            multiplier = -0.05
            self.b0.update(grad_wrt_b0, multiplier)
            self.b1.update(grad_wrt_b1, multiplier)
            self.w0.update(grad_wrt_w0, multiplier)
            self.w1.update(grad_wrt_w1, multiplier)
            cnt+=1

            if cnt%1000 == 0:
                print(cnt, nn.as_scalar(loss))
            if nn.as_scalar(loss) < 0.01:
                print("made mark")
                print(nn.as_scalar(loss))

                break

        return None

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        # "*** YOUR CODE HERE ***"
        n = 256
        self.b0 = nn.Parameter(1, n)
        self.b1 = nn.Parameter(1, n//2)
        self.b2 = nn.Parameter(1, n//4)
        self.b3 = nn.Parameter(1, n//8)
        self.b4 = nn.Parameter(1, 10)
        self.w0 = nn.Parameter(784, n)
        self.w1 = nn.Parameter(n, n//2)
        self.w2 = nn.Parameter(n//2, n//4)
        self.w3 = nn.Parameter(n//4, n//8)
        self.w4 = nn.Parameter(n//8, 10)

        self.param_list = [self.b0, self.b1, self.b2, self.b3, self.b4,
                           self.w0, self.w1, self.w2, self.w3, self.w4]



    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        lin_0 = nn.Linear(x, self.w0)
        relu_0 = nn.ReLU(nn.AddBias(lin_0, self.b0))
        lin_1 = nn.Linear(relu_0, self.w1)
        relu_1 = nn.ReLU(nn.AddBias(lin_1, self.b1))
        lin_2 = nn.Linear(relu_1, self.w2)
        relu_2 = nn.ReLU(nn.AddBias(lin_2, self.b2))
        lin_3 = nn.Linear(relu_2, self.w3)
        relu_3 = nn.ReLU(nn.AddBias(lin_3, self.b3))
        lin_4 = nn.Linear(relu_3, self.w4)
        return nn.AddBias(lin_4, self.b4)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # Run training and return loss
        y_pred = self.run(x)
        return nn.SoftmaxLoss(y_pred, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        # roughly the sqrt of the len of the data set
        batch_size = 250

        # Target validation accuracy, described more below
        validation_target = 0.975

        # Go through the data set forever, training on batches
        # Can get away with this basic setup because the autograder takes
        # care of epochs and plotting for us
        for x, y in dataset.iterate_forever(batch_size):

            # Use typical flow: calculate loss and find the gradient
            # via back propagation
            loss = self.get_loss(x, y)
            gradients = nn.gradients(loss, self.param_list)

            # Our learning rate
            # Currently uses a fixed rate, but it would be interesting
            # to try decay
            multiplier = -0.1

            # Go through and update the weights with the learning rate dotted with the gradients
            for i in range(len(self.param_list)):
                self.param_list[i].update(gradients[i], multiplier)

            # Want test accuracy > 97%
            # Using validation accuracy of 97.5 tends to pass this requirement
            if dataset.get_validation_accuracy() >= validation_target:

                # End training if we hit our target
                break

        return None
