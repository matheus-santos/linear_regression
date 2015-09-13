# !/bin/python
# Author: Matheus Cesario <mts.cesario@gmail.com>
# Description: Getting fancy with linear regression.
#   Using gradient descent to evaluate linear regression problems.

# Imports
import numpy as np


# Linear Regression class
class LinearRegression(object):
    """ LinearRegression model """

    # Properties
    X = []  # Dataset
    Y = []  # Results dataset
    T = []  # Theta parameters
    a = 0.01  # Learning rate alpha
    m = 0  # Size of X
    n = 0  # Number of features in X

    # Ctor
    def __init__(self, X=[], Y=[], a=0.01):
        self.setDataset(X, Y)  # Dataset
        self.a = a  # Defining learning rate alpha

    # Defining learning rate alpha
    def setAlpha(self, a):
        self.a = a
        return a

    # Minimizing dataset using different alpha values
    # until find best suit
    def learnAlpha(self, decrease_factor=0.001):

        previous_e = e = 1000
        step = 0

        while True:
            previous_e = e

            self.minimize()
            e = self.getErrorRate()
            self.setAlpha(self.a - decrease_factor)
            step += 1

            if previous_e <= e or step == 1000:
                break

        return self.a

    # Setting dataset
    def setDataset(self, X=[], Y=[]):

        if len(X) == 0:
            print("Error: Insufficient rows into dataset.")
            return False

        self.X = X
        self.Y = Y
        self.m = len(self.X)  # Lenght of X
        self.n = len(self.X[0])  # Number of features in X
        self.T = [0] * self.n  # Initializing T

        return True

    # Calculating error
    def getErrorRate(self):

        # Calculating error based on dataset
        e = p = 0
        for x, y in zip(self.X, self.Y):
            p = self.predict(x)  # Predicting result
            e += pow((y - p), 2)/y  # Error

        return (e/self.m)

    # Minimizing Theta parameters
    def minimize(self):

        # Iterating trough datasets X and Y
        for x, y, i in zip(self.X, self.Y, xrange(0, self.m)):

            # For each column from x
            for xj, j in zip(x, xrange(0, self.n)):

                # Initializing variables
                previous_cost = cost = 100000
                step = 0

                # Gradient descent algorithm
                while(True):

                    # Defining old theta parameter
                    previous_cost = cost

                    # Calculating new cost function
                    cost = self.derivative(xj)/self.m

                    # Storing new parameter
                    self.T[j] = self.T[j] - self.a * cost

                    # Increasing step counter
                    step += 1

                    # Finishing if converge or reach step count
                    if previous_cost <= cost or step == 1000:
                        break

        # Minimized theta parameters
        return self.T

    # Calculating hyphotesis H(Xi)
    def hypothesis(self, x):
        """ Calculating linear equation H, where x and T are 1xN matrices """

        # Transforming lists into Numpy matrices
        matrix_t = np.matrix(self.T)
        matrix_x = np.matrix(x).transpose()  # Transposing x

        # Extracting product value
        return (matrix_t * matrix_x).flat[0]

    # Calculating partial derivative of xj
    def derivative(self, xj):
        s = 0
        for x, y in zip(self.X, self.Y):
            s += (self.hypothesis(x) - y) * xj
        return s

    # Predicting result
    def predict(self, x):
        return sum(np.array(x) * np.array(self.T))

# Exporting module
if __name__ :
    pass
