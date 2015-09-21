# !/bin/python
# Author: Matheus Cesario <mts.cesario@gmail.com>
# Description: Getting fancy with linear regression.
#   Using gradient descent to evaluate linear regression problems.

# Imports
import numpy as np
import decimal


# Linear Regression class
class LinearRegression(object):
    """ LinearRegression model """

    # Properties
    X = []  # Dataset
    Y = []  # Results dataset
    T = []  # Theta parameters
    a = 0  # Learning rate alpha
    m = 0  # Size of X
    n = 0  # Number of features in X

    # Info

    # Ctor
    def __init__(self, X=[], Y=[], a=0.01, prepend_x0=True):
        self.setDataset(X, Y, prepend_x0=prepend_x0)  # Dataset
        self.a = a  # Defining learning rate alpha

    # Defining learning rate alpha
    def setAlpha(self, a):
        self.a = a
        return a

    # Getting learning rate alpha
    def getAlpha(self):
        return self.a

    # Learning best learning rate alpha
    def learnAlpha(self, exponent=4, sample_size=1):

        exponent_counter = 0
        tries = 0
        factor = -1
        steps = self.getAlpha()/10
        changed_direction = False  # Flag

        # Using 70% of the data to guess alpha
        X = self.X
        sample = int(sample_size * self.m)
        self.X = [self.X[i] for i in xrange(sample)]

        # Error rates (previous and current)
        previous_e = self.getErrorRate()
        e = 1
        a = self.getAlpha()

        # Iterating
        while True:

            tries += 1  # Couting tries

            self.gradientDescent()  # Iterating through data
            e = self.getErrorRate()  # Error rate

            # It is getting worst, must decide pivot or finish
            if previous_e < e:

                if not changed_direction:
                    changed_direction = True
                    factor *= -1

                # Already changed direction
                else:
                    changed_direction = False
                    factor *= -1
                    steps = steps/10
                    exponent_counter += 1

            # Max tries
            if tries == 100:
                break

            # Required exponent
            if exponent_counter >= exponent:
                break

            previous_e = e
            a = self.getAlpha() + factor * steps  # New a
            self.setAlpha(a)

        # Original dataset
        self.X = X

        return self.getAlpha()

    # Setting dataset
    def setDataset(self, X=[], Y=[], prepend_x0=True):

        # Insufficient data
        if len(X) == 0:
            print("Error: Insufficient rows into dataset.")
            return False

        # Prepending x0 = 1 into dataset X
        if prepend_x0:
            X = [([1] + x) for x in X]

        self.X = X  # Copying dataset X
        self.m = len(self.X)  # Lenght of X
        self.n = len(self.X[0])  # Number of features in X
        self.Y = Y  # Results into dataset Y
        self.T = [0] * self.n  # Initializing T

        return True

    # Getting dataset
    def getDataset(self):
        return (self.X, self.Y)

    # Calculating error
    def getErrorRate(self):

        # Calculating error based on dataset
        e = p = 0
        for x, y in zip(self.X, self.Y):
            p = self.predict(x, prepend_x0=False)
            e += (p - y)**2  # Error

        return e/self.m

    # Minimizing Theta parameters
    def minimize(self):

        # NE doesn't work weel with large datasets.
        # Better go with gradient descent
        if self.m > 1000:
            return self.gradientDescent()

        return self.normalEquation()

    # Minimizing Theta parameters using gradient descent method
    def gradientDescent(self):

        # Initializing variables
        step = 0  # Preventing infinity loop
        self.T = aux_T = [0] * self.n  # Initializing T and auxiliar
        previous_e = self.getErrorRate()  # Current error rate
        e = 0  # Error rate from next iteration

        # Gradient descent algorithm
        while True:

            # Calculating theta
            for j in xrange(self.n):

                # Calculating new cost function
                derivative = self.derivative(j)

                # Updating auxiliar theta parameter
                aux_T[j] = self.T[j] - self.a * derivative

            # Updating Theta parameters
            self.T = aux_T

            # Getting error rate
            e = self.getErrorRate()

            # If previous error rate becomes smaller than current,
            # we reached the bottom of the curve, then break loop because
            # the algorithm converted
            if previous_e <= e:
                break

            previous_e = e  # Storing previous error rate
            step += 1  # Increasing step counter

            # Warning! Reached max iterations
            if step == 5000:
                break

        # Minimized theta parameters
        return self.T

    # Solving system by normal equation
    # T = INVERSE(TRANSPOSE(X) * X) * TRANSPOSE(X) * Y
    def normalEquation(self):

        # Variables
        matrix_X = np.matrix(self.X)  # X
        transpose_X = matrix_X.transpose()  # TRANSPOSE(X)
        matrix_Y = np.matrix(self.Y).transpose()  # Y

        # A = INVERSE(TRANSPOSE(X) * X)
        A = transpose_X * matrix_X  # TRANSPOSE(X) * X
        inv_A = np.linalg.inv(A)  # INVERSE(A)

        # INV(A) * TRANSPOSE(X) * Y
        B = inv_A * transpose_X * matrix_Y

        # Getting Theta
        self.T = B.transpose().tolist()[0]

        return self.T

    # Calculating hyphotesis H(Xi)
    # Extracting product value
    def hypothesis(self, X):
        """ Calculating linear equation H, where x and T are 1xN matrices """
        return (np.matrix(self.T) * np.matrix(X).transpose()).flat[0]

    # Calculating partial derivative of xj
    def derivative(self, column=0):
        s = 0

        # For each line from dataset
        for line_X, y in zip(self.X, self.Y):

            h = self.hypothesis(line_X)  # Calculating h(x)
            s += (h - y) * line_X[column]  # Numerical derivative

        return s/self.m

    # Predicting result
    def predict(self, x, prepend_x0=True):

        if prepend_x0:
            x = [1] + x  # Prepending x0 = 1 into dataset

        return (np.matrix(self.T) * np.matrix(x).transpose()).flat[0]
