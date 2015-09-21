from linear_regression.LinearRegression import LinearRegression
import matplotlib.pyplot as plt
from timer.timer import Timer


# Main program
def main():

    # Benchmarking program
    t = Timer(verbose=True)

    # Dataset
    X = [[1], [2], [3], [4], [5], [6], [7]]
    Y = [1, 2, 3, 4, 5, 6, 7]

    # Minimizing using normal equation method
    print("> Using Normal Equation")
    lr = LinearRegression(X, Y)
    t.start()
    lr.normalEquation()
    t.finish()
    predicted_Y1 = [lr.predict(x) for x in X]
    print("Error rate: %f" % lr.getErrorRate())

    # Minimizing using gradient descent method
    print("\n> Using Gradient Descent")
    lr = LinearRegression(X, Y)
    t.start()
    lr.setAlpha(0.001)
    lr.gradientDescent()
    t.finish()
    predicted_Y2 = [lr.predict(x) for x in X]
    print("Error rate: %f" % lr.getErrorRate())

    # Plotting example (2d graph)
    plt.plot(X, Y, 'o')  # Dots are original distribution
    plt.plot(X, predicted_Y1, 'r')  # Red line is Normal equation
    plt.plot(X, predicted_Y2, 'g')  # Green line is Gradient descent
    plt.show()

    return 0

if __name__ == "__main__":
    main()
