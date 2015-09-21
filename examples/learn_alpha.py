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

    # Minimizing using gradient descent method
    lr = LinearRegression(X, Y, 0.03)

    # Learning alpha
    t.start()
    print("> Learning alpha.. "),
    lr.learnAlpha()
    print("done.")
    t.finish()

    # Minimizing theta
    t.start()
    lr.gradientDescent()
    t.finish()

    # Error rate
    print("\n> Using Gradient Descent")
    print("Error rate: %f" % lr.getErrorRate())

    # Plotting example (2d graph)
    predicted_Y2 = [lr.predict(x) for x in X]
    plt.plot(X, Y, 'o')  # Dots are original distribution
    plt.plot(X, predicted_Y2, 'g')  # Green line is Gradient descent
    plt.show()

    return 0

if __name__ == "__main__":
    main()
