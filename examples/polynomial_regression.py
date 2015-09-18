from linear_regression.LinearRegression import LinearRegression
import matplotlib.pyplot as plt

X = [[1], [2], [3], [4], [5], [6], [7]]
Y = [1, 2, 4, 8, 16, 32, 64]


# Main program
def main():

    # Modeling dataset by adding x2=x1^2 and x3=x1^3
    # so that y(x) = ax1^3 + bx1^2 + cx1 + d (polynomial curve)
    X2 = [x + [x[0]**2, x[0]**3] for x in X]

    # Minimizing using normal equation method
    lr = LinearRegression(X2, Y)
    lr.normalEquation()
    predicted_Y1 = [lr.predict(x) for x in X2]

    # Minimizing using gradient descent method
    lr = LinearRegression(X2, Y)
    lr.setAlpha(0.000001)
    lr.gradientDescent()
    predicted_Y2 = [lr.predict(x) for x in X2]

    # Plotting example (2d graph)
    plt.plot(X, Y, 'o')  # Dots are original distribution
    plt.plot(X, predicted_Y1, 'r')  # Red line is Gradient descent
    plt.plot(X, predicted_Y2, 'g')  # Green line is Gradient descent
    plt.show()

    return 0

if __name__ == "__main__":
    main()
