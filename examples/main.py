from linear_regression.LinearRegression import LinearRegression
import matplotlib.pyplot as plt

# Dataset
# X = [ [2104, 5, 1, 45], [1416, 3, 2, 40], [1534, 3, 2, 30], [852,  2, 1, 36]]
# Y = [460, 232, 315, 178]

X = [[1], [2], [4], [8], [16], [32], [64]]
Y = [1, 2, 3, 4, 5, 6, 7]


# Main program
def main():

    # Minimizing using normal equation method
    lr = LinearRegression(X, Y)
    lr.normalEquation()
    predicted_Y1 = [lr.predict(x) for x in X]

    # Minimizing using gradient descent method
    lr = LinearRegression(X, Y)
    lr.setAlpha(0.001)
    lr.gradientDescent()
    predicted_Y2 = [lr.predict(x) for x in X]
    print(lr.getErrorRate())

    # Plotting example (2d graph)
    plt.plot(X, Y, 'o')  # Dots are original distribution
    # plt.plot(X, predicted_Y1, 'r')  # Red line is Normal equation
    plt.plot(X, predicted_Y2, 'g')  # Green line is Gradient descent
    plt.show()

    return 0

if __name__ == "__main__":
    main()
