from models.LinearRegression import LinearRegression

# Dataset
X = [
    [3, 1],
    [6, 2],
    [9, 3],
    [12, 4],
    [15, 5],
    [18, 6],
    [21, 7],
    [24, 8],
    [27, 9],
    [30, 10]
]

# Observed results
Y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# Parameters
T = [0, 0]


# Main program
def main():

    # Model
    lr = LinearRegression(X, Y)
    lr.minimize()
    print(lr.getErrorRate())
    print(lr.predict([3, 1]))

    # Optimizing learning rate
    lr.learnAlpha()
    lr.minimize()
    print(lr.getErrorRate())
    print(lr.predict([3, 1]))

    return 0

if __name__ == "__main__":
    main()
