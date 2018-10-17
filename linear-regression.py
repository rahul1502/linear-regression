import numpy as np
import matplotlib.pyplot as plt


def compute_cost(y, y_pred):
    cost = np.sum(np.square(y_pred - y)) / (2 * len(y))
    return cost

def linear_regression(theta, x_train, y, y_pred, learning_rate):
    print(theta[0])
    print((1/len(y)) * learning_rate * np.sum((y_pred - y) * x_train, axis = 0))

    theta[0] -= (1/len(y)) * learning_rate * np.sum((y_pred - y) * x_train, axis = 0)

    return theta



# training set - force applied
x = np.array([[4], [5], [7], [9] ,[12], [34], [45], [56], [63], [81], [88] ,[92], [96], [100], [120], [135], [149], [164], [178], [186], [194]])
# how far the ball goes
y = np.array([[62], [36], [63], [129], [134], [237], [491], [361], [458], [945], [689], [742], [1020], [890], [1390], [1345], [1578], [1452], [1578],[1987], [1893]])

x_train = np.append(x, np.ones(x.shape), axis = 1)

# x = (x - x.mean())/x.std()
# y = (y - y.mean())/y.std()

# # Plot
# plt.scatter(x, y)
# plt.title('Scatter plot of the Training Data')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

# y = mx + c
# parameters = m, c
# start with the random values of m, class
n = len(y)
learning_rate = 0.0001
epochs = 10

theta = np.array([[1,1]])


for i in range(epochs):

    y_pred = x_train @ theta.T

    print('cost: ', compute_cost(y, y_pred))
    theta = linear_regression(theta, x_train, y, y_pred, learning_rate)
