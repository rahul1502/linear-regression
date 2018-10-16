import numpy as np
import matplotlib.pyplot as plt


def compute_cost(y, y_pred):
    cost = (1/len(y)) * np.sum(np.square(y_pred - y))
    return cost

def linear_regression(m, c, x, y, y_pred, learning_rate):
    new_m = m - (2/len(y)) * learning_rate * np.sum(np.multiply(y-y_pred, x))
    new_c = c - (2/len(y)) * learning_rate * np.sum(y-y_pred)

    param = {
        'm': new_m,
        'c': new_c
    }

    return param


# training set - force applied
x = np.array([[4], [5], [7], [9] ,[12], [34], [45], [56], [63], [81], [88] ,[92], [96], [100], [120], [135], [149], [164], [178], [186], [194]])
# how far the ball goes
y = np.array([[62], [36], [63], [129], [134], [237], [491], [361], [458], [945], [689], [742], [1020], [890], [1390], [1345], [1578], [1452], [1578],[1987], [1893]])

# # Plot
# plt.scatter(x, y)
# plt.title('Scatter plot of the Training Data')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

# y = mx + c
# parameters = m, c
# start with the random values of m, class
n = len(x)
learning_rate = 0.001
epochs = 100

m = 0
c = 0

for i in range(epochs):

    y_pred = m * x + c
    print('cost: ', compute_cost(y, y_pred))
    param = linear_regression(m, c, x, y, y_pred, learning_rate)

    m = param['m']
    c = param['c']
