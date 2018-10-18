import numpy as np
import matplotlib.pyplot as plt




# training set - force applied
x = np.array([[4], [5], [7], [9] ,[12], [34], [45], [56], [63], [81], [88] ,[92], [96], [100], [120], [135], [149], [164], [178], [186], [194]])
# how far the ball goes
y = np.array([[62], [36], [63], [129], [134], [237], [491], [361], [458], [945], [689], [742], [1020], [890], [1390], [1345], [1578], [1452], [1578],[1987], [1893]])

x_train = np.append(x, np.ones(x.shape), axis = 1)

# y = mx + c
# parameters = m, c
# start with the random values of m, class
n = len(y)
learning_rate = 0.0001
epochs = 6

theta = np.array([[0.0,0.0]])

# linear regression
for i in range(epochs):

    y_pred = x_train @ theta.T

    cost = np.sum(np.square(y_pred - y)) / (2 * len(y))
    print('cost: ', cost)
    # regression step
    theta[0] -= (1/len(y)) * learning_rate * np.sum((y_pred - y) * x_train, axis = 0)


# Plot
plt.scatter(x, y)
plt.plot(x, y_pred, 'r')
plt.title('Scatter plot of Prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
