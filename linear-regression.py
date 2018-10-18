import numpy as np
import matplotlib.pyplot as plt
import csv

data = csv.reader(open('data/train.csv', newline=''), delimiter=' ', quotechar='|')

x = np.empty((0,1), float)
y = np.empty((0,1), float)


for row in data:

    x = np.append(x, np.array([[float(row[0].split(',')[0])]]), axis = 0)
    y = np.append(y, np.array([[float(row[0].split(',')[1])]]), axis = 0)


x_train = np.append(x, np.ones(x.shape), axis = 1)

# y = mx + c
# parameters = m, c
# start with the random values of m, class
n = len(y)
learning_rate = 0.0001
epochs = 20

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
