import numpy as np
import matplotlib.pyplot as plt
import csv

# get the training data
data = csv.reader(open('data/train.csv', newline=''), delimiter=' ', quotechar='|')
x_train = np.empty((0,2), float)
y_train = np.empty((0,1), float)
for row in data:
    x_train = np.append(x_train, np.array([[float(row[0].split(',')[0]), 1.0]]), axis = 0)
    y_train = np.append(y_train, np.array([[float(row[0].split(',')[1])]]), axis = 0)
# get the testing data
data = csv.reader(open('data/test.csv', newline=''), delimiter=' ', quotechar='|')
x_test = np.empty((0,2), float)
y_test = np.empty((0,1), float)
for row in data:
    x_test = np.append(x_test, np.array([[float(row[0].split(',')[0]),  1.0]]), axis = 0)
    y_test = np.append(y_test, np.array([[float(row[0].split(',')[1])]]), axis = 0)

# training
print('Training with ' + str(len(x_train)) + ' tuples')

# y = mx + c
# parameters = m, c
# start with the random values of m, class
n = len(y_train)
learning_rate = 0.0001
epochs = 20

# theta = [[m, c]]
theta = np.array([[0.0,0.0]])

# linear regression
for i in range(epochs):
    y_pred = x_train @ theta.T

    error = (1 / (1 * n)) * np.sum(np.square(y_pred - y_train))
    print('error: ', error)

    # regression step
    # theta is (1,2) matrix so get 0th row using theta[0]
    theta[0] -= (1/n) * learning_rate * np.sum((y_pred - y_train) * x_train, axis = 0)

# Plot
plt.scatter(x_train[:,0], y_train)
plt.plot(x_train[:,0], y_pred, 'r')
plt.title('Training')
plt.xlabel('x_train')
plt.ylabel('y_train')
plt.show()

print('---------------------------------------------')
# testing
print('Testing with ' + str(len(x_test)) + ' tuples')
y_test_pred = x_test @ theta.T
plt.scatter(x_test[:,0], y_test)
plt.plot(x_train[:,0], y_pred, 'r')
plt.title('Testing')
plt.xlabel('x_test')
plt.ylabel('y_test')
plt.show()
