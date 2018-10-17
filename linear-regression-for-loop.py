import matplotlib.pyplot as plt

def compute_cost(y, y_pred):
    cost = 0

    for i in range(len(y)):
        cost += (y[i] - y_pred[i]) ** 2

    cost /= len(y)

    return cost

def linear_regression(m, c, x, y, y_pred, learning_rate):
    temp_m = m
    temp_c = c

    del_m = 0
    for i in range(len(x)):
        del_m += (y[i] - y_pred[i]) * x[i]

    m = temp_m - (2/len(y)) * learning_rate * del_m

    del_c = 0
    for i in range(len(x)):
        del_c += (y[i] - y_pred[i])

    c = temp_c - (2/len(y)) * learning_rate * del_c





    param = {
        'm': m,
        'c': c
    }

    return param


# training set - force applied
x = [4, 5, 7, 9 ,12, 34, 45, 56, 63, 81, 88 ,92, 96, 100, 120, 135, 149, 164, 178, 186, 194]
# how far the ball goes
y = [62, 36, 63, 129, 134, 237, 491, 361, 458, 945, 689, 742, 1020, 890, 1390, 1345, 1578, 1452, 1578,1987, 1893]

n = len(x)
learning_rate = 0.001
epochs = 10

m = 1
c = 1


y_pred = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

for i in range(epochs):
    for j in range(n):
        y_pred[j] = m * x[j] + c;

    print('cost: ', compute_cost(y, y_pred))

    param = linear_regression(m, c, x, y, y_pred, learning_rate)

    m = param['m']
    c = param['c']
