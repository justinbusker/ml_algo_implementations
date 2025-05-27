import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

df = pd.read_csv("Real estate.csv")
X = df.iloc[:, :-1].to_numpy()
y = np.array(df.house_price)

iterations = 1000
alpha = 0.0000001
w = np.ones(X.shape[1])
b = 1

def predict_value(X, w, b):
    return np.dot(X, w) + b

def compute_cost(X, w, b, y):
    m = len(X)
    total_cost = 0
    for i in range(m):
        total_cost += ((np.dot(X[i], w) + b) - y[i])**2
    total_cost = total_cost / (2*m)
    print(total_cost)
    return total_cost

def der_w_b(X, w, b, y, alpha):
    der_b = 0
    m,n = X.shape 
    der_w = np.zeros((n,))

    for j in range(m):
        error = (np.dot(X[j], w) + b) - y[j]
        for i in range(n):
            der_w[i] += error*X[j][i]
        der_b += error
    der_w = der_w/m
    der_b = der_b/m
    return der_w, der_b

def gradient_descent(X, w, b, y, alpha, iterations):
    x = np.arange(0,iterations)
    costs = []
    for i in range(iterations):
        der_w, der_b = der_w_b(X,w,b,y,alpha)
        temp_w = w - alpha*der_w
        temp_b = b - alpha*der_b
        w = temp_w
        b = temp_b
        costs.append(compute_cost(X,w,b,y))
    
    return costs

def scale_features(X):
    m,n = X.shape
    rescaled_X = X.copy()

    for i in range(n-1):
        rescaled_X[:, i] = rescaled_X[:, i] / np.max(rescaled_X[:, i])
    return rescaled_X

scale_X = scale_features(X)
scaled_costs = gradient_descent(scale_X, w, b, y, alpha, iterations)
def_costs = gradient_descent(X, w, b, y, alpha, iterations)

x = np.arange(0,iterations)

fig, axs = plt.subplots(1,2)

plt.xlabel("Iterations")
plt.ylabel("Cost")
axs[0].plot( x,def_costs)
axs[0].set_title("Defualt Scaling Descent")
axs[1].plot( x,scaled_costs)
axs[1].set_title("Scaled Descent")
plt.show()

