import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import kagglehub
from kagglehub import KaggleDatasetAdapter

# returns square cost
def get_squared_cost(x_values, real_y_values, w, b):
    cost = 0
    for i in range(0, len(x_values)):
        est_y_val = w*x_values[i] + b
        cost += (est_y_val - real_y_values[i])**2
    cost = cost/(2*len(x_values))
    return cost

#calculates partial derivate of J(w,b) in terms of w
def calc_der_w(w,b,x_values, real_y_values):
    sum = 0
    for i in range(0, len(x_values)):
        est_y_val = w*x_values[i] + b
        sum += x_values[i]*(est_y_val - real_y_values[i])
    sum = sum / len(x_values)
    return sum

#calculates partial derivate of J(w,b) in terms of b
def calc_der_b(w,b,x_values, real_y_values):
    sum = 0
    for i in range(0, len(x_values)):
        est_y_val = w*x_values[i] + b
        sum += (est_y_val - real_y_values[i])
    sum = sum / len(x_values)
    return sum

# gets all y values for a given w,b
def get_y_values(w,b, x_values):
    y = []

    for i in range(len(x_values)):
        y.append(x_values[i]*w + b)
    return y

# main function to run gradient descent
def d_gradient(trials, alpha, w, b, x_values, real_y_values):
    y_arrays = []
    y_arrays.append(get_y_values(w,b,x_values))

    for i in range(0,trials):
        temp_w = w - alpha*(calc_der_w(w,b,x_values,real_y_values))
        temp_b = b - alpha*(calc_der_b(w,b,x_values,real_y_values))
        print(f"trial {i}, temp_w: {temp_w}, temp_b: {temp_b}")
        w = temp_w
        b = temp_b
        print(f"trial {i}, w: {w}, b: {b}, cost: {get_squared_cost(x_values, real_y_values, w,b)}")
        y_arrays.append(get_y_values(w,b,x_values))
    return y_arrays

alpha = 0.001

w = 1
b = 20

cost = 0

# get x and y data
df = pd.read_csv("Real estate.csv")
x = df.house_age
house_prices = np.array(df.house_price)

# plot original house age and price data
plt.scatter(df.house_age, df.house_price, label="House cost")

x = df.house_age
house_prices = np.array(df.house_price)
y = []

for i in x:
    y.append(i*w + b)

regression_vals = d_gradient(15000, alpha, w, b, x, house_prices) #runs 15000 trials
for i in range(len(regression_vals)):
    if(i % 1000 == 0 ):
        plt.plot(x, regression_vals[i], label=f"Trial {i}")

plt.plot(x, y, label="test of regression", color="red")
plt.xlabel("Age (years)")
plt.ylabel("Cost")
plt.legend()
plt.show()
