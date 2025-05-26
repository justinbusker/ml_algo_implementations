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
# split = trial to actually add to y_array
def d_gradient(trials, split, alpha, w, b, x_values, real_y_values):
    y_arrays = []
    costs = []

    for i in range(0,trials):
        temp_w = w - alpha*(calc_der_w(w,b,x_values,real_y_values))
        temp_b = b - alpha*(calc_der_b(w,b,x_values,real_y_values))
        print(f"trial {i}, temp_w: {temp_w}, temp_b: {temp_b}")
        w = temp_w
        b = temp_b
        print(f"trial {i}, w: {w}, b: {b}, cost: {get_squared_cost(x_values, real_y_values, w,b)}")
        if (i % split == 0):
            y_arrays.append(get_y_values(w,b,x_values))
            costs.append(get_squared_cost(x_values, real_y_values, w,b))
    return (y_arrays, costs)

trials = 12000 # number trials model will run
split = 1000 # output to graph every split # of trials


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

regression_vals = d_gradient(trials, split, alpha, w, b, x, house_prices) #runs 15000 trials
print(regression_vals[1][0])
for i in range(len(regression_vals[0])):
    plt.plot(x, regression_vals[0][i], label=f"Trial {(i + 1) * split}, cost: {regression_vals[1][i]:.3f}")

plt.plot(x, y, label="Initial line given", color="red")
plt.xlabel("Age (years)")
plt.ylabel("Price of Unit Area")
plt.legend()
plt.show()
