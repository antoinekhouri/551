# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt


def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df

# Activation function used to map any real value between 0 and 1
def sigmoid(x):
    return 1/ (1+np.exp(-x))

# Computes weighted sum of inputs
def net_input(theta, x):
    return np.dot(x, theta)

# Computes probability after passing through sigmoid
def probability(theta, x):
    return sigmoid(net_input(theta, x))

# Computes cost function for training samples
def cost_function(theta, x, y):
    m = x.shape[0] # Count of data
    # Cost function (math)
    total_cost = -(1 / m) * np.sum(
        y * np.log(probability(theta, x)) + (1 - y) * np.log(
            1 - probability(theta, x)))
    return total_cost

# Computes gradient cost of function at point theta
def gradient(theta, x, y):
    m=x.shape[0] # Count of data
    return (1/m) * np.dot(x.T, sigmoid(net_input(theta, x))-y) # Gradient function for logistic regression

# Fit function, finds the model paramaters that minimize the cost function
def fit(x, y, theta):
    opt_weights = opt.fmin_tnc(func = cost_function, x0=theta, fprime=gradient, args=(x,y.flatten())) # This computes the minimum of cost_function                                                                                                  
    return opt_weights[0]

# --------------- ACCURACY MODEL --------------

def predict(x):
    theta = parameters[:, np.newaxis]
    return probability(theta, x)

def accuracy(x, actual_classes, probab_threshold=0.5):
    predicted_classes = (predict(x) >= 
                         probab_threshold).astype(int)
    predicted_classes = predicted_classes.flatten()
    accuracy = np.mean(predicted_classes == actual_classes)
    return accuracy * 100

if __name__ == "__main__":
    # load the data from the file
    data = load_data("data/marks.txt", None)

    # X = feature values, all the columns except the last column
    X = data.iloc[:, :-1]

    # y = target values, last column of the data frame
    y = data.iloc[:, -1]

    # filter out the applicants that got admitted
    admitted = data.loc[y == 1]

    # filter out the applicants that din't get admission
    not_admitted = data.loc[y == 0]

    # Prepare data

    X = np.c_[np.ones((X.shape[0], 1)), X] #Transform input into array
    y = y[:, np.newaxis] #Transform y into 2D array
    theta = np.zeros((X.shape[1],1)) # initialize
    parameters = fit(X, y, theta)

    # accuracy

    acc = accuracy(X, y.flatten())
    print("Accuracy:")
    print(acc)

    # plots
    plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
    plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
    # plt.legend()
    # Decision boundary plotting
    x_values = [np.min(X[:, 1] - 5), np.max(X[:, 2] + 5)]
    y_values = - (parameters[0] + np.dot(parameters[1], x_values)) / parameters[2]
    plt.plot(x_values, y_values, label='Decision Boundary')
    plt.xlabel('Marks in 1st Exam')
    plt.ylabel('Marks in 2nd Exam')
    plt.legend()
    plt.show()








