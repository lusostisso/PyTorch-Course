import numpy as np
import csv
import matplotlib.pyplot as plt

np.random.seed(4)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i]-y_hat == 1:
            # come closer to the line
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            # move away from the line
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b
    
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 10000):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines


def plotBoundaryLines(X, y, boundary_lines):
    X = np.array(X)  
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)

    for line in boundary_lines[:-1]:
        x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        y_vals = line[0] * x_vals + line[1]
        plt.plot(x_vals, y_vals, 'k--', lw=1, color='lightgray', zorder=1)
        
    final_line = boundary_lines[-1]
    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_vals = final_line[0] * x_vals + final_line[1]
    plt.plot(x_vals, y_vals, 'k-', lw=1, zorder=2)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', edgecolors='k', zorder=3)
    plt.show()


X = []
y = []

with open('data.csv', 'r') as arquivo_csv:
    leitor_csv = csv.reader(arquivo_csv)
    for linha in leitor_csv:
        valor1, valor2, valor3 = linha
        X.append([float(valor1), float(valor2)])
        y.append(int(valor3))


boundary_lines = trainPerceptronAlgorithm(np.array(X), np.array(y))    
plotBoundaryLines(X, y, boundary_lines)