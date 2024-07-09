import numpy as np

dvalues = np.array([[1., 1., 1.]])

weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

dx0 = sum([weights[0][0] * dvalues[0][0], weights[0][1] * dvalues[0][1],
           weights[0][2] * dvalues[0][2]])
dx1 = sum([weights[1][0] * dvalues[0][0], weights[1][1] * dvalues[0][1],
           weights[1][2] * dvalues[0][2]])
dx2 = sum([weights[2][0] * dvalues[0][0], weights[2][1] * dvalues[0][1],
           weights[2][2] * dvalues[0][2]])
dx3 = sum([weights[3][0] * dvalues[0][0], weights[3][1] * dvalues[0][1],
           weights[3][2] * dvalues[0][2]])
dinputs = np.array([dx0, dx1, dx2, dx3])
print(dinputs)
