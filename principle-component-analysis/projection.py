""" Project 3D data onto a 2D subspace """


import numpy as np


def concatenate_vectors(*vectors):
    """
        Concatenate vectors into a matrix
    """
    return np.column_stack((vectors))


def projection_matrix(*vectors):
    """
        Determine the projection distance_matrix P = B @ np.linalg.inv(B.T @ B) @ B.T
    """
    B = concatenate_vectors(*vectors)
    return B @ np.linalg.inv(B.T @ B) @ B.T


def projection(x, P):
    """
        Projection of vector x using projection matrix P
    """
    return P @ x


def lambda_vector(B, x):
    """
        Determine lambda
    """
    return np.linalg.inv(B.T @ B) @ B.T @ x


x = np.array([6, 0, 0])
b1 = np.array([1, 1, 1])
b2 = np.array([0, 1, 2])

y = np.array([3, 2, 2])
c1 = np.array([1, 0, 0])
c2 = np.array([0, 1, 1])

z = np.array([12, 0, 0])
d1 = np.array([1, 1, 1])
d2 = np.array([0, 1, 2])

B1 = concatenate_vectors(b1, b2)

P1 = projection_matrix(b1, b2)
P2 = projection_matrix(c1, c2)
P3 = projection_matrix(d1, d2)
# pi_x = projection(x, P)
# print(pi_x)
# lambda_vector = lambda_vector(B, x)
# print(lambda_vector)

print(f"Question 1: projection matrix =\n{P1}")
print(f"lambda vector = {lambda_vector(B1, x)}")
print(f"Question 1: projection onto subspace U = {projection(x, P1)}")
print(f"Question 2: projection onto subspace U = {projection(y, P2)}")
print(f"Question 3: projection onto subspace U = {projection(z, P3)}")
