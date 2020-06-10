""" dot product functions """

import math

import numpy as np


def dotproduct_definition(vector1, vector2):
    """
        the dot product of vectors x and y is defined as x.T @ y = sum(x[i] * y[i]) for i = 1 to N
    """
    dotproduct = 0

    for element in zip(vector1, vector2):
        dotproduct += int(element[0]) * int(element[0])

    return dotproduct


def dot(vector1, vector2):
    return vector1 @ vector2


def length(vector):
    """
        length of vector x is the square root of the dot product of x with itself
        ||x|| = sqrt(x.T @ x)
    """
    return np.linalg.norm(vector)


def distance(vector1, vector2):
    """
        distance(x, y) = ||x - y|| = sqrt((x - y).T @ (x - y))
    """
    return length(vector1 - vector2)


def angle(vector1, vector2):
    """
        angle between vextors x and y
        cos alpha = x.T @ y / ||x|| *  ||y||
    """
    return math.acos(float(vector1.T @ vector2 / (length(vector1) * length(vector2))))


a = np.array(
    [
        [1],
        [-1],
        [3],
    ]
)

b = np.array(
    [
        [3],
        [4],
    ]
)

c = np.array(
    [
        [-1],
        [-1],
    ]
)


d = np.array(
    [
        [1],
        [1],
    ]
)


e = np.array(
    [
        [1],
        [2],
        [3],
    ]
)

f = np.array(
    [
        [-1],
        [0],
        [8],
    ]
)

print(a.T @ a)
print(np.dot(a.T, a))
print(dot(a.T, a))
print(dotproduct_definition(a, a))
# print(f"Question 1: {length(a)}, which is the square root of {length(a) ** 2}")
# print(f"Question 2: {angle(b, c)}")
# print(f"Question 3: {distance(b, d)}")
# print(f"Question 5: {angle(e, e - f)}")
