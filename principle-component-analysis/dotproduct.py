import math

import numpy as np


def length(vector):
    return np.linalg.norm(vector)

def distance(vector1, vector2):
    return length(vector1 - vector2)

def angle(vector1, vector2):
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
        [1],
        [-1],
    ]
)


d = np.array(
    [
        [1],
        [2],
        [3],
    ]
)

e = np.array(
    [
        [-1],
        [0],
        [8],
    ]
)

print(f"Question 1: {length(a)}, which is the square root of {length(a) ** 2}")
print(f"Question 2: {angle(b, c)}")
print(f"Question 3: {distance(b, c)}")
print(f"Question 5: {angle(d, d - e)}")
