""" inner product functions """

import math

import numpy as np


def innerproduct_definition(x, y):
    """
        vectors x and y are elements of vector space V
        inner product <x, y>: V x V -> R

        - symmetric:
            <x, y> = <y, x>
        - positive definite:
            <x, x> > 0, <x, x> = 0 <==> x = 0
        - bilinear:
            x, y, z e V, lamda e R
            <lambda * x + z, y             > = lambda * <x, y> + <z, y>
            <x             , lambda * y + z> = lambda * <x, y> + <x, z>

        Example in R2:
        <x, y> = x.T @ I @ y = 1 * x[0]y[0] + 0 * x[1]y[0] + 0 * x[0]y[1] + 1 * x[1]y[1] = x[0]y[0] + x[1]y[1] <- dot product
        <x, y> = x.T @ A @ y = a00 * x[0]y[0] + a10 * x[1]y[0] + a01 * x[0]y[1] + a11 * x[1]y[1]               <- inner product

    """
    pass


def inner(vector1, vector2, A=None):
    """
       Inner product of x and y = <x, y>

       where
           <x, y> = x.T @ y = x.T @ I @ y
       or
           <x, y> = x.T @ A @ y
    """
    A = A if A is not None else np.identity(vector1.size)

    return vector1.T @ A @ vector2


def length(vector, A=None):
    """
        ||x|| = sqrt(<x, x>)

        where
            <x, x> = x.T @ x = x.T @ I @ x
        or
            <x, x> = x.T @ A @ x
    """
    A = A if A is not None else np.identity(vector.size)

    return math.sqrt(vector.T @ A @ vector)


def distance(vector1, vector2, A=None):
    """
        d(x, y) = ||x - y|| = sqrt(<(x - y), (x - y)>)
    """
    vector = vector1 - vector2
    A = A if A is not None else np.identity(vector.size)

    return math.sqrt(vector.T @ A @ vector)


def angle(vector1, vector2, A=None):
    """
        cos α = <x, y> / ||x|| ||y||
    """
    A = A if A is not None else np.identity(vector1.size)

    return math.acos((vector1.T @ A @ vector2 / (length(vector1, A) * length(vector2, A))))


def determine_matrix(A, vector):
    return A if A is not None else np.identity(vector.size)


x = np.array(
    [
        [-2],
        [2],
    ]
)

y = np.array(
    [
        [2],
        [3],
    ]
)

z = np.array(
    [
        [4],
        [1],
    ]
)

A = np.array(
    [
        [1, -0.5],
        [-0.5, 1],
    ]
)

xq1 = np.array(
    [
        [1],
        [-1],
        [3],
    ]
)

Aq1 = np.array(
    [
        [2, 1, 0],
        [1, 2, -1],
        [0, -1, 2],
    ]
)

xq2 = np.array(
    [
        [0.5],
        [-1],
        [-0.5],
    ]
)

yq2 = np.array(
    [
        [0],
        [1],
        [0],
    ]
)

xq3 = np.array(
    [
        [-1],
        [1],
    ]
)

Aq3 = np.array(
    [
        [5, -1],
        [-1, 5],
    ]
)


xq4 = np.array(
    [
        [4],
        [2],
        [1],
    ]
)

yq4 = np.array(
    [
        [0],
        [1],
        [0],
    ]
)

Aq4 = np.array(
    [
        [2, 1, 0],
        [1, 2, -1],
        [0, -1, 2],
    ]
)

xq5 = np.array(
    [
        [-1],
        [-1],
        [-1],
    ]
)

aq1 = np.array(
    [
        [1],
        [1],
    ]
)

bq1 = np.array(
    [
        [-1],
        [1],
    ]
)

Mq1 = np.array(
    [
        [2, -1],
        [-1, 4],
    ]
)

aq2 = np.array(
    [
        [0],
        [-1],
    ]
)

bq2 = np.array(
    [
        [1],
        [1],
    ]
)

Mq2 = np.array(
    [
        [1, -0.5],
        [-0.5, 5],
    ]
)

aq3 = np.array(
    [
        [2],
        [2],
    ]
)

bq3 = np.array(
    [
        [-2],
        [-2],
    ]
)

Mq3 = np.array(
    [
        [2, 1],
        [1, 4],
    ]
)

aq4 = np.array(
    [
        [1],
        [1],
    ]
)

bq4 = np.array(
    [
        [1],
        [-1],
    ]
)

Mq4 = np.array(
    [
        [1, 0],
        [0, 5],
    ]
)

aq5 = np.array(
    [
        [1],
        [1],
        [1],
    ]
)

bq5 = np.array(
    [
        [2],
        [-1],
        [0],
    ]
)

Mq5 = np.array(
    [
        [1, 0, 0],
        [0, 2, -1],
        [0, -1, 3],
    ]
)

print(f"*** Inner Product: Lengths and Distances ***")
print(f"Q1: Length = {length(xq1, Aq1)}, which is the square root of {length(xq1, Aq1) ** 2}." )
print(f"Q2: Distance = {distance(xq2, yq2, Aq1)}, which is the square root of {distance(xq2, yq2, Aq1) ** 2}." )
print(f"Q3: Length = {length(xq3, Aq3)}, which is the square root of {length(xq3, Aq3) ** 2}." )
print(f"Q4: Distance = {distance(xq4, yq4, Aq4)}, which is the square root of {distance(xq4, yq4, Aq4) ** 2}." )
print(f"Q5: Length = {length(xq5)}, which is the square root of {length(xq5) ** 2}." )
print()
print(f"*** Inner Product: Angles between vectors using a non-standard inner product ***")
print(f"Q1: Angle = {angle(aq1, bq1, Mq1)} radians")
print(f"Q2: Angle = {angle(aq2, bq2, Mq2)} radians")
print(f"Q3: Angle = {angle(aq3, bq3, Mq3)} radians")
print(f"Q4: Angle = {angle(aq4, bq4, Mq4)} radians")
print(f"Q5: Angle = {angle(aq5, bq5, Mq5)} radians")
# print(f"Length = {length(x)}, which is the square root of {length(x) ** 2}." )
# print(f"Length = {length(x, A)}, which is the square root of {length(x, A) ** 2}." )
# print(f"Distance = {distance(y, z)}, which is the square root of {distance(y, z) ** 2}." )
# print(f"Distance = {distance(y, z, A)}, which is the square root of {distance(y, z, A) ** 2}." )
# print(f"Angle = {angle(y, z)} radians.")
# print(f"Angle = {angle(y, z, A)} radians.")

