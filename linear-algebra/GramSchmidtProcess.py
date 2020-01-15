# coding: utf-8
import numpy as np
import numpy.linalg as la

VERY_SMALL_NUMBER = 1e-14

def gsBasis4(A):
    B = np.array(A, dtype=np.float_)
    B[:, 0] = B[:, 0] / la.norm(B[:, 0])
    B[:, 1] = B[:, 1] - B[:, 1] @ B[:, 0] * B[:, 0]

    if la.norm(B[:, 1]) > VERY_SMALL_NUMBER:
        B[:, 1] = B[:, 1] / la.norm(B[:, 1])
    else:
        B[:, 1] = np.zeros_like(B[:, 1])

    B[:, 2] = B[:, 2] - B[:, 2] @ B[:, 0] * B[:, 0]
    B[:, 2] = B[:, 2] - B[:, 2] @ B[:, 1] * B[:, 1]

    if la.norm(B[:, 2]) > VERY_SMALL_NUMBER:
        B[:, 2] = B[:, 2] / la.norm(B[:, 2])
    else:
        B[:, 2] = np.zeros_like(B[:, 2])

    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 0] * B[:, 0]
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 1] * B[:, 1]
    B[:, 3] = B[:, 3] - B[:, 3] @ B[:, 2] * B[:, 2]

    if la.norm(B[:, 3]) > VERY_SMALL_NUMBER:
        B[:, 3] = B[:, 3] / la.norm(B[:, 3])
    else:
        B[:, 3] = np.zeros_like(B[:, 3])
    return B

def gsBasis(A):
    B = np.array(A, dtype=np.float_)
    for i in range(B.shape[1]):
        for j in range(i):
            B[:, i] = B[:, i] - B[:, i] @ B[:, j] * B[:, j]

        if la.norm(B[:, i]) > VERY_SMALL_NUMBER:
            B[:, i] = B[:, i] / la.norm(B[:, i])
        else:
            B[:, i] = np.zeros_like(B[:, i])

    return B

def dimensions(A):
    return np.sum(la.norm(gsBasis(A), axis=0))

if __name__ == '__main__':
    V = np.array([[1, 0, 2, 6],
                  [0, 1, 8, 2],
                  [2, 8, 3, 1],
                  [1, -6, 2, 3]], dtype=np.float_)
    print(f"Using gsBasis4")
    print(gsBasis4(V))
    print()

    U = gsBasis4(V)
    print(f"Repeat using gsBasis4")
    print(gsBasis4(U))
    print()

    print(f"Now using gsBasis")
    print(gsBasis(V))
    print()

    A = np.array([[3, 2, 3],
                  [2, 5, -1],
                  [2, 4, 8],
                  [12, 2, 1]], dtype=np.float_)
    print(gsBasis(A))

    print(dimensions(A))

    B = np.array([[6, 2, 1, 7, 5],
                  [2, 8, 5, -4, 1],
                  [1, -6, 3, 2, 8]], dtype=np.float_)
    print(gsBasis(B))

    print(dimensions(B))

    C = np.array([[1, 0, 2],
                  [0, 1, -3],
                  [1, 0, 2]], dtype=np.float_)
    print(gsBasis(C))

    print(dimensions(C))
