import numpy as np
from sklearn.decomposition import PCA as SKPCA


def mean(dataset):
    return dataset.mean(axis=0)


def center_dataset(dataset):
    mu = mean(dataset)
    return dataset - mu


def covariance_matrix(dataset):
    centered_dataset = center_dataset(dataset)
    dimensionless_dataset = divide_by_standard_deviation(centered_dataset)
    return np.cov(dimensionless_dataset, bias=True)


def divide_by_standard_deviation(dataset):
    standard_deviation = np.std(dataset, axis=0)
    return dataset / standard_deviation


def normalize(X):
    """Normalize the given dataset X
    Args:
        X: ndarray, dataset

    Returns:
        (Xbar, mean, std): tuple of ndarray, Xbar is the normalized dataset
        with mean 0 and standard deviation 1; mean and std are the
        mean and standard deviation respectively.

    Note:
        You will encounter dimensions where the standard deviation is
        zero, for those when you do normalization the normalized data
        will be NaN. Handle this by setting using `std = 1` for those
        dimensions when doing normalization.
    """
    mu = X.mean(axis=0)
    std = np.std(X, axis=0)
    std_filled = std.copy()
    std_filled[std == 0] = 1.
    Xbar = (X - mu) / std_filled

    return Xbar, mu, std


def eig(S):
    """Compute the eigenvalues and corresponding eigenvectors
        for the covariance matrix S.
    Args:
        S: ndarray, covariance matrix

    Returns:
        (eigvals, eigvecs): ndarray, the eigenvalues and eigenvectors

    Note:
        the eigenvals and eigenvecs should be sorted in descending
        order of the eigen values
    """

    eigenvalues, eigenvectors = np.linalg.eig(S)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors  # <-- EDIT THIS to return eigenvalues and corresp eigenvectors


def projection_matrix(B):
    """Compute the projection matrix onto the space spanned by `B`
    Args:
        B: ndarray of dimension (D, M), the basis for the subspace

    Returns:
        P: the projection matrix
    """
    # return np.eye(B.shape[0]) # <-- EDIT THIS to compute the projection matrix
    return B @ B.T  # <-- EDIT THIS to compute the projection matrix


def PCA(X, num_components):
    """
    Args:
        X: ndarray of size (N, D), where D is the dimension of the data,
           and N is the number of datapoints
        num_components: the number of principal components to use.
    Returns:
        X_reconstruct: ndarray of the reconstruction
        of X from the first `num_components` principal components.
    """
    # *** STEP 1, 2 NORMALIZE DATA ***
    normalized_X = normalize(X)
    Xbar = normalized_X[0]

    # *** STEP 3 FIND DATA COV MATRIX S AND ITS EIGENVALUES, EIGENVECTORS
    S = np.cov(Xbar, rowvar=False)
    eigenvalues, eigenvectors = eig(S)
    basis = eigenvectors[:, :num_components]

    # *** STEP 4 project data point x_star onto pricipal subspace
    P = projection_matrix(basis)

    X_reconstr = (P @ Xbar.T).T

    return X_reconstr


def PCA_high_dim(X, n_components):
    """Compute PCA for small sample size but high-dimensional features. 
    Args:
        X: ndarray of size (N, D), where D is the dimension of the sample,
           and N is the number of samples
        num_components: the number of principal components to use.
    Returns:
        X_reconstruct: (N, D) ndarray. the reconstruction
        of X from the first `num_components` pricipal components.
    """
    # normalized_X = normalize(X)
    # Xbar = normalized_X[0]
    # mu = normalized_X[1]
    # std = normalized_X[2]
    N, D = X.shape
    S = (X @ X.T) / N
    eigenvalues, eigenvectors = eig(S)
    basis = eigenvectors[:, :n_components]
    P = projection_matrix(basis)
    X_reconstr = (P @ Xbar.T).T
    return X_reconstr # <-- EDIT THIS to return the reconstruction of X

X = np.array([[1, 2], [1, 3], [2, 3]])
print(f"X: \n{X}")
Xbar = normalize(X)[0]
mu = normalize(X)[1]
std = normalize(X)[2]
print(f"Xbar: \n{Xbar}, \nmu: \n{mu}, \nstd: \n{std}")
print(f"X: \n{X}")
num_component = 2
pca = SKPCA(n_components=num_component, svd_solver='full')
sklearn_reconst = pca.inverse_transform(pca.fit_transform(Xbar))
print(f"SKLearn reconst: \n{sklearn_reconst}")
reconst = PCA(Xbar, num_component)
print(f"Reconst: \n {reconst}")
print(np.isclose(sklearn_reconst, reconst))
reconst_hidim = PCA_high_dim(Xbar, num_component)
print(f"Reconst hign dimensions: \n {reconst_hidim}")
print(np.isclose(sklearn_reconst, reconst_hidim))
# vector1 = np.array([1, 2, 3])
# vector2 = np.array([4, 5, 6])
# matrix1 = np .column_stack((vector1, vector2))
# print(matrix1)
# rows, columns = matrix1.shape
# print(f"Matrix1 has {rows} rows and {columns} columns")

# datapoint1 = np.array([1, 2])
# datapoint2 = np.array([5, 4])
# dataset1 = np.column_stack((datapoint1, datapoint2))
# print(dataset1)
# print(dataset1.mean(), dataset1.mean(axis=0), dataset1.mean(axis=1))
# print(mean(dataset1))
# print(center_dataset(dataset1))
# print(datapoint1 - mean(dataset1))
# print(datapoint2 - mean(dataset0))
# print(center_datapoint(datapoint1, dataset1))
# print(covariance_element(datapoint1))
# print(np.cov(dataset1, bias=True))
# print(center_dataset(dataset1))
# print(np.std(center_dataset(dataset1), axis=0))
# print(np.cov(center_dataset(dataset1), bias=True))
# print(covariance_matrix(dataset1))
# prepped_matrix = divide_by_standard_deviation(center_dataset(dataset1))
# print(prepped_matrix)
# print(np.var(prepped_matrix, axis=0))
# print(np.cov(prepped_matrix, bias=True))
# datapoint1 = np.array([0, 1000])
# datapoint2 = np.array([1, 1001])
# datapoint3 = np.array([2, 999])
# datapoint4 = np.array([4, 1002])
# datapoint5 = np.array([6, 998])
# dataset1 = np.column_stack((datapoint1, datapoint2, datapoint3, datapoint4, datapoint5))
# covariance_matrix_dataset1 = covariance_matrix(dataset1)
# print(dataset1)
# print(mean(dataset1))
# print(center_dataset(dataset1))
# print(covariance_matrix_dataset1)
# mu = np.zeros(dataset1.shape[1])
# print(mu)
