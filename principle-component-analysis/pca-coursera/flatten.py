import numpy as np

def reshape(x):
	"""return x_reshaped as a flattened vector of the multi-dimensional array x"""
	# x_reshaped = x.flatten()
	x_reshaped = [elem for row in x for elem in row]
	return x_reshaped

data = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
)

flattened_data = reshape(data)
print(flattened_data, len(flattened_data))
print(flattened_data[2])
