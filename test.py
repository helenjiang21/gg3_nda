import numpy as np

# Example 2D array
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# Find the index of the maximum value in the flattened array
max_index_flat = np.argmax(data)

# Convert the flat index to row and column indices
max_index_2d = np.unravel_index(max_index_flat, data.shape)

print(type(max_index_2d))
print("Index of the largest value:", max_index_2d)