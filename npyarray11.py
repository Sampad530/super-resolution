import numpy as np

# Load your npy file
#array = np.load('reconstructed_images/ground_truth_0.npy')
#array = np.load('dcmToNpy/1.npy')
array = np.load('path_to_512_images/1.npy')


# Summarize the array
print(array.shape)  # Print the shape of the array
print(array)  # Print a portion of the array or full array depending on size

# Check the data type (e.g., float32, uint8, etc.)
print("Data type (dtype):", array.dtype)

# Determine the bit depth from the dtype
bit_depth = array.dtype.itemsize * 8
print("Bit depth:", bit_depth)

