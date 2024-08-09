


"""
rt numpy as np
import matplotlib.pyplot as plt
from skimage import io

def box_count_3d(Z, k):
    """Count the number of non-empty kxkxk boxes needed to cover the 3D object."""
    S = np.add.reduceat(
        np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            np.arange(0, Z.shape[1], k), axis=1),
        np.arange(0, Z.shape[2], k), axis=2)
    return len(np.where(S > 0)[0])

def fractal_dimension_3d(Z, threshold=0.9):
    """Calculate the fractal dimension of a 3D object."""
    assert len(Z.shape) == 3

    # Binarize the 3D object
    Z = (Z < threshold)

    # Minimal dimension of box size
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the sizes
    sizes = 2**np.arange(int(np.log(n)/np.log(2)), 1, -1)

    # Count the number of boxes of each size
    counts = [box_count_3d(Z, size) for size in sizes]

    # Fit the points to a line
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

# Load a 3D object (this should be a 3D numpy array)
# Here we create a sample 3D object for demonstration purposes
# Replace this with your actual 3D data loading
def create_sample_3d_object(size):
    Z = np.zeros((size, size, size))
    # Example: create a 3D fractal-like structure (e.g., a 3D cube with smaller cubes removed)
    for x in range(size):
        for y in range(size):
            for z in range(size):
                if (x // 2 + y // 2 + z // 2) % 2 == 0:
                    Z[x, y, z] = 1
    return Z

# Example usage with a sample 3D object
sample_size = 64  # Adjust the size as needed
sample_3d_object = create_sample_3d_object(sample_size)

# Compute the fractal dimension
fd = fractal_dimension_3d(sample_3d_object)
print(f'Fractal Dimension: {fd}')




"""