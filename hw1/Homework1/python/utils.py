# adopted from normal_vec2img.m & normal_img2vec.m
import numpy as np

def normal_vec2img(N_est, height, width, m):
    """
    Converts a vector of normal components to a normal image.

    Args:
        N_est (numpy.ndarray): Vector of normal components.
        height (int): Height of the image.
        width (int): Width of the image.
        m (numpy.ndarray): Indices of the normal components.

    Returns:
        numpy.ndarray: Normal image with shape (height, width, 3).
    """
    # Get the length of input parameter p
    p = len(m)

    # Initialize arrays to store normal components
    n_x = np.zeros(height * width)
    n_y = np.zeros(height * width)
    n_z = np.zeros(height * width)

    # Reorganize the normal components into arrays
    for i in range(p):
        # Store the corresponding normal component at index m[i] - 1
        n_x[m[i] - 1] = N_est[i, 0]
        n_y[m[i] - 1] = N_est[i, 1]
        n_z[m[i] - 1] = N_est[i, 2]

    # Reshape the arrays into normal image components
    n_x = n_x.reshape((height, width))
    n_y = n_y.reshape((height, width))
    n_z = n_z.reshape((height, width))

    # Initialize an array to store the final normal image
    normals = np.zeros((height, width, 3))
    normals[:, :, 0] = n_x
    normals[:, :, 1] = n_y
    normals[:, :, 2] = n_z

    # Replace NaN values with 0
    normals[np.isnan(normals)] = 0

    return normals


def normal_img2vec(N_est, m):
    """
    Converts a normal image to a vector of normal components.

    Args:
        N_est (numpy.ndarray): Normal image with shape (height, width, 3).
        m (numpy.ndarray): Indices of the normal components.

    Returns:
        numpy.ndarray: Vector of normal components.
    """
    p = len(m)

    Nx = N_est[:, :, 0]
    Ny = N_est[:, :, 1]
    Nz = N_est[:, :, 2]

    S = np.zeros((p, 3))

    S[:, 0] = Nx.flatten()[m]
    S[:, 1] = Ny.flatten()[m]
    S[:, 2] = Nz.flatten()[m]

    return S

