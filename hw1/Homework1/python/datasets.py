# adopted from imread_datadir_re.m & load_datadir_re.m
import numpy as np
from skimage import io, transform
import os

# Helper functions
def read_floats__(fn):
    out = np.loadtxt(fn, dtype=float)
    return out

def read_strings__(fn):
    with open(fn, 'r') as file:
        all_lines = file.readlines()
        out = [line.strip() for line in all_lines]
    return out

def imread_datadir_re(datadir, which_image, bitDepth, resize, gamma):
    """
    This function reads an image from a given directory and performs various operations on it.
    
    Args:
        datadir (str or struct): The directory containing the image(s) to be loaded.
            If it is a struct, it should have an 'imgs' field.
            If it is a string, it should be the path to the image file.
        which_image (int): The index of the image to be loaded from the directory.
        bitDepth (int): The bit depth of the image.
        resize (tuple): The desired size of the image after resizing.
        gamma (float): The gamma value for normalizing the image.
    
    Returns:
        numpy.ndarray: The processed image.
    """
    # Check if datadir is a struct with 'imgs' field
    if hasattr(datadir, 'imgs'):
        img = datadir.imgs[which_image]
    else:
        # Load image(s) from disk
        img = io.imread(datadir['filenames'][which_image])
        img = (img / (2**bitDepth - 1))**gamma  # Normalize image
        img = transform.rescale(img, resize, anti_aliasing=False)
        
        H, W, C = img.shape

        # Normalize the image with light intensities
        L_inv = 1.0 / datadir['L'][which_image, :]
        img = np.reshape(np.reshape(img, [H * W, C]) @ np.diag(L_inv), [H, W, C])
        img = np.maximum(0, img)

    return img


def load_datadir_re(datadir, bitDepth, resize, gamma):
    """
    This function loads data from a given directory and performs various operations on it.

    Args:
        datadir (str or dict): The directory containing the data to be loaded.
            If it is a dict, it should have the necessary fields.
            If it is a string, it should be the path to the data directory.
        bitDepth (int): The bit depth of the data.
        resize (tuple): The desired size of the data after resizing.
        gamma (float): The gamma value for normalizing the data.

    Returns:
        dict: The loaded and processed data.
    """

    # Parse options
    load_imgs = True
    load_mask = True
    white_balance = np.eye(3)  # Default white balance
    # white_balance = white_balance.reshape((1, 3, 3))

    # Build data struct
    if isinstance(datadir, dict):
        data = datadir
    elif isinstance(datadir, str):
        data = {}
        data['s'] = read_floats__(os.path.join(datadir, 'light_directions.txt'))
        data['L'] = read_floats__(os.path.join(datadir, 'light_intensities.txt'))
        data['L'] = np.dot(data['L'], white_balance)
        data['filenames'] = read_strings__(os.path.join(datadir, 'filenames.txt'))
        data['filenames'] = [os.path.join(datadir, x) for x in data['filenames']]
    else:
        raise ValueError('datadir is neither a struct nor a string!')

    # Load mask
    if 'mask' not in data and load_mask:
        data['mask'] = io.imread(os.path.join(datadir, 'mask.png'))
        data['mask'] = transform.rescale(data['mask'], resize, anti_aliasing=False)
        
    
    # Load images
    if 'imgs' not in data and load_imgs:
        data_ = data.copy()
        length = len(data['filenames'])
        data['imgs'] = [imread_datadir_re(data_, i, bitDepth, resize, gamma) for i in range(length)]

    return data



