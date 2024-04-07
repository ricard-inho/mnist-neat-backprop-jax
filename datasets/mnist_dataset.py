import numpy as np
import jax

def mnist_transform(x):
    """
    Transform MNIST image data.

    Parameters:
    x (array-like): Input image data.

    Returns:
    array-like: Flattened and normalized image data.
    """
    np_img = np.expand_dims(np.array(x, dtype=np.float32), axis=2) / 255.
    return np_img.flatten()

def mnist_collate_fn(batch):
    """
    Collate function for MNIST dataset.

    Parameters:
    batch (list): List of tuples, each containing input image and corresponding label.

    Returns:
    tuple: Tuple of two arrays - input images and one-hot encoded labels.
    """
    batch = list(zip(*batch))
    x = np.stack(batch[0])
    y = jax.nn.one_hot(np.array(batch[1]), 10)
    return x, y