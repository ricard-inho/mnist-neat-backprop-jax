import numpy as np
import jax

def mnist_transform(x):
    np_img = np.expand_dims(np.array(x, dtype=np.float32), axis=2) / 255.
    return np_img.flatten()

def mnist_collate_fn(batch):
    batch = list(zip(*batch))
    x = np.stack(batch[0])
    y = jax.nn.one_hot(np.array(batch[1]), 10)
    return x, y