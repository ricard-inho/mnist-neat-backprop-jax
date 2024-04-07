from flax import linen as nn

import jax


def create_layers(rng, num_hidden, num_output, prev_activations):
    """
    Create layers and corresponding activations for the model.

    Parameters:
    rng (jax.random.PRNGKey): Random number generator key.
    num_hidden (list): List of integers representing the number of hidden units in each layer.
    num_output (int): Number of output units in the final layer.
    prev_activations (list or None): List of activation functions used in previous layers, or None.

    Returns:
    tuple: Tuple containing lists of layers and activations.
    """
    
    layers = []
    activations = []
    for hidden in num_hidden:
            rng, inp_rng = jax.random.split(rng, 2)
            layers.append(nn.Dense(features=hidden))

            if prev_activations is None or len(num_hidden) != len(prev_activations):
                random_number = jax.random.uniform(inp_rng, shape=(1,)).item()
                if random_number < 0.25:
                    activations.append(nn.relu)
                elif random_number < 0.5:
                    activations.append(nn.sigmoid)
                elif random_number < 0.75:
                    activations.append(nn.tanh)
                else:
                    activations.append(nn.leaky_relu)
            else: 
                activations = prev_activations

    layers.append(nn.Dense(features=num_output))

    return layers, activations


# loaded_model_state = checkpoints.restore_checkpoint(
#                                             ckpt_dir='/workspace/checkpoints/',   # Folder with the checkpoints
#                                             target=model_state,   # (optional) matching object to rebuild state in
#                                             prefix='my_model'  # Checkpoint file name prefix
#                                         )