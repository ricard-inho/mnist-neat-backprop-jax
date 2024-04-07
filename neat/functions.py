from models.utils import create_layers
from models.model import GenomeClassifier

import jax
import jax.numpy as jnp


def copy_layers_over(rng, num_layers, trained_params, prev_activations, cfg):
    """
    Copy trained parameters over to a new model with potentially different layer structure.

    Parameters:
    rng (jax.random.PRNGKey): Random number generator key.
    num_layers (int): Number of layers in the new model.
    trained_params (dict): Trained parameters from a previously trained model.
    prev_activations (list or None): List of activation functions used in previous layers, or None.
    cfg: Configuration object containing network parameters.

    Returns:
    tuple: Tuple containing new model and its initialized parameters.
    """

    layers, activations = create_layers(rng, num_layers, cfg.network.num_output, prev_activations)
    model = GenomeClassifier(layers=layers, activations=activations)

    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (cfg.network.num_inputs,))
    params = model.init(init_rng, inp)

    for layer, value in trained_params.items():
        if layer in params['params']:
            if 'kernel' in value:
                copy_params = jax.device_get(params['params'][layer]['kernel']).copy().T
                copy_train_params = jax.device_get(value['kernel']).copy().T
                
                # print(copy_params)
                # print(copy_train_params)
                # breakpoint()

                if len(copy_params) >= len(copy_train_params):
                    copy_params[:len(copy_train_params), :len(value['kernel'])] = copy_train_params[:,:len(copy_params[0])]
                    params['params'][layer]['kernel'] = jax.numpy.array(copy_params.T)
                else:
                    copy_params = copy_train_params[:len(copy_params), :len(copy_params[0])]
                    params['params'][layer]['kernel'] = jax.numpy.array(copy_params.T)


            if 'bias' in value:
                copy_bias = jax.device_get(params['params'][layer]['bias']).copy()
                copy_train_bias = jax.device_get(value['bias']).copy()
                if len(copy_bias) > len(copy_train_bias):
                    copy_bias[:len(copy_train_bias)] = copy_train_bias
                    params['params'][layer]['bias'] = jax.numpy.array(copy_bias)
                else:
                    copy_bias[:len(copy_train_bias)] = copy_train_bias[:len(copy_bias)]
                    params['params'][layer]['bias'] = jax.numpy.array(copy_bias)

    return model, params


def add_new_layer(rng, num_layers, trained_params, cfg):
    num_layers.append(1)
    return copy_layers_over(rng, num_layers, trained_params, None, cfg)


def add_new_node(rng, num_layers, trained_params, prev_activations, cfg):
    """
    Add a new node to a randomly selected layer in the model.

    Parameters:
    rng (jax.random.PRNGKey): Random number generator key.
    num_layers (list): List containing the number of layers in the model.
    trained_params (dict): Trained parameters from a previously trained model.
    prev_activations (list or None): List of activation functions used in previous layers, or None.
    cfg: Configuration object containing network parameters.

    Returns:
    tuple: Tuple containing new model and its initialized parameters with an additional node.
    """
    if len(num_layers) <= 0:
        num_layers.insert(0, 1)
    else:
        rng, inp_rng = jax.random.split(rng, 2)
        random_element = jax.random.choice(inp_rng, jnp.asarray(num_layers)).item()
        
        index = jnp.where(jnp.array(num_layers) == random_element)[0][0]
        num_layers[index] = random_element + 1

    return copy_layers_over(rng, num_layers, trained_params, prev_activations,cfg)


def remove_node_over(rng, num_layers, trained_params, prev_activations, cfg):
    layers, activations = create_layers(rng, num_layers, cfg.network.num_output, prev_activations)
    model = GenomeClassifier(layers=layers, activations=activations)

    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (cfg.network.num_inputs,))
    params = model.init(init_rng, inp)

    for layer, value in params['params'].items():
        if layer in trained_params:
            if 'kernel' in value:
                copy_params = jax.device_get(params['params'][layer]['kernel']).copy().T
                copy_train_params = jax.device_get(trained_params[layer]['kernel']).copy().T

                copy_params = copy_train_params[:len(copy_params), :len(copy_params[0])]
                params['params'][layer]['kernel'] = jax.numpy.array(copy_params.T)

            if 'bias' in value:
                copy_arr = jax.device_get(value['bias']).copy()
                for num_node_i in range(len(value['bias'].T)):
                    copy_arr[num_node_i] = jax.device_get(trained_params[layer]['bias']).T[num_node_i]
                
                params['params'][layer]['bias'] = jax.numpy.array(copy_arr)            

    return model, params


def remove_node(rng, num_layers, trained_params, prev_activations, cfg):
    if len(num_layers) == 1:
        return copy_layers_over(rng, num_layers, trained_params, prev_activations, cfg)
    selected_node = None
    for i in range(0, len(num_layers)-1):
        if num_layers[i] > num_layers[i+1]:
            selected_node = i
            break
    if selected_node is not None:
        num_layers[selected_node] -= 1
    else:
        return copy_layers_over(rng, num_layers, trained_params, prev_activations, cfg)
    
    print(num_layers)
    return remove_node_over(rng, num_layers, trained_params, prev_activations, cfg)


def remove_layer(rng, num_layers, trained_params, prev_activations, cfg):

    if len(num_layers) == 1:
        return copy_layers_over(rng, num_layers, trained_params, prev_activations, cfg)
    rng, inp_rng = jax.random.split(rng, 2)
    random_layer = jax.random.choice(inp_rng, jnp.asarray(num_layers[:-1])).item()
    num_layers.remove(random_layer)
    print(num_layers)
    return copy_layers_over(rng, num_layers, trained_params, prev_activations, cfg)