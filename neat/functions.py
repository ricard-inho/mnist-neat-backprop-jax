from models.utils import create_layers
from models.model import GenomeClassifier

import jax
import jax.numpy as jnp


def copy_layers_over(rng, num_layers, trained_params, prev_activations, num_output=3):

    layers, activations = create_layers(rng, num_layers,num_output, prev_activations)
    model = GenomeClassifier(layers=layers, activations=activations)

    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (4,))
    params = model.init(init_rng, inp)

    for layer, value in trained_params.items():
        if layer in params['params']:
            if 'kernel' in value:
                copy_params = jax.device_get(params['params'][layer]['kernel']).copy().T
                copy_train_params = jax.device_get(value['kernel']).copy().T
                if len(copy_params) > len(copy_train_params):
                    copy_params[:len(copy_train_params), :len(value['kernel'])] = copy_train_params
                    params['params'][layer]['kernel'] = jax.numpy.array(copy_params.T)
                else:
                    copy_params[:len(copy_train_params), :len(value['kernel'])] = copy_train_params[:len(copy_params)]
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


def add_new_layer(rng, num_layers, trained_params):
    num_layers.append(1)
    return copy_layers_over(rng, num_layers, trained_params, None)


def add_new_node(rng, num_layers, trained_params, prev_activations):
    if len(num_layers) <= 0:
        num_layers.insert(0, 1)
    else:
        rng, inp_rng = jax.random.split(rng, 2)
        random_element = jax.random.choice(inp_rng, jnp.asarray(num_layers)).item()
        
        index = jnp.where(jnp.array(num_layers) == random_element)[0][0]
        num_layers[index] = random_element + 1

    return copy_layers_over(rng, num_layers, trained_params, prev_activations)


def remove_layers_over(rng, num_layers, trained_params, prev_activations, num_output=3):
    layers, activations = create_layers(rng, num_layers, num_output, prev_activations)
    model = GenomeClassifier(layers=layers, activations=activations)

    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (4,))
    params = model.init(init_rng, inp)

    for layer, value in params['params'].items():
        if layer in trained_params:
            if 'kernel' in value:
                copy_params = jax.device_get(params['params'][layer]['kernel']).copy().T
                copy_train_params = jax.device_get(trained_params[layer]['kernel']).copy().T
                copy_params = copy_train_params[:len(copy_params), :len(copy_train_params)]
                params['params'][layer]['kernel'] = jax.numpy.array(copy_params.T)


            if 'bias' in value:
                copy_arr = jax.device_get(value['bias']).copy()
                for num_node_i in range(len(value['bias'].T)):
                    copy_arr[num_node_i] = jax.device_get(trained_params[layer]['bias']).T[num_node_i]
                
                params['params'][layer]['bias'] = jax.numpy.array(copy_arr)

    return model, params


def remove_node(rng, num_layers, trained_params, prev_activations):
    if len(num_layers) == 1:
        return copy_layers_over(rng, num_layers, trained_params, prev_activations)
    selected_node = None
    for i in range(1, len(num_layers)-1):
        if num_layers[i] > num_layers[i+1]:
            selected_node = i
            break
    if selected_node is not None:
        num_layers[selected_node] -= 1
    else:
        return copy_layers_over(rng, num_layers, trained_params, prev_activations)
    
    return remove_layers_over(rng, num_layers, trained_params, prev_activations)


def remove_layer(rng, num_layers, trained_params, prev_activations):

    if len(num_layers) == 1:
        return copy_layers_over(rng, num_layers, trained_params, prev_activations)
    rng, inp_rng = jax.random.split(rng, 2)
    random_layer = jax.random.choice(inp_rng, jnp.asarray(num_layers[:-1])).item()
    num_layers.remove(random_layer)
    return copy_layers_over(rng, num_layers, trained_params, prev_activations)