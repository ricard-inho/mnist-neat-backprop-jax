import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

import flax
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax

import tensorflow as tf

# from datasets.utils import visualize_classification
from utils.drawing import draw_graph


import numpy as np

from jax.scipy.special import logsumexp

from tqdm import tqdm

from functools import partial

def calculate_loss_acc(state, params, batch, num_output):
    data_input, labels = batch

    # if cfg.dataset.dataset_type == 'mnist':
    #     labels = jax.nn.one_hot(labels, cfg.network.num_output)


    # Obtain the logits and predictions of the model for the input data

    logits = state.apply_fn(params, data_input)
    loss = optax.softmax_cross_entropy(logits=logits, labels=labels).mean()

    # Logits to probabilities
    probs = jax.nn.softmax(logits)
    max_index = jnp.argmax(probs, axis=-1)
    
    
    pred_labels = jax.nn.one_hot(max_index, num_output) 
    # Calculate the loss and accuracy
    # (pred_labels == labels).mean()

    acc = jnp.all(pred_labels == labels, axis=-1).mean()

    return loss, acc


# @jax.jit  # Jit the function for efficiency
@partial(jax.jit, static_argnums=(2,))
def train_step(state, batch, num_output):
    # Gradient function
    grad_fn = jax.value_and_grad(calculate_loss_acc,  # Function to calculate the loss
                                 argnums=1,  # Parameters are second argument of the function
                                 has_aux=True  # Function has additional outputs, here accuracy
                                )
    # Determine gradients for current model, parameters and batch
    (loss, acc), grads = grad_fn(state, state.params, batch, num_output)
    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)
    # Return state and any other value we might want
    return state, loss, acc


def train_model(state, train_data_loader, test_data_loader, writer, num_epochs, generation, num_output):
    for epoch in tqdm(range(num_epochs)):
        batch_loss = []
        batch_acc = []
        for batch in train_data_loader:
            state, loss, acc = train_step(state, batch, num_output)
            batch_loss.append(loss)
            batch_acc.append(acc)

        # Log loss and accuracy to TensorBoard
        with writer.as_default():
            tf.summary.scalar(f'generation_{generation}/train_loss', np.mean(batch_loss), step=epoch)
            tf.summary.scalar(f'generation_{generation}/train_accuracy', np.mean(batch_acc), step=epoch)
        writer.flush()
        
        eval_model(state, test_data_loader, epoch, writer, generation, num_output)

    return state


# @jax.jit  # Jit the function for efficiency
@partial(jax.jit, static_argnums=(2,))
def eval_step(state, batch, num_output):
    # Determine the accuracy
    _, acc = calculate_loss_acc(state, state.params, batch, num_output)
    return acc

def eval_model(state, data_loader, epoch, writer, generation, num_output):
    
    all_accs, batch_sizes = [], []
    for batch in data_loader:
        batch_acc = eval_step(state, batch, num_output)
        all_accs.append(batch_acc)
        batch_sizes.append(batch[0].shape[0])
    # Weighted average since some batches might be smaller
    acc = sum([a*b for a,b in zip(all_accs, batch_sizes)]) / sum(batch_sizes)

    # Log accuracy to TensorBoard
    with writer.as_default():
        tf.summary.scalar(f'generation_{generation}/eval_accuracy', acc, step=epoch)
    writer.flush()  
