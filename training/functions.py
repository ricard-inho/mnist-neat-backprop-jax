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

def calculate_loss_acc(state, params, batch):
    data_input, labels = batch
    # Obtain the logits and predictions of the model for the input data

    logits = state.apply_fn(params, data_input)
    loss = optax.softmax_cross_entropy(logits=logits, labels=labels).mean()
    
    pred_labels = jnp.amax(logits, axis=-1)
    # Calculate the loss and accuracy
    acc = (pred_labels == jnp.amax(labels, axis=-1)).mean()
    return loss, acc


@jax.jit  # Jit the function for efficiency
def train_step(state, batch):
    # Gradient function
    grad_fn = jax.value_and_grad(calculate_loss_acc,  # Function to calculate the loss
                                 argnums=1,  # Parameters are second argument of the function
                                 has_aux=True  # Function has additional outputs, here accuracy
                                )
    # Determine gradients for current model, parameters and batch
    (loss, acc), grads = grad_fn(state, state.params, batch)
    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)
    # Return state and any other value we might want
    return state, loss, acc


def train_model(state, train_data_loader, test_data_loader, writer, num_epochs, generation):
    for epoch in tqdm(range(num_epochs)):
        batch_loss = []
        batch_acc = []
        for batch in train_data_loader:
            state, loss, acc = train_step(state, batch)
            batch_loss.append(loss)
            batch_acc.append(acc)

        # Log loss and accuracy to TensorBoard
        with writer.as_default():
            tf.summary.scalar(f'generation_{generation}/train_loss', np.mean(batch_loss), step=epoch)
            tf.summary.scalar(f'generation_{generation}/train_accuracy', np.mean(batch_acc), step=epoch)
        writer.flush()

    return state

    