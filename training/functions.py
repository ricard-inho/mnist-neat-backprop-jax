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
    """
    Calculate loss and accuracy for a batch of data.

    Parameters:
    state: Model state.
    params: Model parameters.
    batch: Tuple containing input data and labels.
    num_output (int): Number of output classes.

    Returns:
    tuple: Tuple containing loss and accuracy.
    """

    data_input, labels = batch

    logits = state.apply_fn(params, data_input)
    loss = optax.softmax_cross_entropy(logits=logits, labels=labels).mean()

    # Logits to probabilities
    probs = jax.nn.softmax(logits)
    max_index = jnp.argmax(probs, axis=-1)
    
    
    pred_labels = jax.nn.one_hot(max_index, num_output) 

    acc = jnp.all(pred_labels == labels, axis=-1).mean()

    return loss, acc


# @jax.jit  # Jit the function for efficiency
@partial(jax.jit, static_argnums=(2,))
def train_step(state, batch, num_output):
    """
    Perform a single training step.

    Parameters:
    state: Model state.
    batch: Tuple containing input data and labels.
    num_output (int): Number of output classes.

    Returns:
    tuple: Tuple containing updated model state, loss, and accuracy.
    """

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
    """
    Train the model for a specified number of epochs.

    Parameters:
    state: Initial model state.
    train_data_loader: Data loader for training dataset.
    test_data_loader: Data loader for testing dataset.
    writer: TensorBoard writer for logging.
    num_epochs (int): Number of epochs to train.
    generation (int): Current generation number.
    num_output (int): Number of output classes.

    Returns:
    state: Trained model state.
    """

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
    """
    Evaluate the model on a single batch.

    Parameters:
    state: Model state.
    batch: Tuple containing input data and labels.
    num_output (int): Number of output classes.

    Returns:
    float: Accuracy.
    """

    # Determine the accuracy
    _, acc = calculate_loss_acc(state, state.params, batch, num_output)
    return acc

def eval_model(state, data_loader, epoch, writer, generation, num_output):
    """
    Evaluate the model on the entire dataset.

    Parameters:
    state: Model state.
    data_loader: Data loader for evaluation dataset.
    epoch (int): Current epoch number.
    writer: TensorBoard writer for logging.
    generation (int): Current generation number.
    num_output (int): Number of output classes.
    """
    
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
