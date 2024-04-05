import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax

import tensorflow as tf

from tqdm.auto import tqdm

from datasets.utils import visualize_classification
from utils.drawing import draw_graph

import matplotlib.pyplot as plt
import numpy as np

def calculate_loss_acc(state, params, batch):
    data_input, labels = batch
    # Obtain the logits and predictions of the model for the input data
    logits = state.apply_fn(params, data_input).squeeze(axis=-1)
    pred_labels = (logits > 0).astype(jnp.float32)
    # Calculate the loss and accuracy
    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    acc = (pred_labels == labels).mean()
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

@jax.jit  # Jit the function for efficiency
def eval_step(state, batch):
    # Determine the accuracy
    _, acc = calculate_loss_acc(state, state.params, batch)
    return acc

def train_model(state, data_loader, test_data_loader, eval_dataset, writer, model, generation, visualize_epochs, num_epochs=100):

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        batch_loss = []
        batch_acc = []
        for batch in data_loader:
            state, loss, acc = train_step(state, batch)
            batch_loss.append(loss)
            batch_acc.append(acc)
            
        # Log loss and accuracy to TensorBoard
        with writer.as_default():
            tf.summary.scalar(f'generation_{generation}/train_loss', np.mean(batch_loss), step=epoch)
            tf.summary.scalar(f'generation_{generation}/train_accuracy', np.mean(batch_acc), step=epoch)
        writer.flush()

        eval_model(state, test_data_loader, epoch, writer, generation)
        
        if visualize_epochs:
            trained_model = model.bind(state.params)
            _ = visualize_classification(trained_model, eval_dataset.data, eval_dataset.label, epoch)
            plt.savefig(f"visualize_classification_gen{generation}_epoch_{epoch}.png")
            plt.close()

            # Drawing graph
            # draw_graph(state)
            # plt.savefig(f"graph_gen{generation}_epoch_{epoch}.png")
            # plt.close()
    
    return state

def eval_model(state, data_loader, epoch, writer, generation):
    
    all_accs, batch_sizes = [], []
    for batch in data_loader:
        batch_acc = eval_step(state, batch)
        all_accs.append(batch_acc)
        batch_sizes.append(batch[0].shape[0])
    # Weighted average since some batches might be smaller
    acc = sum([a*b for a,b in zip(all_accs, batch_sizes)]) / sum(batch_sizes)

    # Log accuracy to TensorBoard
    with writer.as_default():
        tf.summary.scalar(f'generation_{generation}/eval_accuracy', acc, step=epoch)
    writer.flush()  

    # print(f"Accuracy of the model: {100.0*acc:4.2f}%")