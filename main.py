import hydra
import torch.utils.data as data

import tensorflow as tf


## Imports for plotting
import matplotlib.pyplot as plt


import jax
import jax.numpy as jnp
import numpy.random as npr



import optax
from flax.training import train_state, checkpoints

from models.utils import create_layers
from models.model import GenomeClassifier

from neat.functions import add_new_node, add_new_layer, remove_node, remove_layer, copy_layers_over

from torchvision.datasets import MNIST
from datasets.mnist_dataset import FlattenAndCast, NumpyLoader

from datasets.utils import numpy_collate
from datasets.iris_dataset import IrisDataset


from training.functions import train_model, eval_model
import numpy as np

from utils.drawing import draw_graph

from sklearn import datasets
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset




@hydra.main(version_base=None, config_path="configs", config_name="mnist_config")
def main(cfg):

    rng = jax.random.PRNGKey(cfg.jax.PRNGKey)

    num_layers = []
    num_output = 3

    # Model
    layers, activations = create_layers(rng, num_layers, num_output, prev_activations=None)
    model = GenomeClassifier(layers=layers, activations=activations)


    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (4,))

    params = model.init(init_rng, inp)


    optimizer = optax.sgd(learning_rate=0.1)

    for generation in range(cfg.training.generations):

        model_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

        # Logs
        writer = tf.summary.create_file_writer(cfg.logs.log_dir)

        #Dataset 
        iris_dataset = datasets.load_iris()
        X = iris_dataset.data
        y = iris_dataset.target
        X = (X - X.mean()) / np.std(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        train_dataset = IrisDataset(X_train, y_train)
        test_dataset = IrisDataset(X_test, y_test)


        batch_size = 32

        #DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate)
                            
        trained_model_state = train_model(
                                state=model_state, 
                                train_data_loader=train_loader, 
                                test_data_loader=test_loader, 
                                num_epochs=cfg.training.num_epochs, 
                                writer=writer,
                                generation=generation
                            )
        
        eval_model(trained_model_state, test_loader, epoch=cfg.training.num_epochs, writer=writer, generation=generation)


        checkpoints.save_checkpoint(ckpt_dir='/Users/ricardmarsalcastan/Documents/Projects/neat-backprop-jax/checkpoints',  # Folder to save checkpoint in
                                target=trained_model_state,  # What to save. To only save parameters, use model_state.params
                                step=100,  # Training step or other metric to save best model on
                                prefix=f'model_generation_{generation}_',  # Checkpoint file name prefix
                                overwrite=True   # Overwrite existing checkpoint files
                            )
        

        # Drawing graph
        draw_graph(trained_model_state)
        plt.savefig(f"graph_gen{generation}.png")
        plt.close()


        # Evolution
        rng, add_layer_rng, add_node_rng, rmv_layer_rng, rmv_node_rng = jax.random.split(rng, 5)
        trained_params = trained_model_state.params['params']
        modified = False

        # # Add new layer
        # if jax.random.uniform(add_layer_rng, shape=(1,)).item() < 0.2:
        #     print("Adding new layer")
        #     model, trained_params = add_new_layer(rng, num_layers, trained_params)
        #     modified = True

        # # Add new node
        # if jax.random.uniform(add_node_rng, shape=(1,)).item() < 0.9:
        #     print("Adding new node")
        #     model, trained_params = add_new_node(rng, num_layers, trained_params, activations)
        #     modified = True

        # # Remove node
        # if jax.random.uniform(rmv_layer_rng, shape=(1,)).item() < 0.2: 
        #     print("Removing node")
        #     model, trained_params = remove_node(rng, num_layers, trained_params, activations)
        #     modified = True

        # #Remove layer
        # if jax.random.uniform(rmv_node_rng, shape=(1,)).item() < 0.2:   
        #     print("Removing layer") 
        #     model, trained_params = remove_layer(rng, num_layers, trained_params, activations)
        #     modified = True

        print("Adding new node")
        model, trained_params = add_new_node(rng, num_layers, trained_params, activations)
        modified = True

        if modified:
            params = trained_params
        else:
            model, params = copy_layers_over(rng, num_layers, trained_params, activations)


         

if __name__ == "__main__":
    main()