import hydra
import torch.utils.data as data

import tensorflow as tf


## Imports for plotting
import matplotlib.pyplot as plt


import jax
import jax.numpy as jnp



import optax
from flax.training import train_state, checkpoints

from models.utils import create_layers
from models.model import GenomeClassifier

from neat.functions import add_new_node, add_new_layer, remove_node, remove_layer, copy_layers_over

from datasets.dataset import PointsDataset
from datasets.utils import numpy_collate, visualize_classification

from training.functions import train_model, eval_model

from utils.drawing import draw_graph



@hydra.main(version_base=None, config_path="configs", config_name="neat_backprop_config")
def main(cfg):

    rng = jax.random.PRNGKey(cfg.jax.PRNGKey)

    num_layers = [1]

    # Model
    layers, activations = create_layers(rng, num_layers, prev_activations=None)
    model = GenomeClassifier(layers=layers, activations=activations)


    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (2,))
    params = model.init(init_rng, inp)

    optimizer = optax.sgd(learning_rate=0.1)

    for generation in range(cfg.training.generations):

        model_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

        # Logs
        writer = tf.summary.create_file_writer(cfg.logs.log_dir)


        # Datasets
        # Train
        train_dataset = PointsDataset(size=2500, seed=42, cfg=cfg)
        train_data_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=numpy_collate)

        # Test
        test_dataset = PointsDataset(size=500, seed=123, cfg=cfg)
        test_data_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False, collate_fn=numpy_collate)
        
        # Eval
        eval_dataset = PointsDataset(size=200, seed=42, cfg=cfg)
        eval_data_loader = data.DataLoader(eval_dataset, batch_size=8, shuffle=True, collate_fn=numpy_collate)

        # from datasets.utils import visualize_samples
        # visualize_samples(test_dataset.data, test_dataset.label)
        
        # Training Loop
        trained_model_state = train_model(model_state, train_data_loader, test_data_loader, eval_dataset, writer, model, generation, cfg.training.visualize_epochs, num_epochs=cfg.training.num_epochs)

        checkpoints.save_checkpoint(ckpt_dir='/Users/ricardmarsalcastan/Documents/Projects/neat-backprop-jax/checkpoints',  # Folder to save checkpoint in
                                target=trained_model_state,  # What to save. To only save parameters, use model_state.params
                                step=100,  # Training step or other metric to save best model on
                                prefix=f'model_generation_{generation}_',  # Checkpoint file name prefix
                                overwrite=True   # Overwrite existing checkpoint files
                            )

        

        eval_model(trained_model_state, test_data_loader, epoch=cfg.training.num_epochs, writer=writer, generation=generation)


        trained_model = model.bind(trained_model_state.params)

        _ = visualize_classification(trained_model, eval_dataset.data, eval_dataset.label, cfg.training.num_epochs)
        plt.savefig(f"visualize_classification_gen_{generation}.png")
        plt.close()

        writer.close()

        # Drawing graph
        draw_graph(trained_model_state)
        plt.savefig(f"graph_gen{generation}.png")
        plt.close()


        # Evolution
        rng, add_layer_rng, add_node_rng, rmv_layer_rng, rmv_node_rng = jax.random.split(rng, 5)
        trained_params = trained_model_state.params['params']
        modified = False

        # Add new layer
        if jax.random.uniform(add_layer_rng, shape=(1,)).item() < 0.3:
            print("Adding new layer")
            model, trained_params = add_new_layer(rng, num_layers, trained_params)
            modified = True

        # Add new node
        if jax.random.uniform(add_node_rng, shape=(1,)).item() < 0.9:
            print("Adding new node")
            model, trained_params = add_new_node(rng, num_layers, trained_params, activations)
            modified = True

        # Remove node
        if jax.random.uniform(rmv_layer_rng, shape=(1,)).item() < 0.2: 
            print("Removing node")
            model, trained_params = remove_node(rng, num_layers, trained_params, activations)
            modified = True

        #Remove layer
        if jax.random.uniform(rmv_node_rng, shape=(1,)).item() < 0.2:   
            print("Removing layer") 
            model, trained_params = remove_layer(rng, num_layers, trained_params, activations)
            modified = True

        if modified:
            params = trained_params
        else:
            model, params = copy_layers_over(rng, num_layers, trained_params, activations)


         

if __name__ == "__main__":
    main()