# MNIST NEAT Backprop Jax
Implementation of MNIST classification with NEAT algorithm with backpropagation in JAX. 

![gif](https://github.com/ricard-inho/mnist-neat-backprop-jax/blob/main/imgs/net.gif)

## Getting Started

### Installs
```
python3 -m venv .venv
source .venv/bin/activate
pip install hydra-core "jax[cpu]" flax tensorflow matplotlib tqdm scikit-learn
pip install torch torchvision torchaudio
pip install --upgrade setuptools
```
**Note:** Change JAX installation type to fit your environment.

### Execute
While the [NEAT algorithm ](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies) encompasses various aspects in its implementation, our emphasis lies solely on the backpropagation JAX implementation of the addition and removal nodes from the graph, rather than the population and species concept of the algorithm.
```
python3 main.py
```

**Note:** Change hydra configuration file for other [datasets](https://github.com/ricard-inho/mnist-neat-backprop-jax/blob/ce54c8f1ca4c07dc830187b1d7926db2d1c65b9b/main.py#L41) (Iris or Digits)


### Output
After running `main.py` you can access all the training logs. Below is the accuracy graph for the MNIST dataset.

<p align="center">
  <img src="https://github.com/ricard-inho/mnist-neat-backprop-jax/blob/main/imgs/mnist_accuracy.png" />
</p>
