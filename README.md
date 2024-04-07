# MNIST NEAT Backprop Jax
Implementation of MNIST classification with NEAT algorithm and backpropagation in JAX

## Getting Started

### Installs
```
python3 -m venv .venv
source .venv/bin/activate
pip install hydra-core "jax[cpu]" flax tensorflow matplotlib tqdm scikit-learn
pip install torch torchvision torchaudio
pip install --upgrade setuptools
```
**Note:** Change JAX installation type to fit your enviroment.

### Execute
```
python3 main.py
```

**Note:** Change hydra configuration file for other datasets (Iris or Digits)


### Output
After running `main.py` you can access all the training logs. Below is the accuracy graph for the MNIST dataset.
