from flax import linen as nn

class GenomeClassifier(nn.Module):

    layers : list
    activations : list

    @nn.compact
    def __call__(self, x):
        # Perform the calculation of the model to determine the prediction
        for layer_i in range(len(self.layers) - 1):
            x = self.layers[layer_i](x)
            x = self.activations[layer_i](x)


        x = self.layers[-1](x)

        return x
    

class SimpleClassifier(nn.Module):
    num_hidden : int   # Number of hidden neurons
    num_outputs : int  # Number of output neurons

    def setup(self):
        # Create the modules we need to build the network
        # nn.Dense is a linear layer
        self.linear1 = nn.Dense(features=self.num_hidden)
        self.linear2 = nn.Dense(features=self.num_outputs)

    def __call__(self, x):
        # Perform the calculation of the model to determine the prediction
        x = self.linear1(x)
        x = nn.tanh(x)
        x = self.linear2(x)
        return x


if __name__ == '__main__':
    model = SimpleClassifier(num_hidden=8, num_outputs=1)