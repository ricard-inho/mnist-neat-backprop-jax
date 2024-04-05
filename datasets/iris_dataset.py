from torch.utils.data import DataLoader, Dataset
import jax

class IrisDataset(Dataset):
    def __init__(self, X, y, num_classes=3):
        self.X = X
        self.y = jax.nn.one_hot(y, num_classes) 

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]