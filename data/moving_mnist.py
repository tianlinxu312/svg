import numpy as np
from torchvision import datasets, transforms


class MovingMNIST(object):
    
    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train, data_root='../data/mmnist/train/', seq_len=20, num_digits=2, image_size=64, deterministic=True):
        self.path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits  
        self.image_size = image_size 
        self.step_length = 0.1
        self.digit_size = 32
        self.deterministic = deterministic
        self.seed_is_set = False # multi threaded loading
        self.channels = 1

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return 20

    def __getitem__(self, index):
        self.set_seed(index)
        training_data = np.load(self.path) / 255.0
        training_data = np.transpose(training_data, (1, 0, 2, 3))
        x = np.transpose(training_data, (0, 2, 1, 3))
        return x


