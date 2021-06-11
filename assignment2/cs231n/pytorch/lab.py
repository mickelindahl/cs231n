import torch
from pathlib import Path
import pprint

pp = pprint.pprint


class Lab:
    def __init__(
            self,
            loader_train,
            loader_validation,
            loader_test,
            device=torch.device('cuda'),
            dtype=torch.float32,
            base_path_results=None
    ):
        """
        Inputs:
        - loader_train: A torch.utils.data.DataLoader object with training data
        - loader_validation: A torch.utils.data.DataLoader object with validation data
        - loader_test: A torch.utils.data.DataLoader object with test data
        - device: Torch device
        - dtype: Torch data type to be used through out experiment
        """

        self.base_path_results = base_path_results
        self.device = device
        self.dtype = dtype
        self.loader = {
            'train': loader_train,
            'validation': loader_validation,
            'test': loader_test
        }

        self.number_of_samples = {

            'train': sum(map(lambda x: len(x[0]), loader_train)),
            'validation': sum(map(lambda x: len(x[0]), loader_validation)),
            'test': sum(map(lambda x: len(x[0]), loader_test))

        }

        Path(self.base_path_results).mkdir(parents=True, exist_ok=True)
