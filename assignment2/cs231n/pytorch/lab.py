import torch
from pathlib import Path
import pprint
from torch.utils.tensorboard import SummaryWriter

pp = pprint.pprint


class Lab:
    def __init__(
            self,
            loader_train,
            loader_validation,
            loader_test,
            lab_name,
            loader_no_transform=None,
            device=torch.device('cuda'),
            dtype=torch.float32,
            base_path_output=None,

    ):
        """
        Inputs:
        - loader_train: A torch.utils.data.DataLoader object with training data
        - loader_validation: A torch.utils.data.DataLoader object with validation data
        - loader_test: A torch.utils.data.DataLoader object with test data
        - device: Torch device
        - dtype: Torch data type to be used through out experiment
        """

        self.base_path_output = base_path_output
        self.device = device
        self.dtype = dtype
        self.loader = {
            'train': loader_train,
            'validation': loader_validation,
            'test': loader_test,
            'loader_no_transform': loader_no_transform
        }

        self.number_of_samples = {

            'train': sum(map(lambda x: len(x[0]), loader_train)),
            'validation': sum(map(lambda x: len(x[0]), loader_validation)),
            'test': sum(map(lambda x: len(x[0]), loader_test))
        }

        self.base_path_torch = '{}/{}/torch'.format(base_path_output, lab_name)
        self.base_path_tensorboard = '{}/{}/tensorboard'.format(base_path_output, lab_name)
        self.lab_name = lab_name

        Path(self.base_path_output).mkdir(parents=True, exist_ok=True)

        base_path_tensorboard_images = '{}/{}'.format(self.base_path_tensorboard, 'images')
        writer = SummaryWriter(base_path_tensorboard_images)

        for tag, loader in [[lab_name + '_sample', loader_no_transform],
                            [lab_name + '_sample_with_transform', loader_train]]:
            dataiter = iter(loader)
            images, labels = dataiter.next()

            writer.add_images(tag, images)

        writer.close()

    def get_name(self):
        return self.lab_name
