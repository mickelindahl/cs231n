import torch
import torch.nn.functional as F  # useful stateless functions
from cs231n.pytorch.history import History
from copy import deepcopy
from pathlib import Path
import torch.optim as optim
import json
import os
import pprint

pp = pprint.pprint


class Experiment:
    def __init__(
            self,
            model_class,
            optimizer_class,
            params,
            lab,
            experiment_name='default',
            epochs=1,
            verbose=False,
    ):
        """
        Inputs:
        - model: A PyTorch Module giving the model to train.
        - optimizer_params: Optimizer parameters
        - lab object with pointers to test, train and validation data sets as well
        as some meta data
        - epochs: Number of epoch to train
        - verbose: Add printouts
        """
        # model = model.to(device=lab.device)  # move the model parameters to CPU/GPU

        self.model = model_class(**params).to(device=lab.device)
        self.best_model = model_class(**params).to(device=lab.device)


        opt_params={}
        for key, val in params.items():
            if key in ['lr', 'moment' ]:
                opt_params[key] = val

        self.optimizer = optimizer_class(self.model.parameters(), **opt_params)
        self.params = params
        self.loader = lab.loader

        self.number_of_samples = lab.number_of_samples

        self.device = lab.device
        self.dtype = lab.dtype
        self.verbose = verbose

        self.path_result = '{}/{}/{}'.format(lab.base_path_results, self.model.get_name(), experiment_name)

        Path(self.path_result).mkdir(parents=True, exist_ok=True)

        self.history = History()

        self.best_model_state = deepcopy(self.model.state_dict())
        self.best_accuracy = None
        self.best_epoch = None
        self.best_iteration = None
        self.epoch = 0
        self.iteration = 0
        self.epochs = epochs

    def use_model(self, *args, **kwargs):

        model = self.model(*args, **kwargs)

        if self.cache_models[str(model)]:
            self.model = self.cache_models[str(model)]
        else:
            self.cache_models[str(model)] = model
            self.model = model

    def check_accuracy(self, check='validation', best_model=False):
        """
        Check model accuracy

        Inputs:
        - model: A PyTorch Module giving the trained model.
        - check: Data check ot check accuracy against train | test |val

        Returns: Nothing, but prints model accuracies during training.
        """

        loader = self.loader[check]

        if self.verbose:
            print('Checking accuracy on {} set'.format(check))

        num_correct = 0
        num_samples = 0

        model = self.model
        if best_model:
            model = self.best_model

        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
                y = y.to(device=self.device, dtype=torch.long)
                scores = model(x)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            if self.verbose:
                print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

        return acc

    def get_saved_epoch_file_names(self):

        result_files = {}
        if os.path.isdir(self.path_result):

            files = os.listdir(self.path_result)

            for fn in files:
                if fn.split('_')[0] != 'epoch':
                    continue

                name, ending = fn.split('.')
                _, e = name.split('_')

                if not result_files.get(e):
                    result_files[e] = {}

                result_files[e][ending] = fn

        return result_files

    def list_saved_epochs(self):
        print('Saved epochs:')

        epochs = list(self.get_saved_epoch_file_names().items())
        epochs.sort(key=lambda x: int(x[0]))
        pp(epochs)
        print('')

    def load_best_model(self):

        model_file = "{}/{}".format(self.path_result, 'best_model.pt')
        self.best_model.load_state_dict(torch.load(model_file))
        self.best_model.eval()

    def load_epoch(self, epoch):

        result_files = self.get_saved_epoch_file_names()

        if not result_files.get(epoch):
            print('Missing epoch', epoch)
            print('Available')
            self.list_saved_epochs()

        json_file = "{}/{}".format(self.path_result, result_files[epoch]['json'])
        model_file = "{}/{}".format(self.path_result, result_files[epoch]['pt'])

        with open(json_file, "r") as f:
            checkpoint = json.load(f)

        self.history.update_data(checkpoint['history'])
        self.epoch = checkpoint['epoch']
        self.iteration = checkpoint['iteration']

        self.model.load_state_dict(torch.load(model_file))
        self.model.eval()

        self.load_best_model()

    def load_latest_epoch(self):

        epochs = list(self.get_saved_epoch_file_names().keys())
        epochs.sort(key=lambda x: int(x))

        if not epochs:
            return

        self.load_epoch(epochs[-1])

    def train(
            self,
            print_every=100
    ):
        """
        Train a model on training data from the loader using the PyTorch Module API.

        Inputs:
         - loader_train: A torch.utils.data.DataLoader object with training data
        - device: cpu | gpu
        - epochs: (Optional) A Python integer giving the number of epochs to train for

        Returns: Nothing, but prints model accuracies during training.
        """

        loader_train = self.loader['train']

        num_correct = 0
        num_samples = 0
        start = self.epoch

        if self.epoch == self.epochs:
            print('Training done skipping')
            return

        for e in range(start, self.epochs):

            for t, (x, y) in enumerate(loader_train):
                self.model.train()  # put model to training mode
                x = x.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
                y = y.to(device=self.device, dtype=torch.long)

                scores = self.model(x)

                _, preds = scores.max(1)

                num_correct += (preds == y).sum()
                num_samples += preds.size(0)

                loss = F.cross_entropy(scores, y)

                # Zero out all of the gradients for the variables which the optimizer
                # will update.
                self.optimizer.zero_grad()

                # This is the backwards pass: compute the gradient of the loss with
                # respect to each  parameter of the model.
                loss.backward()

                # Actually update the parameters of the model using the gradients
                # computed by the backwards pass.
                self.optimizer.step()

                if t % print_every == 0:

                    if self.verbose: print('Iteration %d, loss = %.4f' % (self.iteration, loss.item()))
                    acc_val = self.check_accuracy(check='validation')

                    if not self.best_accuracy or self.best_accuracy < acc_val:
                        self.best_accuracy = acc_val
                        self.best_epoch = self.epoch
                        self.best_iteration = self.iteration
                        self.best_model_state = deepcopy(self.model.state_dict())
                        self.best_model.load_state_dict(self.best_model_state)

                    acc_train = float(num_correct) / num_samples

                    self.history.add((self.iteration, float(loss.cpu().detach().numpy())),
                                     data_point_type='accuracy_iteration_loss')
                    self.history.add((self.iteration, acc_train), data_point_type='accuracy_iteration_train')
                    self.history.add((self.iteration, acc_val), data_point_type='accuracy_iteration_validation')

                self.iteration += 1

            self.epoch = e + 1

            self._save_checkpoint()

    def _save_checkpoint(self):
        if self.path_result is None:
            return

        checkpoint = {
            "params": self.params,
            "batch_size": {
                'train': self.loader['train'].batch_size,
                'train': self.loader['validation'].batch_size,
                'train': self.loader['test'].batch_size
            },
            "number_of_samples": self.number_of_samples,
            "epoch": self.epoch,
            "iteration": self.iteration,
            "history": self.history.get_data(),
            "best_accuracy": self.best_accuracy,
            "best_epoch": self.best_epoch,
            "best_iteration": self.best_iteration
        }

        filename = "{}/epoch_{}.".format(self.path_result, self.epoch)
        json_file = filename + 'json'
        model_file = filename + 'pt'
        best_model_file = '{}/best_model.pt'.format(self.path_result)

        if self.verbose:
            print('Saving json checkpoint to "%s"' % json_file)
            print('Saving model checkpoint stateto "%s"' % model_file)
            print('Saving best model state to "%s"' % best_model_file)

        with open(json_file, "w") as f:
            json.dump(checkpoint, f, indent=2)

        torch.save(self.model.state_dict(), model_file)
        torch.save(self.best_model_state, best_model_file)
