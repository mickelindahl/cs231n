import torch
import cs231n.pytorch.experiment as e
from ax.service.managed_loop import optimize
from ax import Experiment, save, load
import json
from ax.modelbridge.factory import get_GPEI
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.plot.contour import plot_contour

from pathlib import Path
import pprint

pp = pprint.pprint


class Tune:
    def __init__(self,
                 model_class,
                 optimizer_class,
                 lab,
                 parameters,
                 total_trials=20,
                 random_seed=12345,
                 verbose=False
                 ):
        self.model_class = model_class
        self.optimizer_class = optimizer_class
        self.lab = lab
        self.parameters = parameters
        self.trial = 1
        self.random_seed = 12345
        self.verbose = verbose
        self.data = []

        self.model=None
        self.experiment=None

        self.total_trials = total_trials

        self.path_result = '{}/{}/tune'.format(self.lab.base_path_results, self.model_class.get_name())

        Path(self.path_result).mkdir(parents=True, exist_ok=True)

        torch.manual_seed(random_seed)

    def train_evaluate(self, parameters):
        experiment = e.Experiment(
            self.model_class,
            self.optimizer_class,
            parameters,
            self.lab,
            experiment_name='tune/trial_{}'.format(self.trial),
            epochs=1,
            verbose=False,
        )

        self.trial += 1

        # experiment.load_latest_epoch()
        experiment.train()

        acc = experiment.check_accuracy()

        if self.verbose:
            print('Accuracy:', acc)
            print('Parameters:')
            pp(parameters)

        self.data.append([{
            'parameters': parameters,
            'acc': acc
        }])

        return acc

    def load(self):
        best_parameters_file = '{}/best_parameters.json'.format(self.path_result)
        with open(best_parameters_file, "r") as f:
            best_parameters = json.load(f)

        values_file = '{}/values.json'.format(self.path_result)
        with open(values_file, "r") as f:
            values = json.load(f)

        experiment_file = '{}/experiment.json'.format(self.path_result)

        experiment = load(experiment_file)

        self.model = get_GPEI(experiment, experiment.fetch_data())
        self.experiment = experiment

        # from ax.modelbridge.factory import get_GPEI
        #
        # m = get_GPEI(experiment, experiment.lookup_data_for_trial())
        # gr = m.gen(n=5, optimization_config=optimization_config)

        return best_parameters, values, experiment

    def experiment_exist(self):

        experiment_file = '{}/experiment.json'.format(self.path_result)

        return Path(experiment_file).exists()

    def plot_contour(self, param_x, param_y):
        render(plot_contour(model=self.model, param_x=param_x, param_y=param_y, metric_name='accuracy'))

    def run(self):

        if self.experiment_exist():
            best_parameters, values, experiment = self.load()
            model = get_GPEI(experiment, experiment.fetch_data())
            self.model = model
            self.experiment = experiment

            return best_parameters, values, experiment, model

        best_parameters, values, experiment, model = optimize(
            parameters=self.parameters,
            evaluation_function=self.train_evaluate,
            objective_name='accuracy',
            total_trials=self.total_trials,
            random_seed=self.random_seed
        )

        self.model = model
        self.experiment = experiment

        # Attach model data in order to be able to recreate mode
        # experiment.data_by_trial
        # experiment.fetch_data()
        # experiment.lookup_data_for_trial or experiment.lookup_data_for_ts

        self.save(best_parameters, values, experiment)

        print('Tune done')

        # self.save({
        #     'best_parameters': best_parameters,
        #     'values': values,
        #     'experiment': experiment,
        #     'model': model
        # })

        print('Tune data saved')

        print(best_parameters, values, experiment, model)

        return best_parameters, values, experiment, model

    def save(self, best_parameters, values, experiment):
        meta_file = '{}/meta.json'.format(self.path_result)
        with open(meta_file, "w") as f:
            json.dump({
                'random_seed': self.random_seed
            }, f, indent=2)

        best_parameters_file = '{}/best_parameters.json'.format(self.path_result)
        with open(best_parameters_file, "w") as f:
            json.dump(best_parameters, f, indent=2)

        values_file = '{}/values.json'.format(self.path_result)
        with open(values_file, "w") as f:
            json.dump(values, f, indent=2)

        data_file = '{}/data.json'.format(self.path_result)
        with open(data_file, "w") as f:
            json.dump(self.data, f, indent=2)

        experiment_file = '{}/experiment.json'.format(self.path_result)

        save(experiment, experiment_file)
