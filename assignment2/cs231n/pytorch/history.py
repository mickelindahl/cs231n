import matplotlib.pyplot as plt

class History:
    def __init__(self):
        self.data = {
            'accuracy_iteration_loss': [],
            'accuracy_iteration_train': [],
            'accuracy_iteration_validation': [],

            'accuracy_epoch_loss': [],
            'accuracy_epoch_train': [],
            'accuracy_epoch_validation': []
        }

    def add(self, data_point, data_point_type='accuracy_iteration_loss'):
        self.data[data_point_type].append(data_point)

    def get_accuracy_per_iteration_train(self):
        return dict(self.data['accuracy_iteration_train'])

    def get_accuracy_per_iteration_validation(self):
        return dict(self.data['accuracy_iteration_validation'])

    def show_iteration(self):

        plt.subplot(2, 1, 1)
        plt.plot(*zip(*self.data['accuracy_iteration_loss']))
        plt.xlabel('iteration')
        plt.ylabel('loss')

        plt.subplot(2, 1, 2)
        plt.plot(*zip(*self.data['accuracy_iteration_train']), '-o')
        plt.plot(*zip(*self.data['accuracy_iteration_validation']), '-o')
        plt.legend(['train', 'val'], loc='upper left')
        plt.xlabel('iteration')
        plt.ylabel('accuracy')

        plt.tight_layout()
        plt.show()

    def get_data(self):
        return self.data

    def update_data(self, data):
        self.data.update(data)
