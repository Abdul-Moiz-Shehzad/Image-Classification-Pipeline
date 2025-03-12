import numpy as np
import torch


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, train_dataset):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.
        """
        self.train_dataset = train_dataset
        X = []
        y = []
        for train_image, train_label in train_dataset:
            X.append(train_image.flatten())
            y.append(train_label)
        self.X_train = torch.stack(X) 
        self.y_train = torch.tensor(y) 

    def predict(self, test_dataset, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(test_dataset)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(test_dataset)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(test_dataset)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, test_dataset):
        """
        Compute the Euclidean distance between each test example in test_dataset and each training example
        in self.train_dataset using a nested loop over both the training data and the
        test data.
        """
        num_test = len(test_dataset)
        num_train = len(self.train_dataset)
        dists = torch.zeros((num_test, num_train))
        for i in range(num_test):
            test_image_i, _ = test_dataset[i]
            test_sample_i = test_image_i.flatten()
            for j in range(num_train):
                train_image_j = self.X_train[j]
                dists[i, j] = torch.sqrt(torch.sum((test_sample_i - train_image_j) ** 2))
        return dists

    def compute_distances_one_loop(self, test_dataset):
        """
        Compute the Euclidean distance between each test example in test_dataset and each training example
        in self.train_dataset using a single loop over the test data.
        """
        num_test = len(test_dataset)
        num_train = len(self.train_dataset)
        dists = torch.zeros((num_test, num_train))
        for i in range(num_test):
            test_image_i, _ = test_dataset[i]
            test_sample_i = test_image_i.flatten()
            dists[i, :] = torch.sqrt(torch.sum((self.X_train - test_sample_i) ** 2, dim=1))
        return dists

    def compute_distances_no_loops(self, test_dataset):
        """
        Compute the Euclidean distance between each test example in test_dataset and each training example
        in self.train_dataset using no explicit loops.
        """
        test_samples = torch.stack([test_image.flatten() for test_image, _ in test_dataset])
        test_sum = torch.sum(test_samples ** 2, dim=1, keepdim=True)
        train_sum = torch.sum(self.X_train ** 2, dim=1)
        dists = torch.sqrt(test_sum - 2 * test_samples @ self.X_train.T + train_sum)
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.
        """        
        num_test = dists.size(0)
        y_pred = torch.zeros(num_test, dtype=torch.int64)
        for i in range(num_test):
            closest_y = self.y_train[torch.argsort(dists[i])[:k]]
            y_pred[i] = torch.mode(closest_y).values
        return y_pred
