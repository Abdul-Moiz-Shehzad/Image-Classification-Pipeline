import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def get_CIFAR10_data(data_dir=None, num_training=49000, num_validation=1000, num_test=1000, device="cpu"):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the linear classifier. These are the same steps as we used for the
    SVM, but condensed to a single function.  
    """

    if data_dir is None:
        data_dir = './data'

    # Download and construct the CIFAR-10 dataset.
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir,
                                                train=True, 
                                                transform=transforms.ToTensor(),
                                                download=True)

    test_dataset = torchvision.datasets.CIFAR10(root=data_dir,
                                                train=False, 
                                                transform=transforms.ToTensor(),
                                                download=True)

    print(f'CIFAR-10 training dataset has {len(train_dataset)} images, and test dataset has {len(test_dataset)} images.')

    X_train = torch.stack([x for x, _ in train_dataset]) # (num_train, c, h, w)
    y_train = torch.tensor(train_dataset.targets) # (num_train,)
    X_test = torch.stack([x for x, _ in test_dataset]) # (num_test, c, h, w)
    y_test = torch.tensor(test_dataset.targets) # (num_test,)

    # To accelerate, put data to a CUDA GPU device if you have, e.g. the T4 GPU on Google Colab.
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]

    # Our training set will be the first num_train points from the original
    # training set.
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    # We use the first num_test points of the original test set as our
    # test set.
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Preprocessing: reshape the image data into rows
    X_train = torch.reshape(X_train, (X_train.shape[0], -1))
    X_val = torch.reshape(X_val, (X_val.shape[0], -1))
    X_test = torch.reshape(X_test, (X_test.shape[0], -1))

    # Normalize the data: subtract the mean image
    mean_image = torch.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Add bias dimension and transform into columns
    X_train = torch.hstack([X_train, torch.ones((X_train.shape[0], 1), device=device)])
    X_val = torch.hstack([X_val, torch.ones((X_val.shape[0], 1), device=device)])
    X_test = torch.hstack([X_test, torch.ones((X_test.shape[0], 1), device=device)])

    # Package data into a dictionary
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }