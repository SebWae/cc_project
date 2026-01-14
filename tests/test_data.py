import os
import pytest

import torch
from torch.utils.data import Dataset

from cc_project.data import corrupt_mnist
from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data():
    train, test = corrupt_mnist()
    assert len(train) == 30000, "The trainset did not contain 30,000 samples"
    assert len(test) == 5000, "The testset did not contain 5,000 samples"
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28), "All data samples are not of shape (1, 28, 28)"
            assert y in range(10), "All output labels are not within the range from 0 to 9"
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all()
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all()
