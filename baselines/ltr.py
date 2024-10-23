import sys
import os

sys.path.append('./allRank')

from allrank.data.dataset_loading import load_libsvm_dataset, create_data_loaders
from allrank.models.model_utils import get_torch_device

import torch
from torch.utils.data import DataLoader
from common import UserActivity

def ltr_rank(user_activities: list[UserActivity]):
    """
    Implementing Learning to Rank using the allRank library.
    """

    train_ds, val_ds = load_libsvm_dataset(
        input_path='./',
        slate_length=240,
        validation_ds_role='vali'
    )

    n_features = train_ds.shape[-1]
    assert n_features == val_ds.shape[-1], "Last dimensions of train_ds and val_ds do not match!"

    # train_dl, val_dl
    train_dl, val_dl = create_data_loaders(
        train_ds, val_ds, num_workers=1, batch_size=64)

    # gpu support
    dev = get_torch_device()
    print("Model training will execute on {}".format(dev.type))




    return None

