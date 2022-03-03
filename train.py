import os
import data_helpers
import inference
import evaluation
import model_helpers
from train_helpers import train_model

import numpy as np, os
import pandas as pd, gc
from tqdm import tqdm
from torch import cuda


def train(config):
    model, tokenizer = model_helpers.load_model(
        config["model_name"],
        config["local_model_path"],
        config["pretrained_model_path"],
    )
    print("----- Model loaded -----")

    (
        train_params,
        test_params,
        output_labels,
        ids_to_labels,
        labels_to_ids,
    ) = data_helpers.get_params(config)
    (
        training_loader,
        testing_loader,
        test_texts_loader,
        test_dataset,
        testing_set,
        valid_idx,
        train_df,
        IDS,
    ) = data_helpers.get_train_data(
        tokenizer, config, train_params, test_params, labels_to_ids
    )
    print("----- Data loaded -----")

    train_model(
        model,
        config,
        training_loader,
        train_df,
        valid_idx,
        testing_set,
        test_dataset,
        IDS,
        test_params,
        ids_to_labels,
    )

    return (model,
        config,
        training_loader,
        train_df,
        valid_idx,
        testing_set,
        test_dataset,
        IDS,
        test_params,
        ids_to_labels
    )

if __name__ == "__main__":
    from torch import cuda
    import numpy as np

    config = {
        'model_name': "longformer-base-pretrained",   
        'local_model_path': "longformer-base-pretrained",
        'pretrained_model_path': "longformer-base-pretrained/pytorch_model.bin",
        'data_path': os.path.expanduser("~/Google Drive/My Drive/feedback-kaggle/train_NER.csv"),
        'train_val_split': 0.9,
        'max_length': 1024,
        'train_batch_size': 1,
        'valid_batch_size': 2,
        'epochs':5,
        'learning_rates': np.array([5e-5, 2.5e-5, 5e-5, 2.5e-6, 2.5e-6])/4,
        'max_grad_norm':10,
        'device': 'cuda' if cuda.is_available() else 'cpu',
        'loss_weights': [0.3, 0.5,0.5, 0.5,0.5, 0.5,0.5, 1.,1., 1.,1., 0.5,0.5, 0.5, 0.5]
    }
    train(config)
