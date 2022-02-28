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
        config['model_name'], 
        config['local_model_path'], 
        config['pretrained_model_path']
    )
    print("----- Model loaded -----")

    train_params, test_params, output_labels, ids_to_labels, labels_to_ids = data_helpers.get_params(config)
    training_loader, testing_loader, test_texts_loader, test_dataset, testing_set, valid_idx, train_df, IDS = data_helpers.get_train_data(
        tokenizer, config, train_params, test_params, labels_to_ids
    )
    print("----- Data loaded -----")

    train_model(model, config, training_loader, train_df, valid_idx, testing_set, test_dataset, IDS, test_params)

