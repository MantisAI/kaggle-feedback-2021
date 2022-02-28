from ast import literal_eval
import numpy as np, os 
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
import torch

LABEL_ALL_SUBTOKENS = True

def get_params(config):
    output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 
            'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']
    labels_to_ids = {v:k for k,v in enumerate(output_labels)}
    ids_to_labels = {k:v for k,v in enumerate(output_labels)}

    train_params = {'batch_size': config['train_batch_size'],
                    'shuffle': True,
                    'num_workers': 1,
                    'pin_memory':True
                    }

    test_params = {'batch_size': config['valid_batch_size'],
                    'shuffle': False,
                    'num_workers': 2,
                    'pin_memory':True
                    }

    return train_params, test_params, output_labels, ids_to_labels, labels_to_ids

def construct_formated_data(train_df, train_text_df, csv_path):
    all_entities = []
    for ii,i in enumerate(train_text_df.iterrows()):
        if ii%100==0: print(ii,', ',end='')
        total = i[1]['text'].split().__len__()
        entities = ["O"]*total
        for j in train_df[train_df['id'] == i[1]['id']].iterrows():
            if j[1]['predictionstring'] and entities and i[1]['text']:
                discourse = j[1]['discourse_type']
                list_ix = [int(x) for x in j[1]['predictionstring'].split(' ')]
                
                entities[list_ix[0]] = f"B-{discourse}"
                for k in list_ix[1:]:
                    if k<len(entities):
                        entities[k] = f"I-{discourse}"
        all_entities.append(entities)
    train_text_df['entities'] = all_entities
    train_text_df.to_csv(csv_path, index=False)

def load_formated_data():
    train_df = pd.read_csv('train.csv')
    train_complete = []
    for idx, row in train_df.iterrows():
        train_complete.append({
            "id": row['id'],
            "discourse_type": row['discourse_type'],
            "discourse_text": row['discourse_text'],
            "predictionstring": row['predictionstring']
        })
    #if include_augmented:
        #train_complete = train_complete + get_augmented_data(include_augmented)
    train_df = pd.DataFrame(train_complete)
    test_names, test_texts = [], []
    for f in list(os.listdir('test')):
        test_names.append(f.replace('.txt', ''))
        test_texts.append(open('test/' + f, 'r').read())
    test_texts = pd.DataFrame({'id': test_names, 'text': test_texts})

    train_text_df = pd.read_csv(f'/content/drive/MyDrive/feedback-kaggle/train_NER.csv')
    # pandas saves lists as string, we must convert back
    train_text_df.entities = train_text_df.entities.apply(lambda x: literal_eval(x) )
    return train_df, train_text_df, test_texts

class dataset(Dataset):
  def __init__(self, dataframe, tokenizer, max_len, get_wids, labels_to_ids):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.get_wids = get_wids # for validation
        self.labels_to_ids = labels_to_ids

  def __getitem__(self, index):
        # GET TEXT AND WORD LABELS 
        text = self.data.text[index]        
        word_labels = self.data.entities[index] if not self.get_wids else None

        # TOKENIZE TEXT
        encoding = self.tokenizer(text.split(),
                             is_split_into_words=True,
                             #return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len)
        word_ids = encoding.word_ids()  
        
        # CREATE TARGETS
        if not self.get_wids:
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:                            
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:              
                    label_ids.append( self.labels_to_ids[word_labels[word_idx]] )
                else:
                    if LABEL_ALL_SUBTOKENS:
                        label_ids.append( self.labels_to_ids[word_labels[word_idx]] )
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx
            encoding['labels'] = label_ids

        # CONVERT TO TORCH TENSORS
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        if self.get_wids: 
            word_ids2 = [w if w is not None else -1 for w in word_ids]
            item['wids'] = torch.as_tensor(word_ids2)
        
        return item

  def __len__(self):
        return self.len

def get_loaders(train_df, train_text_df, tokenizer, test_texts, config, train_params, test_params, labels_to_ids):
    IDS = train_df.id.unique()
    np.random.seed(42)
    train_idx = np.random.choice(np.arange(len(IDS)),int(config['train_val_split']*len(IDS)),replace=False)
    valid_idx = np.setdiff1d(np.arange(len(IDS)),train_idx)
    np.random.seed(None)

    data = train_text_df[['id','text', 'entities']]
    train_dataset = data.loc[data['id'].isin(IDS[train_idx]),['text', 'entities']].reset_index(drop=True)
    test_dataset = data.loc[data['id'].isin(IDS[valid_idx])].reset_index(drop=True)
    training_set = dataset(train_dataset, tokenizer, config['max_length'], False, labels_to_ids)
    testing_set = dataset(test_dataset, tokenizer, config['max_length'], True, labels_to_ids)

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    # TEST DATASET
    test_texts_set = dataset(test_texts, tokenizer, config['max_length'], True, labels_to_ids)
    test_texts_loader = DataLoader(test_texts_set, **test_params)
    return training_loader, testing_loader, test_texts_loader, test_dataset, testing_set, valid_idx, train_df, IDS


def get_train_data(tokenizer, config, train_params, test_params, labels_to_ids, include_augmented=[]):
    train_df, train_text_df, test_texts = load_formated_data()
    return get_loaders(train_df, train_text_df, tokenizer, test_texts, config, train_params, test_params, labels_to_ids)

