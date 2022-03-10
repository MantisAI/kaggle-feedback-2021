from ast import literal_eval

from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import torch
import typer

LABELS = [
    "O",
    "B-Lead",
    "I-Lead",
    "B-Position",
    "I-Position",
    "B-Claim",
    "I-Claim",
    "B-Counterclaim",
    "I-Counterclaim",
    "B-Rebuttal",
    "I-Rebuttal",
    "B-Evidence",
    "I-Evidence",
    "B-Concluding Statement",
    "I-Concluding Statement",
]

class KaggleDataset(Dataset):
    def __init__(self, data_path, tokenizer_model, max_length=1024):
        data = pd.read_csv(data_path)
    
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, add_prefix_space=True)
        self.tokens = data.text.str.split()
        self.labels = data.entities.apply(lambda x: literal_eval(x))

        self.label2id = {label: i for i, label in enumerate(LABELS)}
        
        self.max_length = max_length

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.tokens.iloc[idx], truncation=True, padding="max_length", max_length=self.max_length, is_split_into_words=True)
        labels = self.labels.iloc[idx]

        word_ids = inputs.word_ids(batch_index=0)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(self.label2id[labels[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        inputs = {k: torch.tensor(v) for k, v in inputs.items()}
        return inputs, torch.tensor(label_ids)

def train(data_path, model_path, pretrained_model="allenai/longformer-base-4096", epochs:int=5, batch_size:int=32, learning_rate:float=1e-5, max_length:int=1024, mixed_precision:bool=False): 
    dataset = KaggleDataset(data_path, tokenizer_model=pretrained_model, max_length=max_length)
    data = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AutoModelForTokenClassification.from_pretrained(pretrained_model, num_labels=len(LABELS))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    if mixed_precision:
        scaler = GradScaler()

    for i in range(epochs):
        for batch in tqdm(data):
            inputs, labels = batch

            optimizer.zero_grad()

#            with torch.autocast(device_type="cpu"):
            outputs = model(**inputs)
            loss = criterion(outputs.logits.transpose(1,2), labels)
            
            if mixed_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

if __name__ == "__main__":
    typer.run(train)
