from ast import literal_eval
import os

from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import transformers
import torch
import typer
import wandb

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
        data = data[data["kfold"]==0]
    
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

def train(data_path, model_path, pretrained_model="allenai/longformer-base-4096", epochs:int=5, batch_size:int=32,
          learning_rate:float=1e-5, max_length:int=1024, clip_value:int=5, num_warmup_steps:int=200,
          learning_rate_schedule="cosine_warmup",
          gradient_accumulation_steps:int=1, mixed_precision:bool=False, dry_run:bool=False):
    if not dry_run:
        config = {
            "pretrained_model": pretrained_model,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "mixed_precision": mixed_precision,
            "learning_rate_schedule": learning_rate_schedule
        }
        wandb.init(project="kaggle-torch", config=config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = KaggleDataset(data_path, tokenizer_model=pretrained_model, max_length=max_length)
    data = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AutoModelForTokenClassification.from_pretrained(pretrained_model, num_labels=len(LABELS))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    if learning_rate_schedule == "cosine_warmup":
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=epochs*len(data))

    if mixed_precision:
        scaler = GradScaler()

    try:
        for epoch in range(epochs):
            batches = tqdm(data, desc=f"Epoch {epoch:2d}/{epochs:2d}")
            for step, batch in enumerate(batches):
                inputs, labels = batch

                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                labels = labels.to(device)

                if mixed_precision:
                    with autocast():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        loss = criterion(outputs.logits.transpose(1,2), labels)
                    scaler.scale(loss).backward()
                else:
                    outputs = model(**inputs)
                    loss = criterion(outputs.logits.transpose(1,2), labels)

                if step % gradient_accumulation_steps == 0:
                    if mixed_precision:
                        scaler.unscale_(optimizer)
                
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

                    if mixed_precision:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
            
                    optimizer.zero_grad()

                if learning_rate_schedule != "constant":
                    scheduler.step()

                metrics = {"loss": loss.item()}
                batches.set_postfix(metrics)

                if dry_run:
                    break
            
                wandb.log(metrics)

            model.save_pretrained(os.path.join(model_path, f"epoch-{epoch}"))

            if dry_run:
                break

    except KeyboardInterrupt:
        print("Exiting training early...")
    
    model.save_pretrained(model_path)
    dataset.tokenizer.save_pretrained(model_path)

if __name__ == "__main__":
    typer.run(train)
