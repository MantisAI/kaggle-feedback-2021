from sklearn.metrics import accuracy_score
import torch
import gc
from tqdm import tqdm
from transformers import AdamW
from torch.nn import CrossEntropyLoss

import evaluation


def train_epoch(epoch, model, optimizer, scaler, training_loader, config, compute_accuracy=True):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    loss_ftc = None
    if config['loss_weights'] is not None:
        loss_ftc = CrossEntropyLoss(weight=torch.tensor(config['loss_weights']).to(config['device']))
    
    
    # put model in training mode
    model.train()

    loop = tqdm(training_loader, leave=True)
    idx = 0
    for batch in loop:
        
        optimizer.zero_grad()

        ids = batch['input_ids'].to(config['device'], dtype = torch.long)
        mask = batch['attention_mask'].to(config['device'], dtype = torch.long)
        labels = batch['labels'].to(config['device'], dtype = torch.long)

        loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels,
                               return_dict=False)
        if loss_ftc is not None:
            loss = loss_ftc(tr_logits.view(-1, model.num_labels).to(config['device']), labels.view(-1).to(config['device']))
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)
        
        if compute_accuracy:
            # compute training accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
            #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
            
            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            #tr_labels.extend(labels)
            #tr_preds.extend(predictions)

            tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            tr_accuracy += tmp_tr_accuracy
    
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=config['max_grad_norm']
        )
        
        # backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        idx+=1
        loop.set_description(f'Epoch {epoch}')
        if compute_accuracy:
            loop.set_postfix(loss=loss.item(), accuracy=tmp_tr_accuracy)
        else:
            loop.set_postfix(loss=loss.item())


    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    if compute_accuracy:
        print(f"Training accuracy epoch: {tr_accuracy}")

def train_model(model, config, training_loader, train_df, valid_idx, testing_set, test_dataset, IDS, test_params):
    model.to(config['device'])
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['learning_rates'][0])
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config['epochs']):
        print(f"### Training epoch: {epoch + 1}")
        for g in optimizer.param_groups: 
            g['lr'] = config['learning_rates'][epoch]
        lr = optimizer.param_groups[0]['lr']
        print(f'### LR = {lr}\n')
        
        train_epoch(epoch, model, optimizer, scaler, training_loader, config, compute_accuracy=True)
        torch.cuda.empty_cache()
        gc.collect()

        model.eval()
        eval_score = evaluation.evaluate_model(
            model,
            train_df, 
            valid_idx,
            testing_set, 
            test_dataset, 
            IDS, 
            test_params
        )

        model.train()
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Eval Score = {eval_score}")
        torch.save(model.state_dict(), f'longformer_{epoch}.pt')
