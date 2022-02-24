import pandas as pd 
import torch


def inference(batch, model):
    global config, ids_to_labels   
    # MOVE BATCH TO GPU AND INFER
    ids = batch["input_ids"].to(config['device'])
    mask = batch["attention_mask"].to(config['device'])
    outputs = model(ids, attention_mask=mask, return_dict=False)
    all_preds = torch.argmax(outputs[0], axis=-1).cpu().numpy() 

    # INTERATE THROUGH EACH TEXT AND GET PRED
    predictions = []
    for k,text_preds in enumerate(all_preds):
        token_preds = [ids_to_labels[i] for i in text_preds]

        prediction = []
        word_ids = batch['wids'][k].numpy()  
        previous_word_idx = -1
        for idx,word_idx in enumerate(word_ids):                            
            if word_idx == -1:
                pass
            elif word_idx != previous_word_idx:              
                prediction.append(token_preds[idx])
                previous_word_idx = word_idx
        predictions.append(prediction)
    
    return predictions

def get_predictions(model, df, loader):
    global config
    # put model in training mode
    model.eval()
    
    # GET WORD LABEL PREDICTIONS
    y_pred2 = []
    for batch in loader:
        try:
            labels = inference(batch)
            y_pred2.extend(labels)
        except Exception as e:
            pass

    final_preds2 = []
    for i in range(len(df)):
        idx = df.id.values[i]
        #pred = [x.replace('B-','').replace('I-','') for x in y_pred2[i]]
        pred = y_pred2[i] # Leave "B" and "I"
        j = 0
        while j < len(pred):
            cls = pred[j]
            if cls == 'O': j += 1
            else: cls = cls.replace('B','I') # spans start with B
            end = j + 1
            while end < len(pred) and pred[end] == cls:
                end += 1
            
            if cls != 'O' and cls != '' and end - j > config['min_entity_length']:
                final_preds2.append((idx, cls.replace('I-',''),
                                    ' '.join(map(str, list(range(j, end))))))
        
            j = end
        
    oof = pd.DataFrame(final_preds2)
    oof.columns = ['id','class','predictionstring']

    return oof