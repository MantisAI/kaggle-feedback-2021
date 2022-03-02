import pandas as pd 
import torch
from tqdm import tqdm

map_clip = {'Lead':9, 'Position':5, 'Evidence':14, 'Claim':3, 'Concluding Statement':11,
             'Counterclaim':6, 'Rebuttal':4}

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

def get_predictions(model, df, loader, display_tqdm=False):    
    # put model in training mode
    model.eval()
    
    # GET WORD LABEL PREDICTIONS
    final_preds2 = []
    i=0
    if display_tqdm:
        loop = tqdm(loader, leave=True)
    else:
        loop = loader
    for batch in loop:
        labels = inference(model, batch)
        
        # try and remove some blank spaces
        try:
            for y in labels:
                last_non_o = None
                last_non_o_pos = 0
                ii = 0
                while ii < len(y):
                    if y[ii]!='O':
                        if last_non_o:
                            if ii-last_non_o_pos < 3 and y[ii][0]=='I' and last_non_o[2:]==y[ii][2:]:#and last_non_o[0]=='I'

                                for j in range(last_non_o_pos+1, ii):
                                    y[j] = y[ii]
                            last_non_o = y[ii]
                            last_non_o_pos = ii
                        else:
                            last_non_o = y[ii]
                            last_non_o_pos = ii
                        ii+=1
                    else:
                        while ii<len(y) and y[ii]=='O':
                            ii+=1
        except Exception as e:
            print(e)
            pass
        
        for ii in range(len(labels)):
            idx = df.id.values[i]
            
            pred = labels[ii]
            j = 0
            while j < len(pred):
                cls = pred[j]
                if cls == 'O': j += 1
                else: cls = cls.replace('B','I') # spans start with B
                end = j + 1
                while end < len(pred) and pred[end] == cls:
                    end += 1
                
                if cls != 'O' and cls != '':
                    current_class = cls[2:]
                    if end - j >= map_clip[current_class]:
                        final_preds2.append((idx, cls.replace('I-',''),
                                             ' '.join(map(str, list(range(j, end))))))

                j = end
                
            i+=1
        
        if display_tqdm:
            loop.update()
        
        
    oof = pd.DataFrame(final_preds2)
    oof.columns = ['id','class','predictionstring']

    return oof