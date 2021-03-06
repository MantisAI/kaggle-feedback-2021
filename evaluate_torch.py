from transformers import AutoModelForTokenClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
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

LABEL_LEN_THRESHOLD = {
    "Lead": 9,
    "Position": 5,
    "Evidence": 14,
    "Claim": 3,
    "Concluding Statement": 11,
    "Counterclaim": 6,
    "Rebuttal": 4,
}

# from Rob Mulla @robikscube
# https://www.kaggle.com/robikscube/student-writing-competition-twitch
def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    set_pred = set(row.predictionstring_pred.split(" "))
    set_gt = set(row.predictionstring_gt.split(" "))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter / len_pred
    return [overlap_1, overlap_2]


def score_feedback_comp(pred_df, gt_df):
    """
    A function that scores for the kaggle
        Student Writing Competition

    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = (
        gt_df[["id", "discourse_type", "predictionstring"]]
        .reset_index(drop=True)
        .copy()
    )
    pred_df = pred_df[["id", "class", "predictionstring"]].reset_index(drop=True).copy()
    pred_df["pred_id"] = pred_df.index
    gt_df["gt_id"] = gt_df.index
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(
        gt_df,
        left_on=["id", "class"],
        right_on=["id", "discourse_type"],
        how="outer",
        suffixes=("_pred", "_gt"),
    )
    joined["predictionstring_gt"] = joined["predictionstring_gt"].fillna(" ")
    joined["predictionstring_pred"] = joined["predictionstring_pred"].fillna(" ")

    joined["overlaps"] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined["overlap1"] = joined["overlaps"].apply(lambda x: eval(str(x))[0])
    joined["overlap2"] = joined["overlaps"].apply(lambda x: eval(str(x))[1])

    joined["potential_TP"] = (joined["overlap1"] >= 0.5) & (joined["overlap2"] >= 0.5)
    joined["max_overlap"] = joined[["overlap1", "overlap2"]].max(axis=1)
    tp_pred_ids = (
        joined.query("potential_TP")
        .sort_values("max_overlap", ascending=False)
        .groupby(["id", "predictionstring_gt"])
        .first()["pred_id"]
        .values
    )

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined["pred_id"].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query("potential_TP")["gt_id"].unique()
    unmatched_gt_ids = [c for c in joined["gt_id"].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    # calc microf1
    my_f1_score = TP / (TP + 0.5 * (FP + FN))
    return my_f1_score


class TestDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data.iloc[idx]["text"].split()
        inputs = self.tokenizer(tokens, padding="max_length", truncation=True, max_length=self.max_length, is_split_into_words=True)
        return {k: torch.tensor(v) for k,v in inputs.items()}


def align_preds_words(preds, inputs):
    aligned_preds = []
    for i, pred in enumerate(preds):
        word_ids = inputs["input_ids"][i]
    
        aligned_pred = []
        previous_word_id = None
        for word_idx, word_id in enumerate(word_ids):
            # When we reach a pad token, sequence has ended
            if word_id == 1:
                break

            word_pred = pred[word_idx]
        
            # Only for first word
            if not previous_word_id:
                aligned_pred.append(word_pred)
            # Same subword
            elif previous_word_id == word_id:
                pass
            # Start of new subword 
            else:
                aligned_pred.append(word_pred)
            
            previous_word_id = word_id
        aligned_preds.append(aligned_pred)
    return aligned_preds


def convert_to_submission_data(preds, test_data):
    id2label = {i: label for i, label in enumerate(LABELS)}
    
    pred_data = []
    for i, pred in enumerate(preds):
        words = test_data.iloc[i]["text"].split()
        doc_id = test_data.iloc[i]["id"]
    
        entity_word_ids = []
        entity_words = []
        for word_idx, word in enumerate(words):
            label = id2label[pred[word_idx]]
            if entity_words and (label[0] in ["O", "B"] or (label[0] == "I" and label[2:] != entity_label)):
                if len(entity_words) >= LABEL_LEN_THRESHOLD[entity_label]:
                    entity_text = " ".join(entity_words)
                    predictionstring = " ".join(entity_word_ids)
                    pred_data.append({"id": doc_id, "class": entity_label, "predictionstring": predictionstring, "entity_text": entity_text})
                entity_words = []
                entity_word_ids = []
            elif label[0] == "O":
                pass
            else:
                _, label_name = label.split("-")
                entity_words.append(word)
                entity_word_ids.append(str(word_idx))
                entity_label = label_name

    return pd.DataFrame(pred_data)


def evaluate(model_path, data_path, competition_data_path, pretrained_model="allenai/longformer-base-4096", batch_size:int=32, max_length:int=1024, dry_run:bool=False):
    # WARNING: Need to split train_NER to train_data.csv and train.csv to test_data.csv
    test_data = pd.read_csv(data_path) 
    test_data = test_data[test_data["kfold"]==1]

    competition_data = pd.read_csv(competition_data_path)
    competition_data = competition_data.merge(test_data, on="id")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, add_prefix_space=True)
    dataset = TestDataset(test_data, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = AutoModelForTokenClassification.from_pretrained(pretrained_model, num_labels=15)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    
    aligned_preds = []
    for inputs in tqdm(dataloader):
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs.logits, dim=2).cpu().numpy()
        
        aligned_preds.extend(align_preds_words(preds, inputs))

        if dry_run:
            break
    
    pred_data = convert_to_submission_data(aligned_preds, test_data)

    f1 = score_feedback_comp(pred_data, competition_data.merge(pred_data[["id"]].drop_duplicates()))
    print(f1)

if __name__ == "__main__":
    typer.run(evaluate)
